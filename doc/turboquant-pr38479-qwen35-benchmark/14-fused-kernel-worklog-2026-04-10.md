# TurboQuant 融合算子实验工作日志

日期：`2026-04-10`

## 实验目标

围绕 `Qwen3.5-35B-A3B`、`vLLM 0.19.0` 和 `turboquant-vllm`，完成以下工作：

- 跑通 `tq-fused` 非 `eager` 服务路径
- 对比 `baseline-bf16`、`baseline-fp8`、`tq-unfused`、`tq-fused`
- 统计性能和 KV cache 容量差异

## 固定环境

- 远端机器：`10.90.24.4`
- 登录用户：`guipeng`
- GPU 规划：`gpu4-7`
- 官方镜像：`vllm/vllm-openai:v0.19.0-x86_64-cu130`
- 模型：`/share/models/official/Qwen3.5-35B-A3B`
- TurboQuant 源码：`/ceph/User/E01442/turboquant/turboquant-vllm`

## 预设容器与端口

| 配置 | GPU | 端口 | 容器名 |
| --- | ---: | ---: | --- |
| `baseline-bf16` | `4` | `8050` | `zgp-vllm019-fusedstudy-bf16` |
| `baseline-fp8` | `5` | `8051` | `zgp-vllm019-fusedstudy-fp8` |
| `tq-unfused` | `6` | `8052` | `zgp-vllm019-fusedstudy-tq-unfused` |
| `tq-fused` | `7` | `8053` | `zgp-vllm019-fusedstudy-tq-fused` |

## 执行记录

### 18:xx 设计与计划落盘

- 新增设计文档：`12-fused-kernel-benchmark-design-2026-04-10.md`
- 新增执行计划：`13-fused-kernel-execution-plan-2026-04-10.md`
- 确认代码路径：
  - `TQ4_USE_FUSED_PAGED=1` 才会进入 `_fused_decode_path`
  - `tq-unfused` 仍走 `paged decompress + flash_attn_varlen_func`
  - `TQ4MetadataBuilder.get_cudagraph_support()` 只在 fused 可用且 `k_bits == v_bits` 时返回 `UNIFORM_SINGLE_TOKEN_DECODE`
  - `paged decompress` 仍然存在动态 fallback 分支

### 待补充

- 单卡 `tq-fused` 非 eager 启动结果
- 四组服务正式启动命令
- smoke benchmark 结果
- full benchmark 结果
- 清理记录

### 19:05 前的首轮启动观察

- 首轮四组服务全部按 `enforce_eager=False` 启动，统一参数为：
  - `max_model_len=16384`
  - `gpu_memory_utilization=0.90`
  - 模型路径 `/share/models/official/Qwen3.5-35B-A3B`
- `baseline-bf16` 和 `baseline-fp8` 第一次启动时，直接把 `vllm serve ...` 作为镜像 command 传入，和官方镜像默认 entrypoint 叠加，导致容器立即退出。随后改成 `--entrypoint bash -lc "vllm serve ..."` 重新拉起。
- `tq-unfused` 第一次启动成功进入 `AttentionBackendEnum.CUSTOM`，随后完成权重加载并进入 `torch.compile` 阶段。
- `tq-fused` 第一次启动也成功进入非 eager 配置，日志中可见：
  - `enforce_eager=False`
  - `compilation_config ... cudagraph_mode=FULL_AND_PIECEWISE`
  - 完成权重加载和一次 `torch.compile`
- 但首轮 `tq-fused` 容器在尚未 ready 前自行退出，`gpu7` 显存回落到空闲。由于首轮容器带了 `--rm`，退出后无法直接保留完整崩溃日志，因此立刻改为第二轮 debug 复现：
  - 新容器名：`zgp-vllm019-fusedstudy-tq-fused-debug`
  - 去掉 `--rm`，保留退出后的完整 `docker logs`
- 截至 `2026-04-10 19:05:35 +0800`，`baseline-bf16`、`baseline-fp8`、`tq-unfused` 三组都已完成模型加载并完成一轮 `torch.compile`，但 `/v1/models` 尚未 ready；这说明当前 `Qwen3.5-35B-A3B + vLLM 0.19.0` 的非 eager 首次 warmup 本身就较慢，不能把 `tq-fused` 的退出简单归因于“启动慢”。

### 非 eager fused 失败根因

第二轮 `zgp-vllm019-fusedstudy-tq-fused-debug` 去掉了 `--rm`，完整保留了崩溃日志。最终根因已经定位为：

- `torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing`
- 随后在 `torch.cuda.graph(...).capture_end()` 处触发 `cudaErrorStreamCaptureInvalidated`
- 上层表现为 `RuntimeError: Engine core initialization failed`

调用链落点在：

- `gpu_worker.py -> determine_available_memory()`
- `gpu_model_runner.py -> profile_cudagraph_memory()`
- `gpu_model_runner.py -> _warmup_and_capture()`
- `torch.cuda.graph(...).capture_end()`

结论很明确：当前 `turboquant-vllm 1.5.0 + vLLM 0.19.0 + Qwen3.5-35B-A3B + A100` 组合下，`TQ4_USE_FUSED_PAGED=1` 的非 eager 路径仍然不能安全参与 CUDA graph capture。这个现象和作者仓库旧实验里“fused + CUDA graphs blocked”的结论一致。

### 非 eager ready 状态

随后三条非 eager 服务都先后 ready：

- `baseline-bf16` on `8050`
- `baseline-fp8` on `8051`
- `tq-unfused` on `8052`

`tq-fused` 的非 eager 版本没有任何一次成功 ready。

### KV cache 对比

从服务日志提取到的 KV cache 指标如下：

| 配置 | Available KV cache memory | GPU KV cache size | 16K/request 最大并发 |
| --- | ---: | ---: | ---: |
| `baseline-bf16` | `3.74 GiB` | `48,576 tokens` | `9.74x` |
| `baseline-fp8` | `3.74 GiB` | `96,416 tokens` | `17.00x` |
| `tq-unfused` | `3.76 GiB` | `97,152 tokens` | `16.91x` |

阶段性结论：

- TurboQuant 在当前实现上的实际 cache 容量与 `fp8` 几乎持平，只比 `fp8` 多 `736 tokens`
- 相对 `bf16`，`fp8` 和 `tq-unfused` 都接近 `2x` cache 容量
- 这说明当前 `turboquant-vllm` 在 vLLM allocator 里的主要 cache 优势不是显著超过 `fp8`，而是“基本达到 fp8 水平”

### Smoke 1: 偏 decode 负载

参数：

- `num_prompts=64`
- `input_len=1024`
- `output_len=512`
- `request_rate=inf`
- `temperature=0`

结果如下：

| 配置 | req/s | out tok/s | total tok/s | median TTFT | median TPOT | median ITL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-bf16` | `2.018` | `1033.175` | `3099.524` | `3076.14 ms` | `32.02 ms` | `25.36 ms` |
| `baseline-fp8` | `2.187` | `1119.956` | `3359.868` | `2632.74 ms` | `31.99 ms` | `28.33 ms` |
| `tq-unfused` | `1.508` | `771.978` | `2315.934` | `3094.08 ms` | `44.78 ms` | `39.57 ms` |

这组结果表明，在偏 decode 的负载下：

- `fp8` 明显优于 `bf16`
- `tq-unfused` 明显慢于 `fp8` 和 `bf16`
- `tq-unfused` 的 `median TPOT` 相对 `fp8` 恶化约 `40%`

### Smoke 2: 长上下文负载

参数：

- `num_prompts=32`
- `input_len=12000`
- `output_len=256`
- `request_rate=inf`
- `temperature=0`

结果如下：

| 配置 | req/s | out tok/s | total tok/s | median TTFT | median TPOT | median ITL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-bf16` | `0.812` | `207.793` | `9948.059` | `18300.28 ms` | `51.94 ms` | `20.90 ms` |
| `baseline-fp8` | `0.814` | `208.383` | `9976.073` | `16071.44 ms` | `75.05 ms` | `23.72 ms` |
| `tq-unfused` | `0.474` | `121.470` | `5815.380` | `29656.00 ms` | `117.89 ms` | `45.76 ms` |

这组长上下文结果没有体现出 TurboQuant 相对 `fp8` 的收益，反而进一步暴露了解压后 attention 路径的代价：

- `tq-unfused` 的 `median TTFT` 比 `fp8` 慢约 `84%`
- `tq-unfused` 的 `median TPOT` 比 `fp8` 慢约 `57%`
- `tq-unfused` 的整体 token throughput 也显著落后

### Eager 对照线

为了把“融合 kernel 本身”和“非 eager/cudagraph 失败”拆开，新增了一条 GPU7 eager 对照线：

- `zgp-vllm019-fusedstudy-tq-unfused-eager`

当前状态：

- 已确认 `--enforce-eager` 生效
- `torch.compile` 与 `cudagraph` 均被禁用
- 最终两条 eager 服务都成功 ready

短序列 eager A/B 参数：

- `num_prompts=64`
- `input_len=1024`
- `output_len=512`
- `request_rate=inf`
- `temperature=0`

结果如下：

| 配置 | req/s | out tok/s | total tok/s | median TTFT | median TPOT | median ITL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tq-unfused-eager` | `0.582` | `298.085` | `894.254` | `3554.04 ms` | `108.47 ms` | `100.77 ms` |
| `tq-fused-eager` | `0.485` | `248.377` | `745.131` | `3343.04 ms` | `114.96 ms` | `101.60 ms` |

阶段性结论：

- 在当前 eager 对照下，`tq-fused` 没有跑赢 `tq-unfused`
- `tq-fused-eager` 的输出吞吐比 `tq-unfused-eager` 低约 `16.7%`
- `tq-fused-eager` 的 `median TPOT` 比 `tq-unfused-eager` 更差
- `tq-fused-eager` 仅在 `median TTFT` 上有轻微改善，但不足以抵消 decode 吞吐下降

### 收尾清理

本轮实验完成后，已在 `10.90.24.4` 上清理以下容器：

- `zgp-vllm019-fusedstudy-bf16`
- `zgp-vllm019-fusedstudy-fp8`
- `zgp-vllm019-fusedstudy-tq-unfused`
- `zgp-vllm019-fusedstudy-tq-fused-eager`

清理后再次检查，`.4` 上已无 `zgp-vllm019-fusedstudy-*` 残留容器。

### 22:23 后的第二轮修复

在第一轮结论的基础上，继续对 `turboquant-vllm` 做了两条代码修复：

1. 把 `_padded_slot_bytes()` 从 `next_power_of_2(raw_slot)` 改回真实 packed slot；
2. 在 `register_tq4_backend()` 中 patch `HybridAttentionMambaModelConfig.verify_and_update_config()`，让 hybrid 模型在配置阶段就按 TQ4 的真实 page bytes 计算 attention block size。

同时，为了让 fused 路径先稳定跑起来，又把 `TQ4MetadataBuilder.get_cudagraph_support()` 暂时改成保守的 `NEVER`，避免在 A100 + vLLM 0.19 上因为 CUDA graph capture 直接启动失败。

### 第二轮单测

使用官方 `v0.19.0-x86_64-cu130` 镜像，在容器内补装 `pytest` 后执行：

- `tests/test_vllm_registration.py`
- `tests/test_vllm_cache.py`
- `tests/test_vllm_cache_cudagraph.py`

结果：`59 passed`

### 第二轮 KV cache 复测

重新启动：

- `zgp-vllm019-fusedstudy-fp8-r2`
- `zgp-vllm019-fusedstudy-tq-unfused-r2`
- `zgp-vllm019-fusedstudy-tq-fused-r3`

服务端日志显示：

| 配置 | block_size | GPU KV cache size | 16K/request 最大并发 |
| --- | ---: | ---: | ---: |
| `fp8-r2` | `2096` | `96,416 tokens` | `17.00x` |
| `tq-unfused-r2` | `4080` | `187,680 tokens` | `23.38x` |
| `tq-fused-r3` | `4080` | `187,680 tokens` | `23.38x` |

这一步确认两件事：

- 第一轮“KV cache 只和 fp8 持平”的问题已经修掉
- 修复后 TurboQuant 的实际 cache 容量约为 `1.95x fp8`

### 第二轮长上下文高压力 benchmark

参数：

- `num_prompts=64`
- `input_len=12000`
- `output_len=256`
- `request_rate=inf`
- `temperature=0`

结果如下：

| 配置 | req/s | out tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: |
| `fp8-r2b` | `0.832` | `212.973` | `35568.76 ms` | `86.25 ms` |
| `tq-unfused-r2` | `0.434` | `111.049` | `92514.84 ms` | `168.00 ms` |
| `tq-fused-r3` | `0.438` | `112.173` | `91007.70 ms` | `167.68 ms` |

修复后的结论：

- cache 容量已经显著大于 `fp8`
- 但性能仍然远落后于 `fp8`
- `tq-fused-r3` 只比 `tq-unfused-r2` 略好约 `1%`
- 当前 fused 实现距离“显著强于 fp8 flash-attn”仍有明显差距

### 2026-04-14: Qwen3 非 hybrid 复测

为了验证 `tq-fused + cudagraph` 的启动失败是否主要由 `Qwen3.5` 的 hybrid 架构触发，做了一轮额外复测：

- 先把 `turboquant-vllm` 本地代码改动通过 `git stash push -u -m codex-qwen3-rerun-2026-04-14` 收起，恢复到原始 `v1.5.0` 状态
- 使用 `.4` 的空闲 `gpu2/gpu3`
- 模型切换为 `/share/models/official/Qwen3-32B-FP8`
- 服务命名：
  - `zgp-vllm019-qwen3-unfused` on `gpu3`, port `8063`
  - `zgp-vllm019-qwen3-fused-debug` on `gpu2`, port `8062`

关键信息：

- `Qwen3-32B-FP8` 的架构是 `Qwen3ForCausalLM`，不是 Qwen3.5 那种 hybrid `Qwen3_5MoeForConditionalGeneration`
- `zgp-vllm019-qwen3-unfused` 可以正常启动并返回 `/v1/models`
- 对 `8063` 发送 `/v1/completions` 短请求，可以正常生成
- `zgp-vllm019-qwen3-fused-debug` 依旧在非 eager 初始化阶段失败

这次 `Qwen3` dense 模型上的 fused 错误与 `Qwen3.5` 上的根因一致，仍然发生在：

- `gpu_worker.py -> determine_available_memory()`
- `gpu_model_runner.py -> profile_cudagraph_memory()`
- `torch.cuda.graph(...).capture_end()`

错误要点仍然是：

- `torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing`
- `cudaErrorStreamCaptureInvalidated`
- 上层 `RuntimeError: Engine core initialization failed`

这说明：

- `Qwen3.5` hybrid 架构不是 fused+cudagraph 失败的唯一根因
- fused 路径当前在更普通的 dense `Qwen3` 模型上也会因 CUDA graph capture 而失败
- hybrid 架构会显著影响 KV cache page-size / block-size 计算，但并不能单独解释 fused+cudagraph 的启动失败

复测结束后，已清理：

- `zgp-vllm019-qwen3-unfused`
- `zgp-vllm019-qwen3-fused-debug`
