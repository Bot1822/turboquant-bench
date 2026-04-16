# Qwen3 TurboQuant fused CUDA graph 调试记录

日期：`2026-04-14`

## 目标

本轮调试聚焦 `turboquant-vllm` 的 fused paged TQ4 decode 路径在 vLLM 0.19 非 eager 模式下无法通过 CUDA graph capture 的问题。目标不是通过关闭 CUDA graph 或改用 eager 绕过问题，而是定位 fused wrapper 或 Triton kernel 中的非 capture-safe 操作，修复后让 `TQ4_USE_FUSED_PAGED=1` 能在 Qwen3 服务上正常完成 cudagraph 初始化，并在修复后跑一次 `lm_eval` 抽样能力回归。

## 固定环境

模型使用 `/share/models/official/Qwen3-32B-FP8`。服务优先放在 `guipeng@10.90.24.4`，因为检查时 GPU2/GPU3 均为空闲状态，显存占用为 `17 MiB` 且 GPU 利用率为 `0%`。官方镜像继续使用 `vllm/vllm-openai:v0.19.0-x86_64-cu130`，容器名称必须带 `zgp-` 前缀。客户端和单测优先使用同一个官方镜像挂载本地 `turboquant-vllm` 源码。

## 调试策略

调试分三层进行。第一层是最小 CUDA graph 单测，直接对 fused paged TQ4 wrapper 做 warmup、capture 和 replay，确认失败是否来自 wrapper 内部的 Python tensor allocation、Triton autotune/JIT 或 kernel launch 本身。第二层是 vLLM 服务启动验证，使用 Qwen3、`TQ4_USE_FUSED_PAGED=1`、非 eager 和 `--attention-backend CUSTOM`，确认是否能通过 `profile_cudagraph_memory()` 与 `_warmup_and_capture()`。第三层是修复后的抽样 `lm_eval`，只做 smoke/regression，不替代之前的全量能力评测。

## 执行记录

### 选卡

本机 GPU 状态：GPU3/GPU7 空闲，GPU3 为 `17 MiB/0%`，GPU7 为 `17 MiB/0%`。远端 `.4` GPU 状态：GPU2/GPU3 空闲，均为 `17 MiB/0%`；GPU0/GPU1/GPU4-GPU7 已有高显存占用。按项目 GPU selection runbook，本轮选择 `.4` 的 GPU2/GPU3。

### 初始代码状态

根仓有历史文档和结果未提交，`turboquant-vllm` 工作树干净。`turboquant-vllm` 中存在一个历史 stash：`stash@{0}: On main: codex-qwen3-rerun-2026-04-14`。该 stash 包含 KV cache 真实 packed slot 修复、hybrid block-size patch、以及临时把 `TQ4MetadataBuilder.get_cudagraph_support()` 改为 `NEVER` 的绕过方案。本轮将恢复其中的结构性修复作为基础，但不会接受“关闭 cudagraph”作为最终修复。

### 根因定位

第一层最小 CUDA graph 探针显示，旧 fused wrapper 在“先 warmup 再 capture”的场景里可以通过；这说明问题不是 Triton kernel 每次 launch 都天然不支持 CUDA graph。进一步探针把 warmup 去掉后，第一次 captured fused decode 先失败在 `q_rot = torch.matmul(q.float(), rotation.T)`，报 `CUBLAS_STATUS_NOT_INITIALIZED` 并导致 `cudaErrorStreamCaptureInvalidated`。这说明 vLLM 默认 `cudagraph_num_of_warmups=0` 时，首次进入 capture 可能触发 cuBLAS handle 初始化，而该初始化不是 capture-safe。

随后只在 capture 前预热一次 rotation matmul，但不预热 fused wrapper。失败点移动到 Triton autotune：`triton.runtime.autotuner -> triton.testing.do_bench -> torch.cuda.synchronize()`，错误为 `operation not permitted when stream is capturing`。因此本轮根因分成两部分：第一，fused wrapper 在 capture 内首次使用 cuBLAS rotation matmul；第二，`_fused_paged_tq4_decode_kernel` 使用 `@triton.autotune`，首次 shape 进入 capture 时会执行 benchmark/synchronize。

### 代码修复

修复集中在两个文件。`fused_paged_tq4_attention.py` 去掉 runtime `@triton.autotune`，改成固定 launch config：`BLOCK_N=64`、`num_stages=3`、`num_warps=4`，避免 capture 内 benchmark。该 wrapper 同时增加可选的预分配 scratch 参数，包括 fp32 query cast、fp32 q rotation、compute-dtype q rotation、rotated output、fp32 post-rotation input/output，避免 captured decode 中反复创建临时 tensor。

`tq4_backend.py` 恢复 `TQ4MetadataBuilder.get_cudagraph_support()` 的 conditional 逻辑：当 `TQ4_USE_FUSED_PAGED=1`、fused kernel 可导入且 `k_bits == v_bits` 时返回 `UNIFORM_SINGLE_TOKEN_DECODE`，否则返回 `NEVER`。backend 在 fused 可用且设备为 CUDA 时，会在模型构建阶段做一次很小的 fp32 rotation matmul，用于提前初始化 cuBLAS。`_init_cg_buffers()` 新增 `_cg_q_fp32` 和 `_cg_fused_out_rot`，`_fused_decode_path()` 调用 fused wrapper 时传入这些 scratch。

### 单测验证

新增 `tests/test_fused_paged_tq4_cudagraph.py`，覆盖两种场景：一是提供预分配 scratch 后 warmup/capture/replay 应正常完成；二是只预热 cuBLAS、不预热 fused kernel 时，首次 fused kernel launch 发生在 capture 内也不应触发 Triton autotune。修复前第一条测试因 wrapper 不支持 scratch 参数失败；进一步手工探针确认旧实现会分别卡在 cuBLAS 初始化和 Triton autotune。

修复后在 `.4` 的 GPU2 上使用官方镜像 `vllm/vllm-openai:v0.19.0-x86_64-cu130` 执行：

```bash
pytest -q tests/test_fused_paged_tq4_cudagraph.py tests/test_vllm_cache_cudagraph.py -q
```

结果为 `20 passed`。继续执行：

```bash
pytest -q tests/test_fused_paged_tq4_attention.py tests/test_fused_paged_tq4_composition.py tests/test_vllm_registration.py tests/test_vllm_cache.py tests/test_vllm_fused_gating.py -q
```

结果为全部通过，共 `77 passed`。这轮验证说明固定 launch config 没有破坏 fused paged TQ4 的正确性，KV cache 相关修复和 backend 注册逻辑也保持通过。

### batched decode 与服务层迭代

第一次服务重试表明，cuBLAS 初始化和 Triton autotune 两个问题修掉后，vLLM 仍然会在 cudagraph profiling 阶段进入 `_tq4_prefill()`。根因不是 `FULL_DECODE_ONLY` 没生效，而是 backend 把 decode 错误判定成了 `num_actual_tokens == 1`。对 vLLM 来说，uniform single-token decode batch 中 `num_actual_tokens` 等于 batch size，本轮实际触发值是 `256`。因此新增 batched decode 回归测试，把判定改为 `attn_metadata.max_query_len == 1`，并把 fused decode 的 query/compress/output scratch 改为按 `max_cudagraph_capture_size` 预分配。

第二次服务重试时，服务已经能真正进入 `_fused_decode_path()`，但又暴露出另一个实现问题：`tq4_compress(out=...)` 在 GPU 路径上把预分配 `out` 当作扁平 `M=N*H` 形状使用，而 `_compress_and_store()` 直接把整块 scratch 返回值 `reshape(N, -1)`，在当前 batch 小于 scratch capacity 时把多余行也吞了进去，形成 `512` 对 `1024` 的列数不匹配。为此新增“active slice”单测，把 `_compress_and_store()` 调整为只把当前 `N*H` 范围的 active slice 传给压缩核，同时让 `tq4_compress` 的 CPU fallback 也兼容扁平 out 形状。

修复后，又额外通过了一轮聚焦 batched decode 的回归：

```bash
pytest -q tests/test_vllm_cache_cudagraph.py tests/test_vllm_fused_gating.py tests/test_fused_paged_tq4_cudagraph.py -q
```

最终结果为 `50 passed`。

### Qwen3 服务启动结果

在 `.4` 的 GPU2 上重新拉起 `zgp-vllm019-qwen3-fused-cgfix` 后，服务成功走完了之前一直失败的 cudagraph profiling 阶段，日志关键信号包括：

- `Profiling CUDA graph memory: FULL=35 (largest=256)`
- `Estimated CUDA graph memory: 6.72 GiB total`
- `Available KV cache memory: 37.19 GiB`
- `GPU KV cache size: 573,424 tokens`
- `Maximum concurrency for 16,384 tokens per request: 35.00x`

这说明 fused kernel 已经可以在 Qwen3 + vLLM 0.19 + 非 eager 下完成 CUDA graph capture，不再在 `_warmup_and_capture()` 阶段崩溃。随后对外接口验证也通过：

- `GET http://10.90.24.4:8062/v1/models` 正常返回
- `POST http://10.90.24.4:8062/v1/completions` 短请求正常生成

### lm-eval 抽样回归

服务 ready 后，继续沿用 `local-completions` 路径从本机连接 `.4:8062` 做抽样回归。过程中发现本机 `lm_eval` 是 uv tool 环境，最初缺少 `tenacity`、`openai`、`transformers`、`langdetect`、`immutabledict` 等 API / task 依赖，因此补齐了这些客户端依赖后才完成评测。

已跑通的抽样结果如下。

`leaderboard_mmlu_pro`：

- 命令参数：`limit=20`，`num_fewshot=5`，`num_concurrent=8`
- 结果：`acc=0.65 ± 0.1094`
- 用时：`96.53 s`

`leaderboard_ifeval`：

- 命令参数：`limit=10`，`num_concurrent=8`
- 结果：`prompt_level_strict_acc=0.4000 ± 0.1633`
- 结果：`prompt_level_loose_acc=0.4000 ± 0.1633`
- 结果：`inst_level_strict_acc=0.6667`
- 结果：`inst_level_loose_acc=0.6667`
- 用时：`98.60 s`

`leaderboard_math_hard` 抽样未继续完成。失败原因不是服务侧，而是本机 `lm_eval` uv 环境缺少 math task 依赖，错误明确要求补装 `math-verify`、`sympy>=1.12` 和 `antlr4-python3-runtime==4.11`。本轮在已经完成 `MMLU-Pro` 和 `IFEval` 两类抽样回归的前提下，没有继续扩张本机评测环境。

### 最终验证与清理

最终回归命令覆盖 fused kernel、CUDA graph、vLLM 注册/cache、paged decompress、fused gating 和 INT8 gating：

```bash
pytest -q tests/test_fused_paged_tq4_attention.py tests/test_fused_paged_tq4_composition.py tests/test_fused_paged_tq4_cudagraph.py tests/test_vllm_registration.py tests/test_vllm_cache.py tests/test_vllm_cache_cudagraph.py tests/test_vllm_paged_decompress.py tests/test_vllm_fused_gating.py tests/test_vllm_int8_gating.py -q
```

结果为全部通过，共 `135 passed`。验证结束后，`.4` 上的 `zgp-vllm019-qwen3-fused-cgfix` 容器已清理，不再占用 GPU2 和端口 `8062`。

## Qwen3 fused CUDA graph 性能复测

用户进一步要求在“修复 CUDA graph 之后，用融合算子，在 Qwen3 上再测一次”，因此新增本节，直接比较 `Qwen3-32B-FP8` 上的 `fp8 KV baseline` 和修复后的 `TQ fused + CUDA graph`。

实验约定如下：模型使用 `/share/models/official/Qwen3-32B-FP8`，镜像使用 `vllm/vllm-openai:v0.19.0-x86_64-cu130`。服务端在 `guipeng@10.90.24.4` 上启动，检查时 GPU2/GPU3 均为空闲，显存占用为 `17 MiB` 且利用率为 `0%`。`fp8 baseline` 使用 GPU2、端口 `8062`、容器名 `zgp-vllm019-qwen3-fp8-bench`，显式设置 `--kv-cache-dtype fp8`。`TQ fused` 使用 GPU3、端口 `8063`、容器名 `zgp-vllm019-qwen3-tqfused-bench`，设置 `TQ4_USE_FUSED_PAGED=1`、`TQ4_K_BITS=4`、`TQ4_V_BITS=4` 和 `--attention-backend CUSTOM`。两者都使用 `--max-model-len 16384`、`--gpu-memory-utilization 0.90`、非 eager。

benchmark 计划使用 `vllm bench serve` 的 `/v1/completions` 路径，从 `.4` 上的独立 client 容器发请求，保存 raw JSON 到 `doc/turboquant-pr38479-qwen35-benchmark/results/fused_kernel_v019/qwen3_cgfix_benchmark_20260414/`。第一组是偏 decode 负载：`num_prompts=64`、`random_input_len=1024`、`random_output_len=512`、`request_rate=inf`。第二组是长上下文负载：`num_prompts=64`、`random_input_len=12000`、`random_output_len=256`、`request_rate=inf`。

### Qwen3 baseline vs fused 结果

两条服务最终都在 `.4` 上成功 ready，并返回 `/v1/models`。从服务日志提取到的 KV cache 指标如下：

| 配置 | GPU | 端口 | GPU KV cache size | 16K/request 最大并发 |
| --- | ---: | ---: | ---: | ---: |
| `fp8 baseline` | `2` | `8062` | `304,672 tokens` | `18.60x` |
| `tq-fused + cudagraph` | `3` | `8063` | `573,424 tokens` | `35.00x` |

这里可以确认修复后的 Qwen3 fused 路径已经把 KV cache 优势真正兑现了：相对 `fp8 baseline`，TurboQuant fused 的 cache 容量达到约 `1.88x`，16K 请求最大并发也从 `18.60x` 提升到 `35.00x`。

第一组 benchmark 使用偏 decode 负载：`input_len=1024`、`output_len=512`、`num_prompts=64`。结果如下：

| 配置 | req/s | out tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: |
| `fp8 baseline` | `1.29` | `658.98` | `14,524.47 ms` | `68.01 ms` |
| `tq-fused + cudagraph` | `0.45` | `230.38` | `19,526.26 ms` | `237.62 ms` |

这一组说明 fused 修复并没有把 decode 性能推到接近 fp8。相对 `fp8 baseline`，TurboQuant fused 的 median TTFT 慢约 `34.4%`，median TPOT 慢约 `3.49x`，输出吞吐只有约 `35.0%`。

第二组 benchmark 使用长上下文负载。原计划是 `num_prompts=64`，但 fp8 baseline 的 `n=32` 结果已经耗时 `208.87 s`，因此为了控制总实验时间，最终采用更可执行的同条件对照：`input_len=12000`、`output_len=256`、`num_prompts=16`。结果如下：

| 配置 | req/s | out tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: |
| `fp8 baseline` | `0.16` | `41.46` | `52,526.01 ms` | `177.62 ms` |
| `tq-fused + cudagraph` | `0.06` | `16.25` | `60,862.79 ms` | `720.57 ms` |

这一组长上下文结果同样没有体现出 TurboQuant fused 在延迟或 decode 速度上的优势。相对 `fp8 baseline`，TurboQuant fused 的 median TTFT 慢约 `15.9%`，median TPOT 慢约 `4.06x`，输出吞吐只有约 `39.2%`。

阶段性结论很明确：在 Qwen3 上，修复 CUDA graph 之后，TurboQuant fused 已经具备稳定的 KV cache 容量优势，但端到端性能仍然明显落后于 `fp8 baseline`。也就是说，这次修复解决的是“能不能启动并参与 decode graph capture”的稳定性问题，而不是“是否已经在 TTFT / TPOT / 吞吐上赢过 fp8”的性能问题。

## 性能根因定位

在 Qwen3 `64Q / 8KV / head_dim=128` 的真实配置下，新增了组件级 microbenchmark 脚本 `scripts/profile_qwen3_decode_breakdown.py`，直接拆分 fused decode path 的关键成本：`compress_store`、`pre/post rotation`、`fused attention wrapper`、`kernel-only` 和整条 `_fused_decode_path()`。profiling 分别覆盖两组与线上 benchmark 对应的形状：

- 短负载：`batch=64, seq=1024`
- 长上下文：`batch=16, seq=12000`

稳定版 profiling 结果如下。

短负载 `batch=64, seq=1024`：

| 组件 | median |
| --- | ---: |
| `pre_rotation` | `0.055 ms` |
| `post_rotation` | `0.056 ms` |
| `compress_store` | `0.228 ms` |
| `fused_wrapper_total` | `2.305 ms` |
| `full_fused_decode_path` | `2.549 ms` |

长上下文 `batch=16, seq=12000`：

| 组件 | median |
| --- | ---: |
| `pre_rotation` | `0.055 ms` |
| `post_rotation` | `0.056 ms` |
| `compress_store` | `0.225 ms` |
| `fused_wrapper_total` | `7.101 ms` |
| `full_fused_decode_path` | `7.344 ms` |

这两组数据说明得很清楚：当前主瓶颈不是 rotation，也不是在线压缩，而是 fused attention kernel 本身。即使把 `compress_store` 和前后 rotation 全部视作“免费”，短负载的理论上限也只是从 `2.549 ms` 降到约 `2.21 ms`，长上下文的理论上限也只是从 `7.344 ms` 降到约 `7.01 ms`，不足以解释和 `fp8 baseline` 的巨大差距。

### fixed config sweep

因为为了 CUDA graph 安全移除了 Triton autotune，所以又补做了一轮 A100 上的 fixed config sweep，覆盖：

- `BLOCK_N in {32, 64}`
- `num_stages in {2, 3}`
- `num_warps in {4, 8}`

短负载 `batch=64, seq=1024` 的结果表明当前配置已经接近最优：

| 配置 | full fused decode median |
| --- | ---: |
| `64/2/4` | `2.540 ms` |
| `64/3/4` | `2.549 ms` |
| `32/2/4` | `2.825 ms` |
| `32/3/4` | `3.053 ms` |
| `64/2/8` | `3.424 ms` |
| `64/3/8` | `3.436 ms` |

长上下文 `batch=16, seq=12000` 下，`64/2/4` 也只是比原来的 `64/3/4` 快约 `0.3%`：

| 配置 | full fused decode median |
| --- | ---: |
| `64/2/4` | `7.322 ms` |
| `64/3/4` | `7.346 ms` |

因此，fixed config 不是主要矛盾，只能作为很小的低风险修正。最终把默认参数从 `64/3/4` 调整为 `64/2/4`。

### 已证伪的尝试

为验证是否存在低成本的大幅收益，又做了两条试验：

第一条是把 decode kernel 中的 `QK` 和 `PV` 从逐元素 `tl.sum` 改成 `tl.dot`。结果在 `BLOCK_M=1` 的 decode 场景下明显变慢，短负载 `full_fused_decode_path` 从约 `2.55 ms` 恶化到约 `14.31 ms`，因此已回退。

第二条是把 program 粒度从“每个 query head”改成“每个 KV head group”，希望在 GQA 下复用同一份解压后的 K/V。这条路的方向本身有价值，但在 Triton 当前写法下踩到了编译期索引限制，而且没有在稳定版上拿到可靠的正向收益，因此也已回退，没有并入主线。

### 结论

当前证据链已经比较完整：

1. `KV cache` 优势已经修出来，且在 Qwen3 上接近 `1.88x fp8`。
2. 端到端性能差距的主因在 attention kernel，而不是压缩或 rotation。
3. 关掉 autotune 后的固定参数选择只影响亚百分位到几个百分点，不是决定性因素。
4. 想要真正追平或超过 `fp8 baseline`，需要的是更激进的 decode kernel 重写，而不是继续在现有 kernel 周围做小修补。

### baseline attention 对照

为回答“baseline attention 本身是多少毫秒、和 TurboQuant fused 差多少”，又补跑了 vLLM attention benchmark helper。这里直接比较单层 decode attention 的端到端 `impl.forward()` 耗时，覆盖：

- `FLASH_ATTN`：dense KV cache attention
- `FLASHINFER`：`fp8` KV cache baseline 的真实 attention backend

两组和 Qwen3 线上服务对应的 batch spec 为：

- 短负载：`64q1s1k`
- 长上下文：`16q1s12000`

结果如下：

| 形状 | `FLASH_ATTN` | `FLASHINFER(fp8)` | `TQ fused` |
| --- | ---: | ---: | ---: |
| `64q1s1k` | `0.205 ms/layer` | `0.142 ms/layer` | `2.549 ms/layer` |
| `16q1s12000` | `0.505 ms/layer` | `0.420 ms/layer` | `7.344 ms/layer` |

按这个口径直接比较，当前 TurboQuant fused 相对 baseline attention 的差距是：

- `64q1s1k`：相对 `FLASH_ATTN` 慢约 `12.4x`，相对 `FLASHINFER(fp8)` 慢约 `17.9x`
- `16q1s12000`：相对 `FLASH_ATTN` 慢约 `14.5x`，相对 `FLASHINFER(fp8)` 慢约 `17.5x`

这一步把问题进一步钉死了：不是服务调度造成的轻微放大，而是 attention 路径本身就比 baseline backend 慢一个数量级。也就是说，当前 TurboQuant fused decode kernel 和成熟的 `FLASH_ATTN / FLASHINFER` backend 还不在同一个性能层级。
