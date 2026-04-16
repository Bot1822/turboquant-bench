# TurboQuant 融合算子测试报告

日期：`2026-04-10`

## 结论摘要

本轮在 `A100 80GB + vLLM 0.19.0 官方镜像 + Qwen3.5-35B-A3B` 上对 `turboquant-vllm` 做了两条线的验证。

第一条是主线，也就是用户优先要求的非 `eager` 路径。结果是：

- `baseline-bf16`、`baseline-fp8`、`tq-unfused` 都可以在 `enforce_eager=False` 下稳定启动并完成 benchmark。
- `tq-fused` 在 `TQ4_USE_FUSED_PAGED=1` 且非 `eager` 时仍然不能稳定启动，根因是 CUDA graph capture 阶段触发 `operation not permitted when stream is capturing`，随后 `Engine core initialization failed`。

第二条是兜底对照线，也就是在 `--enforce-eager` 下单独比较 `tq-fused` 和 `tq-unfused`。结果是：

- `tq-fused-eager` 可以启动并完成 benchmark。
- 但在当前短序列 decode-heavy 负载上，`tq-fused-eager` 没有比 `tq-unfused-eager` 更快，反而更慢。

因此，针对“当前社区里是否存在稳定、可直接用于后续复现和实验的 TurboQuant 融合算子 vLLM 实现”这个问题，本轮实测结论是：截至当前组合，`turboquant-vllm` 的 fused 路径仍然不具备稳定的非 `eager` 可用性；即便退回到 `eager`，融合 kernel 也没有体现出优于“先解压再 attention”的性能收益。

## 测试环境与启动方式

服务端为 `10.90.24.4`，使用后四张卡。统一模型路径为 `/share/models/official/Qwen3.5-35B-A3B`，统一镜像为 `vllm/vllm-openai:v0.19.0-x86_64-cu130`。

本轮实际运行了五组服务：

| 配置 | GPU | 端口 | 说明 |
| --- | ---: | ---: | --- |
| `baseline-bf16` | `4` | `8050` | 官方 vLLM，默认 KV cache |
| `baseline-fp8` | `5` | `8051` | 官方 vLLM，`--kv-cache-dtype fp8` |
| `tq-unfused` | `6` | `8052` | `CUSTOM` backend，非 fused，非 eager |
| `tq-fused-debug` | `7` | `8053` | `CUSTOM` backend，`TQ4_USE_FUSED_PAGED=1`，非 eager，保留错误日志 |
| `tq-unfused-eager` / `tq-fused-eager` | `7` | `8054` | eager 对照线，顺序运行 |

所有 benchmark 客户端均在本机通过官方镜像容器执行 `vllm bench serve`，并显式挂载本地 tokenizer 路径。

## 非 eager 主线结果

### 1. KV cache 容量

三条可用非 eager 服务的 KV cache 指标如下：

| 配置 | GPU KV cache size | 16K/request 最大并发 |
| --- | ---: | ---: |
| `baseline-bf16` | `48,576 tokens` | `9.74x` |
| `baseline-fp8` | `96,416 tokens` | `17.00x` |
| `tq-unfused` | `97,152 tokens` | `16.91x` |

这里最重要的观察不是 TurboQuant 相对 `bf16` 的优势，而是它相对 `fp8` 的真实位置。当前实现下，`tq-unfused` 的 cache 容量只比 `fp8` 多 `736 tokens`，提升不到 `1%`。这意味着如果系统已经能稳定使用 `fp8` KV cache，那么 TurboQuant 在 vLLM 0.19 这条路径上的 cache 红利并不明显。

### 2. 短序列、偏 decode 负载

测试参数为 `64 prompts / 1024 input / 512 output / request-rate=inf / temperature=0`。

| 配置 | req/s | out tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: |
| `baseline-bf16` | `2.018` | `1033.175` | `3076.14 ms` | `32.02 ms` |
| `baseline-fp8` | `2.187` | `1119.956` | `2632.74 ms` | `31.99 ms` |
| `tq-unfused` | `1.508` | `771.978` | `3094.08 ms` | `44.78 ms` |

在这组更看重 decode 的负载上，`fp8` 是最优配置。`tq-unfused` 相比 `fp8`：

- request throughput 下降约 `31%`
- output throughput 下降约 `31%`
- median TPOT 恶化约 `40%`

这说明“先解压再 attention”的 TurboQuant 路径在 decode-heavy 场景下，当前开销显著大于 cache 压缩带来的收益。

### 3. 长上下文负载

测试参数为 `32 prompts / 12000 input / 256 output / request-rate=inf / temperature=0`。

| 配置 | req/s | out tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: |
| `baseline-bf16` | `0.812` | `207.793` | `18300.28 ms` | `51.94 ms` |
| `baseline-fp8` | `0.814` | `208.377` | `16071.44 ms` | `75.05 ms` |
| `tq-unfused` | `0.474` | `121.470` | `29656.00 ms` | `117.89 ms` |

这组结果本来最有机会体现 TurboQuant 的长上下文优势，但实测并没有。相反，`tq-unfused` 相比 `fp8`：

- output throughput 下降约 `42%`
- median TTFT 增加约 `84%`
- median TPOT 增加约 `57%`

换句话说，至少在当前 `turboquant-vllm` 的 vLLM 路径中，“更长上下文、更大 KV cache 压力”并没有把 TurboQuant 推到一个更有利的位置。

## 融合算子结果

### 1. 非 eager fused 为何失败

`tq-fused-debug` 的崩溃日志已经给出明确根因：

- `torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing`
- 随后在 `torch.cuda.graph(...).capture_end()` 处触发 `cudaErrorStreamCaptureInvalidated`
- 上层抛出 `RuntimeError: Engine core initialization failed`

也就是说，当前 fused 路径的核心问题不是“性能不好”而是“根本不能稳定参与非 eager 的 CUDA graph capture”。只要服务端还想走 vLLM 默认的 graph/capture 初始化链路，这个 fused 路径就不稳定。

### 2. eager 下的公平 A/B

为了把“graph capture 失败”和“融合 kernel 本身的效果”拆开，本轮又在同一张 GPU7 上顺序跑了两组 eager 服务，统一参数为 `64 prompts / 1024 input / 512 output / request-rate=inf / temperature=0`。

| 配置 | req/s | out tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: |
| `tq-unfused-eager` | `0.582` | `298.085` | `3554.04 ms` | `108.47 ms` |
| `tq-fused-eager` | `0.485` | `248.377` | `3343.04 ms` | `114.96 ms` |

这个对照说明，哪怕完全绕开 CUDA graph，把两者放在同一个 eager 执行模式下，`tq-fused` 依然没有跑赢 `tq-unfused`：

- output throughput 再低约 `16.7%`
- median TPOT 再差约 `6.0%`
- 只在 median TTFT 上有轻微改善

因此，本轮不能支持“融合算子已经比解压后 attention 更快”的判断。

## 综合判断

从实验设计的两个目标来看，本轮可以得出三条比较明确的判断。

第一，`turboquant-vllm` 当前 fused 路径仍然不是一个稳定的非 `eager` 实现。它在 vLLM 0.19 的 CUDA graph capture 过程中直接失败，这个问题不是偶发 benchmark 波动，而是初始化阶段的硬错误。

第二，即使不考虑这个稳定性问题，把 fused 路径退回到 eager 后单独测，它也没有体现出优于 `tq-unfused` 的性能。至少在当前 `Qwen3.5-35B-A3B` 和 A100 组合上，融合 kernel 还没有带来净收益。

第三，TurboQuant 在当前 vLLM 适配里的 KV cache 收益，主要体现在相对 `bf16` 的容量翻倍，但相对 `fp8` 几乎没有实质优势；而性能上，无论短序列还是长上下文，`tq-unfused` 都明显慢于 `fp8`。

## 当前最稳妥的判断

如果后续工作的目标是“稳定复现 TurboQuant 并继续做系统实验”，当前最稳妥的表述应该是：

- `turboquant-vllm` 的非 eager fused 路径目前仍不鲁棒，不能作为稳定主线
- 当前可运行的 TurboQuant 主线仍然是 `tq-unfused`
- 但这条主线相对 `fp8` 没有体现出更好的综合性价比

如果后续还要继续深挖，优先级建议是：

1. 先解决 fused 路径的 CUDA graph capture 不安全问题
2. 再讨论 fused 是否真的能在 decode 上带来收益
3. 若目标是生产可用的 cache 压缩方案，则必须把 TurboQuant 与 `fp8` 直接比较，而不是只和 `bf16` 比较

## 第二轮修复与复测

在第一轮结果基础上，继续做了两处代码级修复：

1. 取消了 `_padded_slot_bytes()` 中把 TQ4 slot 强行提升到 `next_power_of_2` 的做法；
2. 在 `register_tq4_backend()` 中补了 hybrid config monkey-patch，让 Qwen3.5 这类 hybrid 模型在配置阶段就按 TQ4 的真实 page bytes 计算 attention block size，而不是沿用默认 dense `FullAttentionSpec`。

对应地，又新增了一组单测，覆盖：

- TQ4 page bytes 应显著小于 `fp8`
- hybrid block size 计算应基于 TQ4 真实 page bytes
- `register_tq4_backend()` 还会 patch hybrid config 入口

这组单测在官方 `v0.19.0-x86_64-cu130` 镜像内实跑通过，共 `59 passed`。

### 修复后的 KV cache

修复后重新在 `.4` 上拉起 `fp8-r2` 和 `tq-r2` 非 eager 服务，KV cache 变成：

| 配置 | GPU KV cache size | 16K/request 最大并发 |
| --- | ---: | ---: |
| `fp8-r2` | `96,416 tokens` | `17.00x` |
| `tq-unfused-r2` | `187,680 tokens` | `23.38x` |

也就是说，容量问题的根因已经被真正修掉了。`tq-unfused-r2` 相对 `fp8-r2` 的 cache 容量提升约为 `1.95x`，这和 `K4/V4 + norm` 相对 `fp8` 的理论预期已经比较接近。

### 修复后的非 eager fused

同时又做了一处稳定性修复：

- 将 `TQ4MetadataBuilder.get_cudagraph_support()` 暂时改成保守的 `NEVER`

这样做的目的不是说 fused 已经 capture-safe，而是为了先让 fused backend 在非 eager 下不要因为 CUDA graph capture 直接启动失败。修复后，`tq-fused-r3` 可以在 `enforce_eager=False` 下稳定启动，KV cache 也达到：

| 配置 | GPU KV cache size | 16K/request 最大并发 |
| --- | ---: | ---: |
| `tq-fused-r3` | `187,680 tokens` | `23.38x` |

这说明 fused 路径的“非 eager 起不来”问题已经从“硬崩溃”降到了“可以启动、可以 benchmark”的状态。

### 修复后的长上下文高压力对照

在 `64 prompts / 12000 input / 256 output / request-rate=inf / temperature=0` 这个更偏向 cache 压力的负载上，修复后的结果如下：

| 配置 | req/s | out tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: |
| `fp8-r2b` | `0.832` | `212.973` | `35568.76 ms` | `86.25 ms` |
| `tq-unfused-r2` | `0.434` | `111.049` | `92514.84 ms` | `168.00 ms` |
| `tq-fused-r3` | `0.438` | `112.173` | `91007.70 ms` | `167.68 ms` |

这组修复后的结果非常关键，因为它把“容量问题”和“性能问题”彻底分开了：

- 容量方面，TurboQuant 已经明显大于 `fp8`
- 但性能方面，`tq-unfused-r2` 依然只有 `fp8-r2b` 大约一半的 output throughput
- `tq-fused-r3` 相比 `tq-unfused-r2` 只有极小的改善，远远谈不上“显著强于 fp8 flash-attn”

换句话说，当前瓶颈已经不再是“KV cache 没压缩成功”，而是“压缩成功以后，TurboQuant 路径本身的在线计算代价仍然太高”。

## 现阶段最准确的判断

经过这轮代码修复和复测，可以把结论更新得更精确一些。

第一，TurboQuant 当前在 vLLM 里的 cache 容量问题并不是算法不行，而是实现问题。修掉 `_padded_slot_bytes()` 和 hybrid block-size 计算链之后，`K4/V4` 的 cache 容量已经能做到接近 `2x fp8`。

第二，融合算子路径的主要稳定性问题原先确实卡在 CUDA graph capture，而不是单纯“kernel 跑不动”。把 cudagraph 支持保守关掉后，`tq-fused-r3` 已经可以在非 eager 模式下启动和跑 benchmark。

第三，也是最关键的一点，当前 fused kernel 仍然没有把性能推到一个足以击败 `fp8 flash-attn` 的水平。即使 cache 容量已经修好，`tq-fused-r3` 的 output throughput 仍然只有 `112.17 tok/s`，而 `fp8-r2b` 是 `212.97 tok/s`。它只比 `tq-unfused-r2` 好了大约 `1%`，但仍然比 `fp8` 慢了接近一半。

因此，截至当前这轮迭代，最准确的状态描述应该是：

- **KV cache 提升已经修出来了**
- **非 eager fused 启动稳定性已经部分修出来了**
- **但 fused 还没有把 TurboQuant 的端到端性能推到显著强于 `fp8 flash-attn`**
