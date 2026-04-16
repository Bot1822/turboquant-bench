# TurboQuant 融合算子 Benchmark 设计

日期：`2026-04-10`

## 目标

本轮实验聚焦 `turboquant-vllm` 的三件事：

1. 在 `vLLM 0.19.0` 官方镜像上跑通融合算子版本的 TurboQuant，优先不使用 `--enforce-eager`。
2. 对比三条路径的真实收益与代价：非 TurboQuant baseline、TurboQuant 解压后再走 FlashAttention、TurboQuant 融合 decode kernel。
3. 给出 TurboQuant 在同等 GPU 预算下的两类结论：一类是服务性能，另一类是 KV cache 容量提升。

本轮默认模型为 `/share/models/official/Qwen3.5-35B-A3B`，远端服务机为 `10.90.24.4`，优先使用后四张卡 `gpu4-7`。

## 被测配置

为了把“是否启用 TurboQuant”和“是否启用融合 decode kernel”拆开，本轮先做四组配置：

| 配置 | 说明 | 目的 |
| --- | --- | --- |
| `baseline-bf16` | 官方 vLLM，无 TurboQuant，默认 KV cache dtype | 作为原生精度基线 |
| `baseline-fp8` | 官方 vLLM，无 TurboQuant，`--kv-cache-dtype fp8` | 作为现有 KV cache 压缩基线 |
| `tq-unfused` | `turboquant-vllm`，`--attention-backend CUSTOM`，不设 `TQ4_USE_FUSED_PAGED` | 作为“先解压再 attention”的 TurboQuant 路径 |
| `tq-fused` | `turboquant-vllm`，`--attention-backend CUSTOM`，`TQ4_USE_FUSED_PAGED=1` | 作为融合 decode kernel 路径 |

说明：

- `tq-fused` 与 `tq-unfused` 都先固定为 `K4/V4`，避免异构 bit-width 直接让 fused gate 失效。
- 第一目标是跑通非 `eager`；只有在当前组合下确认服务无法稳定启动或 CUDA graph capture 失败，才把 `--enforce-eager` 作为保底回退。
- `TQ4_USE_INT8_PREFILL` 暂不打开，避免把 prefill 侧的额外优化混进“融合 decode vs 解压后 attention”的主比较。

## 观测指标

### 1. 启动与运行时状态

- 服务能否健康启动
- 是否实际走到 `CUSTOM` backend
- 是否打印 `Fused paged TQ4 decode: enabled`
- 是否进入 CUDA graph 相关路径，或在 capture 阶段失败
- 是否出现 `Paged decompress ... using dynamic fallback`

### 2. 服务性能

优先使用在线 serving benchmark，关注：

- `TTFT`
- `TPOT`
- `ITL`
- `request throughput`
- `output throughput`

为了让 decode 成为主瓶颈，benchmark 会采用较长输入、较长输出和较高并发，至少覆盖：

- 中等长度：`input≈4K`, `output≈512`
- 长上下文：`input≈12K-16K`, `output≈512`
- 高并发压力：优先把客户端并发打到 `64-128`

### 3. KV cache 容量

从服务日志和运行配置同时记录：

- `num_gpu_blocks`
- 启动后静态显存占用
- 同一 `gpu_memory_utilization` 下的最大可用 cache block 数
- 长上下文/大 batch 下的 OOM 或 scheduler 饱和点

结论上既看理论压缩倍率，也看 vLLM 实际 allocator 最终给到的 block 提升比例。

## 实验阶段

### 阶段 A：跑通与定位

先在 `.4` 上把四组服务都拉起来，确认：

- 容器镜像可用
- `turboquant-vllm` 能在官方 `v0.19.0-x86_64-cu130` 镜像中安装
- `tq-fused` 在 A100 上能成功 import / JIT / 首次请求
- 不加 `--enforce-eager` 时，服务是否能稳定 warmup

### 阶段 B：小规模 smoke benchmark

在四组都能返回结果后，先做小规模压力测试，快速看三点：

- `tq-fused` 相对 `tq-unfused` 是否已经有 decode 侧优势
- 非 `eager` 是否真的生效，还是已经退回到等价 eager
- TurboQuant 是否仍然因为 paged fallback 吞掉收益

### 阶段 C：正式 benchmark

如果 `tq-fused` 跑通，则进行正式对比：

- `baseline-bf16` vs `baseline-fp8`
- `baseline-fp8` vs `tq-unfused`
- `tq-unfused` vs `tq-fused`

若四张卡都可用，则四组服务并行常驻，客户端分批对四个端口发压。

## 关键假设

当前代码显示，`tq-fused` 只在单 token decode 生效，prefill 仍然不是完整 fused FlashAttention；因此本轮不预期它在所有 workload 上都明显更快。真正可能出现收益的场景是：

- decode 占比高
- cache block 引用较集中
- 没有大量 paged fallback
- 服务端没有因为 Triton/capture 限制退回到等价慢路径

如果结果仍然不如 `tq-unfused`，优先排查三类原因：

1. 融合 kernel 实际没有命中或没有稳定命中；
2. CUDA graph 没有生效，Python/调度开销仍在；
3. TurboQuant 的压缩收益被 codebook lookup、在线解压和 block remap 开销抵消。

## 产出物

本轮最终需要形成一份中文报告，至少包含：

- 启动方式与环境
- 代码路径与实际命中情况
- 服务性能对比
- KV cache 容量对比
- 融合算子相对解压后 attention 的净收益
- TurboQuant 相对非 TurboQuant baseline 的整体收益与代价
