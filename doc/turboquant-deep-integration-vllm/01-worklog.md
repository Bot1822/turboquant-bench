# TurboQuant 深度集成 vLLM Worklog

## Objective

将当前以 layer hook + side-cache 为主的 TurboQuant 原型，演进为更深层的
vLLM 集成方案，优先打通单实例在线服务路径，并保留完整设计、开发与验证记录。

## Chronological Log

### 2026-03-27 12:00 Asia/Shanghai

- 用户要求转向“整体重构 turboquant，基于论文，深度集成 vllm”。
- 明确这条主线与前面的实验文档分离记录。
- 在 `turboquant` 子仓库创建了隔离工作树：
  `/ceph/User/E01442/turboquant/turboquant/.worktrees/deep-integration-vllm`
- `.worktrees/` 已加入子仓库 `.gitignore` 并单独提交。

### 2026-03-27 14:40 Asia/Shanghai

- 在工作树中实现了最小单实例在线集成：
  - `tq_api_server.py`
  - `turboquant/server_integration.py`
  - `test_tq_server.py`
- `tq_api_server.py` 现在会：
  - 使用标准 vLLM OpenAI server 启动链
  - 在 engine 初始化完成后通过 `collective_rpc` 安装 TurboQuant hooks
  - 暴露 `/turboquant/status`、`/turboquant/capture`、`/turboquant/free`、
    `/turboquant/reset` 管理接口

### 2026-03-27 15:15 Asia/Shanghai

- 用 `vllm/benchmarks/multi_turn` 跑通了 baseline / TQ 的单实例 multi-turn
  smoke A/B。
- 使用了适配 `max_model_len=2048` 的 synthetic conversation 配置，保证
  benchmark 不因为上下文越界而失真。
- 当前阶段结论：
  - TQ server 启动链路已经打通
  - 单实例在线 smoke 跑通
  - 与 baseline 相比，当前 hooks 安装本身没有显示出明显在线性能退化

### 2026-03-27 15:20 Asia/Shanghai

- 为 single-instance online smoke 新增了适配 `max_model_len=2048` 的
  synthetic workload 配置：
  `doc/turboquant-deep-integration-vllm/generate_multi_turn_fit2048.json`
- 这避免了过多的上下文越界请求，使 baseline / TQ 的 server A/B 至少能在
  一个干净工作点上比较。

### 2026-03-27 17:40 Asia/Shanghai

- 新增了长 decode streaming benchmark：
  `doc/turboquant-deep-integration-vllm/benchmark_long_decode_stream.py`
- 在 worktree 里为它补了纯 helper 和测试：
  - `turboquant/stream_bench.py`
  - `test_stream_bench.py`
- 长 decode 三组结果：
  - baseline: `tps_post_ttft ≈ 165.176`
  - TQ/no-free: `tps_post_ttft ≈ 164.634`
  - TQ/free-after-first-chunk: `tps_post_ttft ≈ 36.978`
- 当前结论：
  - 默认 TQ 路径在 paged KV 仍存在时，decode 性能与 baseline 基本一致
  - 这符合远端修复后的设计：默认 decode 仍走 flash attention fast path
  - 当 mid-stream free 后，真正切到基于量化 cache 的 decode 路径时，
    当前实现明显更慢，不是更快
