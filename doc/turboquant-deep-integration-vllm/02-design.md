# TurboQuant 深度集成 vLLM 设计

## 目标

把当前 TurboQuant 从“实验型 hook 集成”推进到“更接近在线服务可用”的
vLLM 深度集成。

## 当前问题

- 现状以 `turboquant.vllm_attn_backend` 的 monkey-patch 为主
- 证明了方法能压缩 KV
- 但离 request-aware / scheduler-aware 的 cache engine 还差很远

## 当前设计方向

分阶段推进：

1. **最小在线集成**
   - 保持当前 TurboQuant 压缩逻辑
   - 打通单实例 server 启动后自动安装 TQ hooks
   - 让 `vllm/benchmarks/multi_turn` 可直接对 baseline / TQ 做 A/B
   - 先证明 hooks 在线路线上不会明显拖慢服务

2. **更深层的 cache 接入**
   - 研究在不依赖全局 `free_kv_cache()` 的情况下，如何把压缩后的 KV
     表示纳入更稳定的运行时路径
   - 优先聚焦 request-level 生命周期与 admission 影响

3. **长期目标**
   - 逐步逼近真正的 cache engine 替换/接管，而不是只在 attention 层 patch

## 本轮默认目标

本轮先完成第一阶段：

- 单实例在线服务集成
- 多用户多轮长上下文 benchmark 跑通
- 保留完整记录

## 已确认的代码落点

从本地 `vllm==0.17.0` 代码可以确定，request-level 的 KV 生命周期核心不在
attention backend，而在以下层：

- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/vllm/v1/core/sched/scheduler.py`

当前判断：

- `turboquant.vllm_attn_backend` 适合做“方法接入”和“实验型替换”
- 但如果要走向 request-aware / scheduler-aware 的稳定在线方案，
  下一阶段必须下沉到 block/request 生命周期层

## 本轮已完成的最小在线集成

已在隔离工作树中实现：

- `tq_api_server.py`
  - 单实例 OpenAI-compatible TQ server launcher
- `turboquant/server_integration.py`
  - 服务器侧 TQ 集成 helper
- `/turboquant/status`
- `/turboquant/capture`
- `/turboquant/free`
- `/turboquant/reset`

它的意义是：

- baseline / TQ 的单实例 online A/B 已经可运行
- 当前 hooks 安装本身没有表现出明显 serving 开销

它的限制是：

- 目前还没有把 TurboQuant 变成 request-aware 的 cache engine
- 仍然主要是“server 启动后安装 hooks”的集成
