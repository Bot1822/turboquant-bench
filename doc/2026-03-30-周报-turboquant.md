# TurboQuant 复现与评测周报

日期：2026-03-30  
汇报人：Codex 协助整理

## 一、本周目标

本周工作围绕两条主线展开：

1. 复现并验证 TurboQuant 在 vLLM 上的基础可运行性
2. 评估 TurboQuant 在长上下文场景下的实际价值，重点包括：
   - KV cache 压缩是否真实成立
   - 是否能带来可观的显存/容量收益
   - 是否会影响吞吐与精度

本周主要测试了两个模型：

- `Qwen3.5-35B-A3B`
- `Qwen3-30B-A3B-Instruct-2507`

其中后者是本周的重点模型。

## 二、本周完成的工作

### 1. 基础环境与代码整理

- 建立了完整实验文档结构，所有实验过程、设计、计划、结果均已落到 `doc/`。
- 为实验补充了多份辅助脚本，包括：
  - 直接 KV 占用诊断
  - baseline 生成脚本
  - 容量/准入探针
  - `lm_eval` benchmark 脚本
- 吸收了远端 `turboquant` 最新的 5 个关键提交，主要集中在：
  - GQA decode shape 修复
  - hook 开销优化
  - lazy capture paged KV
  - `free_kv_cache` 行为修正

### 2. Qwen3.5-35B-A3B 结果

该模型主要用于摸清当前实现的边界。

已验证：

- baseline 能正常运行
- TurboQuant hook 可以安装，覆盖 `10` 个 `full_attention` 层
- 输出和 baseline 在表面上基本一致
- 吞吐接近 baseline：
  - baseline：`18.4 tok/s`
  - TurboQuant：`17.9 tok/s`
  - 比值：`0.97x`

已发现问题：

- 该模型只覆盖 `10` 个 `full_attention` 层，不能代表全模型
- 虽然 `free_kv_cache()` 返回了较大的 `freed_bytes`，但：
  - `nvidia-smi` 没有明显下降
  - `torch.cuda.memory_allocated()` 也没有明显下降

结论：

- 这条线证明了 **功能可运行**
- 但不能证明 **真实显存已经被回收**
- 因此不适合作为核心收益模型

### 3. Qwen3-30B-A3B-Instruct-2507 结果

该模型是当前最有代表性的实验对象。

模型特征：

- 架构：`Qwen3MoeForCausalLM`
- 层数：`48`
- KV heads：`4`
- `head_dim = 128`
- 文本-only，结构更适合隔离 KV cache 问题

#### 3.1 直接 KV 占用压缩结果

这是本周最核心的结果。

在如下配置下：

- `2 x A100 80GB`
- `TP=2`
- `MAX_MODEL_LEN=8192`
- 输入目标长度约 `2048` tokens
- `CONCURRENCY=1`

测得：

- baseline active KV occupancy：
  `102,088,704 bytes/rank`
- TurboQuant active KV occupancy：
  `20,470,912 bytes/rank`
- 压缩比：
  `4.99x`

结论：

- TurboQuant 在这个模型上，**已经明确实现了 KV cache 压缩**
- 这是本周最可靠、最直接的主结论

#### 3.2 显存回收结果

在 `proof.py` 跑出的结果中，Qwen3-30B 这条线上显示出明显的显存变化：

- `48` 个 hooks 安装成功
- `nvidia-smi`：
  - after gen: 约 `70.6 GB/GPU`
  - after free: 约 `32.8 GB/GPU`
- CUDA allocator:
  - after gen: 约 `71.69 GB`
  - after free: 约 `31.23 GB`

结论：

- 与 Qwen3.5-35B-A3B 不同
- 在 Qwen3-30B 这条线上，当前实现已经能看到 allocator-visible 的显存回收

说明：

- 这说明当前 vLLM 集成在该模型上表现更好
- 但这仍然属于“集成效果”，不是 TurboQuant 方法本身的定义

#### 3.3 吞吐结果

`benchmark.py` 的结果：

- baseline：`122.4 tok/s`
- TurboQuant：`128.8 tok/s`
- 比值：`1.05x`

结论：

- 当前测试点上没有观察到明显吞吐惩罚
- 甚至略高于 baseline

### 4. 系统级容量探针

做过一个“驻留第一条服务后，第二个引擎还能不能在同卡启动”的探针实验。

结果：

- baseline 驻留后，第二个引擎起不来
- TurboQuant 驻留后，第二个降配引擎能起

但这一结果 **不作为主结论**，因为：

- 它是跨进程系统级探针
- 不是单实例在线服务的真实用户场景
- baseline 还会受到 vLLM 启动预留显存机制影响

因此该实验只保留为补充记录，不用于最终核心论证。

### 5. 精度评测结果

本周开始引入 `lm_eval` 做 benchmark 式精度验证。

先做了 `MMLU-Pro` 的 smoke run：

环境：

- `CUDA_VISIBLE_DEVICES=2,3`
- `TP=2`
- `MAX_MODEL_LEN=16384`
- `batch_size=4`
- `limit=5`
- `num_fewshot=0`
- `HF_ENDPOINT=https://hf-mirror.com`

结果：

- baseline：`71.43`
- `tq_no_alloc`：`0.0`

结论：

- 当前**真正压缩态 decode** 路径存在严重精度问题
- 现在不能宣称 TurboQuant 在 benchmark 级任务上“精度基本不变”

## 三、本周确认的关键问题

### 1. 当前 `tq_no_alloc` 路径存在高概率实现 bug

目前对精度崩塌的根因已经有较强证据，主要包括：

1. **多请求共享同一个 `default` TQ cache**
   - 不同请求/样本的 KV 很可能被混入同一个压缩缓存
   - 导致跨请求 attention 污染

2. **no-alloc prefill 没有正确处理 batched 多序列边界**
   - 当前实现更像把 batch 中多个请求拼成一条长序列做 causal attention
   - 这会直接破坏正确性

3. **batched decode 的 query reshape 逻辑可疑**
   - 实现明显偏向 `num_actual == 1`
   - 多请求 decode 时很可能形状就已经错了

这三点可以解释为什么：

- 单请求 demo 看起来“还能跑”
- 但一上 `lm_eval` 这种 batched benchmark，就直接崩成 `0.0`

### 2. 当前 demo 还不是在线 serving 级方案

原因不是模型本身，而是集成层级不够深。

当前实现本质是：

- layer hook
- side-cache
- 外部替换原始 paged KV

它还不是：

- request-aware
- scheduler-aware
- block-manager-aware

的 cache engine。

因此现在更准确的定位应该是：

- **TurboQuant 方法的工程原型**
- 而不是已经可直接上生产的在线长对话 cache 引擎

## 四、本周阶段性结论

本周可以对导师明确汇报的结论是：

1. TurboQuant 在 `Qwen3-30B-A3B-Instruct-2507` 上已经成功实现了 **KV cache 压缩**
2. 当前实验点下，active KV occupancy 压缩比约 **4.99x**
3. 在该模型上，当前 vLLM 集成还观察到了显存回收和接近 baseline 的吞吐表现
4. 但当前真正的 compressed-decode 路径存在严重精度问题，已经在 `MMLU-Pro` smoke run 中暴露
5. 因此当前状态应定义为：
   - **压缩效果成立**
   - **集成有前景**
   - **精度与在线多请求正确性仍需修复**

## 五、下周计划

下周工作建议按下面顺序推进：

1. 先修复 `tq_no_alloc` 的正确性问题
   - request-level cache 分离
   - batched prefill 边界处理
   - batched decode query 形状处理

2. 修复后重新跑 benchmark 式精度评测
   - `MMLU-Pro`
   - `IFEval`
   - `math_hard`

3. 在精度基本恢复后，再做单实例在线多用户长对话实验
   - 不再依赖“双实例共存”探针
   - 改做同一服务实例下的真实 admission / latency / throughput 测试

4. 如果目标是长期应用到在线服务，则需要进一步研究：
   - 深度接入 vLLM cache engine
   - 替换/接管 block manager
   - 建立 request-aware 的 KV 生命周期管理
