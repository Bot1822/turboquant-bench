# TurboQuant 双周周报

统计周期：`2026-03-30` 至 `2026-04-13`

## 一、工作概况

本周期的工作主线围绕 TurboQuant 在 vLLM 上的两条实现路线展开：一条是上游原生集成路线 PR `#38479`，另一条是社区独立插件仓 `turboquant-vllm`。工作内容覆盖了社区现状调研、代码结构评估、官方镜像环境复现、能力与性能 benchmark、融合算子问题定位，以及后续针对 KV cache 容量异常和 fused 路径稳定性的代码修复与复测。整体上，本周期已经把“当前社区实现是否可用”“不同路线的工程成熟度如何”“TurboQuant 的能力和性能代价是什么”“融合算子为什么没体现出预期收益”这几类核心问题全部落到了可复现证据上。

从阶段成果看，社区调研已经形成了比较明确的判断：截至 `2026-04-08`，上游 vLLM 仍没有合并后的官方稳定 TurboQuant 支持，PR `#38479` 依旧是主要的原生实现路径；`turboquant-vllm` 则是当前最成熟、迭代最快的社区实现，更适合做复现、对比和外部实验。代码评估也已完成，结论是 `turboquant-vllm` 在工程完整度、鲁棒性和二次开发友好度上明显优于 PR `#38479`，而 PR 更能代表未来 upstream native 方案的形态。

## 二、研发与调研进展

在环境和工程侧，本周期完成了 TurboQuant 工作区初始化，整理了 `vllm` 与 `turboquant-vllm` 两条代码线，并补齐了实验设计、执行计划、工作日志和专题报告体系。基于官方镜像 `vllm/vllm-openai:v0.19.0-x86_64-cu130`，在 A100 环境上完成了 `Qwen3.5-35B-A3B` 的多轮复现。期间确认了几个关键事实：官方 `0.19` 基线在该模型上必须显式使用 `kv_cache_dtype=fp8` 才能稳定走通；插件仓 `turboquant-vllm` 在同一镜像内可以通过 `CUSTOM` backend 跑起来；而其最激进的 fused paged decode 路径虽然代码存在，但在真实服务初始化中仍然存在明显的稳定性问题。

在能力评测侧，本周期分别完成了 subsample、systematic 与 full 三层评测。面向系统化能力回归，使用 `IFEval`、`MMLU-Pro` 和 `Math Hard` 三个数据集做了正式比较。full 结果显示，plugin 在 `MMLU-Pro` 上略低于 baseline，`acc` 从 `0.6208` 下降到 `0.6171`；在 `Math Hard` 上从 `0.6254` 下降到 `0.6224`；在 `IFEval` 上则略有提升，但耗时显著增长。换句话说，TurboQuant 插件并没有带来大幅能力崩塌，但也没有在这些任务上稳定实现“零代价”，更没有把效率优势转化为整体评测收益。

在 serving 性能与实现排障侧，本周期最核心的发现来自 `turboquant-vllm` 的 fused 路径。第一轮实验表明，`TQ4_USE_FUSED_PAGED=1` 在非 `eager` 模式下会在 vLLM 的 CUDA graph capture 阶段报 `operation not permitted when stream is capturing`，随后触发 `Engine core initialization failed`。这意味着当前 fused 路径不是简单“性能不好”，而是默认初始化链路就不稳定。进一步的 eager 对照也说明，即使完全绕开 cudagraph，融合算子也没有跑赢“先解压再 attention”的非融合路径。

## 三、关键实验结论

本周期最重要的技术收获其实不是某一条 benchmark 的数值，而是把两个长期混在一起的问题拆开了。第一是 KV cache 容量问题。最初观测到 `tq-unfused` 的 KV cache 大小只有 `97,152 tokens`，几乎与 `fp8` 的 `96,416 tokens` 持平，这与 `K4/V4` 相对 `fp8` 理论上应接近翻倍的预期明显不符。进一步沿着 `slot bytes -> page size -> hybrid block size -> num_blocks` 这条链路排查后，最终确认根因是实现层把 TQ4 的 packed slot 强行做了 `next_power_of_2` padding，再叠加 Qwen3.5 hybrid 模型的 Mamba page alignment，导致理论压缩率几乎被完全吃掉。

围绕这个问题，本周期继续做了代码修复：一方面取消了 `_padded_slot_bytes()` 中对 TQ4 slot 的粗暴幂次对齐，另一方面在 `register_tq4_backend()` 中增加了针对 hybrid 模型配置阶段的 monkey-patch，让 attention block size 在配置阶段就按 TQ4 的真实 page bytes 计算，而不是沿用 dense `FullAttentionSpec`。随后补了一组单测，实跑 `59 passed`。修复后，在同样的 `.4` A100 环境下，`tq-unfused-r2` 和 `tq-fused-r3` 的 `GPU KV cache size` 都提升到了 `187,680 tokens`，相对 `fp8-r2` 的 `96,416 tokens` 已经达到约 `1.95x`。这说明先前“TurboQuant 的 cache 并没有明显变大”不是算法结论，而是实现缺陷。

第二个被拆开的，是“融合算子不稳定”和“融合算子不够快”这两个问题。为了解决前者，本周期将 `TQ4MetadataBuilder.get_cudagraph_support()` 暂时改为保守的 `NEVER`，让 fused 路径先不要参与 vLLM 的 CUDA graph capture。修复后，`tq-fused-r3` 已经能够在非 `eager` 模式下稳定启动并进入 benchmark，而不再像第一轮那样在初始化阶段硬崩溃。这是一个明确的稳定性改进。

但即使这样，融合算子的性能问题依旧存在。第一轮 eager 对照显示，在 `64 prompts / 1024 input / 512 output` 的偏 decode 负载下，`tq-unfused-eager` 的输出吞吐是 `298.08 tok/s`，`tq-fused-eager` 只有 `248.38 tok/s`，后者反而更慢。第二轮在修复 KV cache 后，又做了更偏 cache 压力的长上下文高负载对照：`64 prompts / 12000 input / 256 output / request-rate=inf`。结果中，`fp8-r2b` 的输出吞吐为 `212.97 tok/s`，修复后的 `tq-unfused-r2` 仅有 `111.05 tok/s`，而 `tq-fused-r3` 也只有 `112.17 tok/s`，只比 `unfused` 高约 `1%`。也就是说，尽管 cache 容量问题已经修复，且 fused 路径的非 eager 启动稳定性已有改进，但 fused kernel 仍然远远没有达到“显著强于 fp8 flash-attn”的目标。

## 四、交付物与文档沉淀

本周期已沉淀多份专题文档，覆盖社区调研、实现状态、代码评估、能力评测、full benchmark 和融合算子专项排障等内容。核心文档包括：社区现状报告 [06-community-status-2026-04-08.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/06-community-status-2026-04-08.md)、代码调研报告 [08-code-evaluation-pr38479-vs-plugin-2026-04-08.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/08-code-evaluation-pr38479-vs-plugin-2026-04-08.md)、vLLM 0.19 对比轨迹 [09-turboquant-vllm-v019-comparison-2026-04-08.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/09-turboquant-vllm-v019-comparison-2026-04-08.md)、系统化能力报告 [10-systematic-capability-report-2026-04-10.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/10-systematic-capability-report-2026-04-10.md)、full benchmark 记录 [11-full-benchmark-record-2026-04-10.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/11-full-benchmark-record-2026-04-10.md)，以及融合算子设计、过程和总结 [12-fused-kernel-benchmark-design-2026-04-10.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/12-fused-kernel-benchmark-design-2026-04-10.md)、[14-fused-kernel-worklog-2026-04-10.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/14-fused-kernel-worklog-2026-04-10.md)、[15-fused-kernel-report-2026-04-10.md](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/15-fused-kernel-report-2026-04-10.md)。

除文档外，还形成了一批可复用的 benchmark 脚本和结果目录，包括 decode、retrieval、container runner、benchmark logic 与评分脚本，后续继续复现或扩展对比时可直接复用。

## 五、当前问题与下阶段重点

当前阶段最明确的结论是，TurboQuant 在 vLLM 里的 cache 容量问题已经从实现层修复出来了，非 eager fused 启动稳定性也有了阶段性进展，但性能瓶颈并没有因此消失。换句话说，现阶段的主要矛盾已经从“压缩没生效”转移到了“压缩后的在线计算成本仍然太高”。

下阶段如果继续推进，工作重点应当放在两件事上。第一，继续围绕 fused kernel 本身做更细的微基准和单测，确认它只比 unfused 快 `1%` 的根因到底来自 codebook lookup、在线 decompression、tile 组织、还是 vLLM 调度边界。第二，系统性评估当前 TurboQuant 路径与 `fp8` 的真实适用边界，而不是只盯着容量优势；也就是说，要找到在什么样的上下文长度、并发和调度条件下，新增的 KV cache 容量能转化为实质收益，否则现阶段仍然无法支持把 TurboQuant 作为一个显著优于 `fp8 flash-attn` 的方案来推广。
