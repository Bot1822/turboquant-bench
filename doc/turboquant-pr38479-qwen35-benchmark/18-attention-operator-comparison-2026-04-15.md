# TurboQuant Attention 算子实现对比研究

日期：`2026-04-15`

## 研究范围与结论摘要

本报告比较三条实现路线在 attention 算子上的差异：`turboquant-vllm` 插件方案、`vllm-project/vllm` 的 PR `#38479` 方案，以及论文 `TurboQuant` 原文。讨论重点不是量化思想本身，而是 attention 算子的系统实现：prefill 与 decode 如何分工，`K/V` 是否采用不同计算路径，decode 时究竟是“先重建再算”还是“压缩域直接估计分数”。其中，PR 方案部分仅依据 GitHub upstream 上 PR `#38479` 当前公开信息撰写。

结论可以概括为三点。第一，论文原文实际上提出了两种不同目标的量化器：`TurboQuantmse` 和 `TurboQuantprod`，这意味着在 attention 实现上，`K` 与 `V` 更适合不同路径。第二，`turboquant-vllm` 插件主线并未真正采用这种分工，而是把 `K/V` 都放进“重建后再做 attention”的框架里，只是把重建位置从图外移动到了 kernel 内部。第三，PR `#38479` 当前公开方案比插件更接近论文的分工思路：prefill 与 decode 分路，`K` 与 `V` 分路，但它并没有照搬论文中的 `QJL residual correction`，而是演化成更偏 `WHT + norm correction` 的工程实现。

## 一、论文原文中的 attention 算子含义

论文的核心不在于一个现成的 attention kernel，而在于两个不同目标的量化器。`TurboQuantmse` 优化均方误差，用于让向量在重建后尽可能接近原向量。`TurboQuantprod` 则显式优化 inner product，它的定义是先用 `b-1` bit 的 `TurboQuantmse` 量化，再对 residual 做一次 `1-bit QJL`，从而构造无偏的内积估计器。论文在 `TurboQuantprod` 的定义中把估计器写成两部分之和：一部分来自 `Qmse` 的重建项，另一部分来自 residual 的 `QJL` 校正项。这个结构非常关键，因为它说明论文作者并不把“还原出完整向量再点积”视为内积场景的最佳实现。

如果把这个理论结构映射到 KV cache attention，最自然的系统解释是：`K` 的主要用途是计算 `Q·K^T`，因此更适合走 `TurboQuantprod`；`V` 的主要用途是做 `softmax(scores) @ V`，因此更适合走 `TurboQuantmse` 或其他 reconstruction-friendly 路径。论文在 LongBench 端到端实验里进一步说明了这一点的工程含义：其 KV cache 压缩在 streaming generation 中持续生效，而且采用了 outlier / non-outlier channel 分组，不同通道组分配不同 bit-width，形成 `2.5-bit` 和 `3.5-bit` 这样的有效精度。换句话说，论文原文更像是在描述一种“`K` 和 `V` 不同职责、不同算子路径”的 attention 设计原则，而不是一条统一的“把 `K/V` 都解压出来再做 attention”的实现。

需要特别说明的是，论文没有给出一个可直接部署的 Triton/CUDA attention kernel，也没有细化 prefill 和 decode 在系统层面应该如何拆分。因此，论文只能提供 attention 算子设计的原则，不能直接回答 kernel 该如何分块、如何组织 GQA、如何做 paged KV 访问。这些部分必须由系统实现自行决定。

## 二、`turboquant-vllm` 插件方案

`turboquant-vllm` 当前主线在线路径集中在 [`tq4_backend.py`](/ceph/User/E01442/turboquant/turboquant-vllm/src/turboquant_vllm/vllm/tq4_backend.py) 和 [`fused_paged_tq4_attention.py`](/ceph/User/E01442/turboquant/turboquant-vllm/src/turboquant_vllm/triton/fused_paged_tq4_attention.py)。它的核心特征，是把 `K` 和 `V` 都存成 packed TQ4 cache，然后在 decode 时要么通过 paged decompress 还原成 FP16/BF16 再调用 `flash_attn_varlen_func`，要么在 fused decode kernel 中边解压边做 attention。无论哪条路径，`K` 的参与方式本质上仍然是“恢复出数值后再计算 attention 分数”，而不是论文 `TurboQuantprod` 意义上的压缩域内积估计。

更具体地说，prefill 路径里，插件先调用 `_compress_and_store()` 把新 token 的 `K/V` 写进 packed cache，然后 `_tq4_prefill()` 会从 paged cache 中把相关 block 通过 `_decompress_cache_paged()` 还原为连续的 `key_cache` 和 `value_cache`，随后直接调用 `flash_attn_varlen_func`。decode 的 unfused 路径 `_tq4_decode()` 也是同一逻辑：压缩写入新 token，旋转 query，分页解压 K/V，再交给 FlashAttention。即使 decode 的 fused 路径 `_fused_decode_path()` 不再显式生成完整的 `key_cache/value_cache` 张量，它在 kernel 内部做的事依然是：从 cache 中取 packed K/V，解压成每个 tile 的实数向量，算 `QK` 分数，再对解压后的 `V` 做加权和。也就是说，fused 版本只是把“解压”从 Python/PyTorch 图外移到了 kernel 内部，并没有改变 attention 算子的理论结构。

这一点和仓内其它代码也能互相印证。`turboquant-vllm` 仓里确实实现了论文意义上的 `TurboQuantProd` 与 `asymmetric_attention_scores()`：[`quantizer.py`](/ceph/User/E01442/turboquant/turboquant-vllm/src/turboquant_vllm/quantizer.py) 定义了 `MSE + QJL residual correction` 的无偏内积估计器，[`compressors.py`](/ceph/User/E01442/turboquant/turboquant-vllm/src/turboquant_vllm/compressors.py) 也实现了“直接从压缩 keys 计算 attention scores”的接口。但这套路径没有接入当前 vLLM backend 主线，原因在仓内注释里说得非常明确：[`kv_cache.py`](/ceph/User/E01442/turboquant/turboquant-vllm/src/turboquant_vllm/kv_cache.py#L136) 直接指出，标准 attention 做的是“对解压后的 keys 计算 `Q @ K^T`”，因此 `QJL` 在这条路径里是“不可见”的，白白牺牲了一部分 MSE bit-width。换句话说，插件仓本身同时包含“论文一致的 key inner-product 实现原型”和“当前实际在线的 vLLM 集成路径”，但主线线上路径选择的是后者。

这也决定了插件方案在 attention 算子上的真实定位：它是一个“压缩存储 + 解压计算”的系统。其主要优化点在于 cache 容量与内存布局，而不是把 `K-score` 计算真正改写成压缩域算子。

## 三、PR `#38479` 方案

PR `#38479` 这里仅讨论 GitHub upstream 当前公开状态。根据 PR 页面摘要，这条路线已经明确把 prefill 和 decode 分开处理。prefill 继续使用原生 attention 路径，例如 `flash_attn_varlen_func` 等现有后端；TurboQuant 只在 store path 中介入，把写入 KV cache 的 `K/V` 变成压缩表示。decode 则改成专门的 TurboQuant 路径：从 cache 中读取压缩后的 `K/V`，对 `K` 做解包和 score 计算，对 `V` 做解包后再参与加权求和。

PR 页面给出的架构图对这一点描述得比较直接。它把 store path 写成“`K → WHT rotation → Lloyd-Max quantize → bit-pack`，`V → uniform quantize → bit-pack`”；把 decode path 写成“`cache → unpack K → dequant → Q·K scores`”以及“`cache → unpack V → dequant → score·V`”；把 prefill path 写成“`Raw Q, K, V → flash attention → output`”。从这个结构上看，PR 方案已经不再是插件那种“prefill / decode 都以解压后 attention 为核心”的统一路径，而是明确把 decode 做成一个专门的压缩 cache 读取算子。

不过，PR 当前公开方案并不是论文原文里 `TurboQuantprod` 的直接系统实现。PR 页面摘要明确提到两项重要工程改写。第一，rotation 采用的是 `Walsh-Hadamard Transform (WHT) + random sign flips`，而不是论文原文里更一般的随机正交旋转。第二，PR 明确写了 `No QJL`，理由是社区实验认为 QJL 在 attention 中会放大 softmax 方差、损害质量。因此，这条方案虽然保留了论文“让 `K-score` 路径和 `V` 路径分离”的方向，但在具体 attention 算子实现上，已经从“`MSE + residual QJL` 的无偏内积估计”演化成“`WHT + Lloyd-Max + norm correction` 的工程化 score path”。

PR 页面还展示了它支持的 cache preset，例如 `turboquant_k8v4`、`turboquant_4bit_nc`、`turboquant_k3v4_nc` 和 `turboquant_3bit_nc`。这些命名本身就反映出该方案已经显式区分了 `K` 和 `V`：有的 preset 选择 `K=FP8, V=4-bit uniform`，有的则给 `K` 和 `V` 都用低比特加 `norm correction (NC)`。这和论文“`K`、`V` 在 attention 中职责不同”是同方向的，但表达形式明显更工程化。

## 四、三者差异的核心归纳

如果只看 attention 算子的结构，不看包装层，这三条线可以概括成三种不同的方法论。

论文原文提供的是“理论分工”。它强调 `K` 的核心任务是内积估计，因此应该尽量走 `TurboQuantprod`；`V` 的核心任务是被加权聚合，因此应该走 `TurboQuantmse` 或其他 reconstruction-friendly 路径。论文没有规定一个具体 GPU kernel，但清楚地区分了“score path”和“value path”。

`turboquant-vllm` 插件主线提供的是“压缩存储 + 解压计算”。它把 `K/V` 都存成低比特 packed cache，但无论是 unfused 还是 fused，attention 计算最终都建立在“已经解压成实数”的 `K/V` 上。fused 版本只是把解压移到 kernel 内部，没有把 `K-score` 改造成压缩域 inner-product 算子。因此，它在 attention 设计上更接近“把量化当成 cache layout 优化”，而不是“把 attention score 算子本身改写成 TurboQuantprod”。

PR `#38479` 方案提供的则是“prefill 原生 attention + decode 专用压缩域 score”。prefill 完全沿用 raw `Q/K/V` attention，保持稳定性；decode 则让 `K` 走压缩域 score path，让 `V` 走解量化后聚合。这一结构显著更接近论文对 `K` 和 `V` 的职责划分。不过，PR 当前公开方案没有照搬论文里的 `QJL residual correction`，而是已经演化成更偏 `WHT + Lloyd-Max + norm correction` 的工程实现，因此它不能被简单视为“论文原文的直接系统实现”，而更像是“吸收论文思想后形成的 vLLM 工程化 attention backend”。

## 五、对当前重构工作的启示

如果目标是继续重构当前 `turboquant-vllm` 的 attention kernel，那么最重要的结论不是某个 Triton 参数该怎么调，而是实现路线本身需要调整。现有插件主线的瓶颈来自 fused attention kernel，但其根因并不只是内核写得不够快，更在于它仍然沿用“解压 `K/V` 再算 attention”的理论结构。因此，即使继续围绕当前 kernel 做优化，也只能在同一种方法论里打补丁。

真正和论文、也和 PR 主线更一致的重构方向，是把 `K-score` 路径从“重建后点积”改成“压缩域内积估计”，并把 `V` 保留在 reconstruction 路径。这意味着下一步应该优先做一个 `K-only compressed score kernel` 原型，而不是继续围绕当前 “in-kernel dequant K/V” 版本做局部微调。换句话说，插件方案若想在 attention 算子层面真正向论文和 PR 靠近，核心不在于把现有 kernel 再提速 10%，而在于把 attention 的分工方式重新划分为“`K` 为 score 服务，`V` 为 value aggregation 服务”。
