# TurboQuant-vLLM Full Benchmark Record

## 说明

本文件只记录本轮 full benchmark，不混入此前的 smoke 或系统化子样本结果。当前 full 阶段使用 `vllm/vllm-openai:v0.19.0-x86_64-cu130` 官方镜像、`Qwen3.5-35B-A3B` 模型、远端主机 `10.90.24.4`，并把客户端并发统一提高到 `num_concurrent=128`。截至当前，`IFEval`、`Math Hard`、`MMLU-Pro` 三组 full 结果都已经完成。

## 服务与任务分配

本轮 full benchmark 使用两对服务并行推进：

| 组 | GPU | 端口 | Baseline 参数 | Plugin 参数 | 负责任务 |
| --- | --- | --- | --- | --- | --- |
| A | `gpu4/gpu5` | `8040/8041` | `--kv-cache-dtype fp8 --max-model-len 16384 --enforce-eager` | `--attention-backend CUSTOM --max-model-len 16384 --enforce-eager` | `IFEval`，之后接 `Math Hard` |
| B | `gpu6/gpu7` | `8042/8043` | `--kv-cache-dtype fp8 --max-model-len 16384 --enforce-eager` | `--attention-backend CUSTOM --max-model-len 16384 --enforce-eager` | `MMLU-Pro` |

客户端统一使用：

```bash
--model local-completions
--model_args model=qwen35-local,tokenizer=/share/models/official/Qwen3.5-35B-A3B,max_length=16384,num_concurrent=128,max_retries=10,tokenizer_backend=huggingface,tokenized_requests=False
```

## 已完成结果

### MMLU-Pro Full c128

结果文件：

- baseline: [mmlu_pro_baseline_full_c128](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/results/plugin_v019_full/mmlu_pro_baseline_full_c128)
- plugin: [mmlu_pro_plugin_full_c128](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/results/plugin_v019_full/mmlu_pro_plugin_full_c128)

聚合结果如下：

| 指标 | Baseline | Plugin |
| --- | ---: | ---: |
| `acc` | `0.6208 ± 0.00442` | `0.6171 ± 0.00443` |
| 近似 `ELAPSED_SECONDS` | `9075.45` | `10173.82` |

这里的 elapsed 是根据本轮 `c128` 启动时间 `2026-04-10 14:48:41` 和结果文件时间戳估算得到的近似值，足够用于阶段性吞吐判断。

这组全量结果把此前 `limit=200` 的趋势放大成了更稳定的结论：plugin 在 `MMLU-Pro` 上仍略低于 baseline，差距约 `0.37` 个百分点。因为这是全量 `12032` 题、约 `113990` 个 loglikelihood 请求，这个差值比小样本阶段更有说服力。

### IFEval Full c128

结果文件：

- baseline: [ifeval_baseline_full_c128](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/results/plugin_v019_full/ifeval_baseline_full_c128)
- plugin: [ifeval_plugin_full_c128](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/results/plugin_v019_full/ifeval_plugin_full_c128)

聚合结果如下：

| 指标 | Baseline | Plugin |
| --- | ---: | ---: |
| `prompt_level_strict_acc` | `0.4473` | `0.4658` |
| `prompt_level_loose_acc` | `0.4787` | `0.5083` |
| `inst_level_strict_acc` | `0.5683` | `0.5827` |
| `inst_level_loose_acc` | `0.5911` | `0.6115` |
| `ELAPSED_SECONDS` | `526.40` | `1072.71` |

样本级比较：

- 总样本数：`541`
- `prompt_level_strict` 相同：`497`
- `prompt_level_loose` 相同：`493`
- `inst_level_strict` 相同：`480`
- `inst_level_loose` 相同：`478`
- `prompt_level_strict` 上 baseline-only better：`17`
- `prompt_level_strict` 上 plugin-only better：`27`

解释上，这组 full 结果仍然支持“plugin 在 IFEval 上略高于 baseline”，但 wall time 明显更长，说明提高并发并没有让 plugin 在这条生成路径上获得更好的综合效率。

### Math Hard Full c128

结果文件：

- baseline: [math_hard_baseline_full_c128_rerun](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/results/plugin_v019_full/math_hard_baseline_full_c128_rerun)
- plugin: [math_hard_plugin_full_c128_rerun](/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/results/plugin_v019_full/math_hard_plugin_full_c128_rerun)

聚合结果如下：

| 指标 | Baseline | Plugin |
| --- | ---: | ---: |
| `exact_match` | `0.6254 ± 0.0123` | `0.6224 ± 0.0123` |
| `ELAPSED_SECONDS` | `814.38` | `1850.88` |

样本级比较：

- 总样本数：`1324`
- same exact-match label：`1084`
- baseline-only correct：`122`
- plugin-only correct：`118`

分学科结果如下：

| 子任务 | Baseline | Plugin |
| --- | ---: | ---: |
| `algebra` | `0.8241` | `0.8534` |
| `counting_and_probability` | `0.6098` | `0.5691` |
| `geometry` | `0.5000` | `0.5455` |
| `intermediate_algebra` | `0.3786` | `0.4000` |
| `number_theory` | `0.7792` | `0.7468` |
| `prealgebra` | `0.7824` | `0.7461` |
| `precalculus` | `0.4222` | `0.3630` |

这组 full 结果显示 plugin 和 baseline 的 aggregate 分数非常接近，但 plugin 的 wall time 明显更长。分学科看，plugin 并不是所有数学子任务都更差，而是在不同子任务上有增有减，最终总分略低于 baseline。

## 运行时观察

`c128` 以后，baseline 的 GPU 利用率已经明显抬升。监控时曾看到：

- `gpu6/gpu7` 的 `MMLU-Pro c128` 在 `93%~96%`
- `gpu4` 的 `Math Hard baseline c128` 在 `52%`
- 服务端日志里出现 `Running: 106 reqs, Waiting: 150, GPU KV cache usage: 99.3%`

这说明 baseline 在高并发下已经从“客户端并发不够”转到“服务端 KV cache 和调度真正被压满”的状态。

plugin 侧则出现了更明显的 TurboQuant paged-decompress dynamic fallback。此前小样本阶段常见的是 `39` 或 `122` 个 unique blocks 超出容量；在 full `c128` 压力下，日志中已经出现接近 `889/890 unique blocks exceed pre-allocated capacity (32 blocks)` 的 warning。这是当前 full 阶段最稳定的负面运行时信号。

## 当前仍在运行

本轮 full benchmark 三组任务都已经完成。当前不再有活跃的 full `lm_eval` 客户端进程。

## 收尾清理

`2026-04-10 18:38:35 +0800`，对远端 `10.90.24.4` 做了一次 full benchmark 服务清理。最初尝试 `ssh -p 2222` 被拒绝，随后改用默认 SSH 端口登录 `guipeng@10.90.24.4` 执行删除。

本次移除的残留服务容器为：

- `zgp-vllm019-cu130-full-baseline-a`
- `zgp-vllm019-cu130-full-plugin-a`
- `zgp-vllm019-cu130-full-baseline-b`
- `zgp-vllm019-cu130-full-plugin-b`

清理后再次检查：

- `docker ps -a` 中已无 `zgp-` 前缀容器
- `8040` 到 `8043` 端口已无监听

至此，`.4` 上本轮 full benchmark 遗留 server 已全部清空。
