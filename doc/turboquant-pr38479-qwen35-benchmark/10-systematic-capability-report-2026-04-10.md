# TurboQuant-vLLM 系统化能力测试报告

## 摘要

本轮测试在 `vllm/vllm-openai:v0.19.0-x86_64-cu130` 官方镜像上，对 `turboquant-vllm` 插件和固定后的官方 baseline 做了三类能力对比：指令跟随 `IFEval`、通用常识 `MMLU-Pro`、数学推理 `Math Hard`。服务部署在 `10.90.24.4`，本地机器只作为 benchmark client 发请求。为了覆盖约 `16K` 的上下文预算，两边服务都以 `--max-model-len 16384` 启动；为了比此前单并发测试更接近实际服务压力，`lm_eval` 客户端统一使用 `num_concurrent=8`。

最终结果不是单向的。`MMLU-Pro` 上 plugin 比 baseline 低 `1` 个百分点，`Math Hard` 上 plugin 低约 `1.4` 个百分点；但在 `IFEval` 上，plugin 的四个指令跟随指标都高于 baseline。吞吐方面，plugin 在 `MMLU-Pro` 和 `Math Hard` 上更慢，在 `IFEval` 上略快。运行时日志显示，plugin 在高并发和长生成场景下会持续触发 TurboQuant paged-decompress dynamic fallback，这是比小幅分数波动更稳定的负面信号。

## 测试环境与启动方式

| 项目 | Baseline | Plugin |
| --- | --- | --- |
| 镜像 | `vllm/vllm-openai:v0.19.0-x86_64-cu130` | `vllm/vllm-openai:v0.19.0-x86_64-cu130` |
| 模型 | `/share/models/official/Qwen3.5-35B-A3B` | `/share/models/official/Qwen3.5-35B-A3B` |
| 远端机器 | `10.90.24.4` | `10.90.24.4` |
| GPU | `gpu6` | `gpu7` |
| 端口 | `8042` | `8043` |
| vLLM 参数 | `--kv-cache-dtype fp8 --max-model-len 16384 --enforce-eager` | `--attention-backend CUSTOM --max-model-len 16384 --enforce-eager` |

Baseline 服务使用如下命令启动：

```bash
ssh guipeng@10.90.24.4 "docker run -d --name zgp-vllm019-cu130-sys-baseline --entrypoint bash --network host --gpus 'device=6' -v /ceph/User/E01442/turboquant:/ceph/User/E01442/turboquant -v /share/models/official:/share/models/official vllm/vllm-openai:v0.19.0-x86_64-cu130 -lc 'vllm serve /share/models/official/Qwen3.5-35B-A3B --served-model-name qwen35-local --dtype float16 --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code --host 0.0.0.0 --port 8042 > /tmp/sys-baseline.log 2>&1'"
```

Plugin 服务使用如下命令启动：

```bash
ssh guipeng@10.90.24.4 "docker run -d --name zgp-vllm019-cu130-sys-plugin --entrypoint bash --network host --gpus 'device=7' -v /ceph/User/E01442/turboquant:/ceph/User/E01442/turboquant -v /share/models/official:/share/models/official vllm/vllm-openai:v0.19.0-x86_64-cu130 -lc 'pip install -e /ceph/User/E01442/turboquant/turboquant-vllm >/tmp/sys-plugin-pip.log 2>&1 && vllm serve /share/models/official/Qwen3.5-35B-A3B --served-model-name qwen35-local --dtype float16 --attention-backend CUSTOM --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code --host 0.0.0.0 --port 8043 > /tmp/sys-plugin.log 2>&1'"
```

服务启动后通过 `/v1/models` 做 readiness 检查。两边都返回 `max_model_len=16384`，并且 `/v1/completions` 支持 `echo=true` 和 `logprobs=1`，因此本轮统一使用 `lm_eval --model local-completions`，不使用 chat-completions 路径。

## 测试场景

本轮选择 `lm_eval` 的 leaderboard 任务名，避免把聊天模板、思维链前缀和 prompt 变体混入能力结论。`IFEval` 使用 `leaderboard_ifeval`，是 `generate_until` 任务，默认 `max_gen_toks=1280`；`MMLU-Pro` 使用 `leaderboard_mmlu_pro`，是 multiple-choice 任务，通过 loglikelihood 打分；`Math Hard` 使用 `leaderboard_math_hard`，它是 7 个 MATH-lighteval Level 5 子任务组成的 group，默认 `max_gen_toks=1024`。

本地 Qwen3.5 tokenizer 对前 50 条样本的估算显示，这三类任务的 prompt 都远小于 `16K`：`IFEval` 约 `17-90` tokens，`MMLU-Pro` 约 `822-1154` tokens，`Math Hard` 约 `882-1138` tokens。因此 `16384` 的服务配置主要是保留长上下文预算，而不是这三个任务本身会自然触达 16K。

| 数据集 | lm_eval 任务 | 类型 | 样本预算 | 说明 |
| --- | --- | --- | --- | --- |
| IFEval | `leaderboard_ifeval` | generate | `limit=50` | 全量为 `541`，但单样本生成成本高，本轮采用 50 条子样本 |
| MMLU-Pro | `leaderboard_mmlu_pro` | multiple choice | `limit=200` | 全量 test 为 `12032`，本轮取 200 条 |
| Math Hard | `leaderboard_math_hard` | generate | `limit=10` | 实际为 7 个子任务各 10 条，共 70 条 |

正式测试命令统一使用 `HF_ENDPOINT=https://hf-mirror.com`、`max_length=16384`、`num_concurrent=8`、`tokenized_requests=False`。例如 MMLU-Pro baseline 命令为：

```bash
HF_ENDPOINT=https://hf-mirror.com /usr/bin/time -f 'ELAPSED_SECONDS=%e' /ceph/User/E01442/turboquant/vllm/.venv/bin/lm_eval --model local-completions --model_args model=qwen35-local,tokenizer=/share/models/official/Qwen3.5-35B-A3B,base_url=http://10.90.24.4:8042/v1/completions,timeout=99999999,max_length=16384,num_concurrent=8,max_retries=10,tokenizer_backend=huggingface,tokenized_requests=False --tasks leaderboard_mmlu_pro --output_path /ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/results/plugin_v019_systematic/mmlu_pro_baseline_l200_c8 --trust_remote_code --num_fewshot 5 --limit 200 --log_samples --seed 0,0,0,0
```

其他任务只替换 `--tasks`、`--limit`、`--output_path`，plugin 只把 `base_url` 从 `8042` 切到 `8043`。

## 结果

| 任务 | 样本数 | Baseline 结果 | Plugin 结果 | Baseline wall time | Plugin wall time | Baseline 吞吐 | Plugin 吞吐 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `leaderboard_mmlu_pro` | 200 | `acc=0.48 ± 0.0354` | `acc=0.47 ± 0.0354` | `174.52 s` | `192.69 s` | `1.146 samples/s` | `1.038 samples/s` |
| `leaderboard_ifeval` | 50 | `prompt_strict=0.44`, `inst_strict=0.5526` | `prompt_strict=0.52`, `inst_strict=0.6184` | `577.27 s` | `545.69 s` | `0.0866 samples/s` | `0.0916 samples/s` |
| `leaderboard_math_hard` | 70 | `exact_match=0.5714 ± 0.0579` | `exact_match=0.5571 ± 0.0545` | `538.54 s` | `608.82 s` | `0.1300 samples/s` | `0.1150 samples/s` |

`MMLU-Pro` 的 200 条样本共对应 `1979` 个 loglikelihood 请求。baseline 约 `11.34 requests/s`，plugin 约 `10.27 requests/s`。样本级别看，baseline 和 plugin 在 `192/200` 条上对错一致；baseline-only correct 为 `5` 条，plugin-only correct 为 `3` 条。因此 plugin 的 `1` 个百分点下降来自少量选项偏好漂移，不是大面积错误。

`IFEval` 的 50 条样本里，plugin 的四个指标都高于 baseline。`prompt_level_strict` 上两者有 `46/50` 条一致，plugin-only better 为 `4` 条，baseline-only better 为 `0` 条。这个结果说明在当前 slice 上 plugin 没有指令跟随退化，但由于样本数较小，不能外推为 plugin 在 IFEval 上稳定更强。

`Math Hard` 的 70 条样本中，baseline 与 plugin 在 `53/70` 条上 exact-match 一致；baseline-only correct 为 `9` 条，plugin-only correct 为 `8` 条。分学科看，plugin 在 `algebra` 和 `number_theory` 上更高，baseline 在 `counting_and_probability`、`prealgebra` 和 `precalculus` 上更高，其中 `precalculus` 差距最大：baseline `0.7`，plugin `0.4`。

| Math Hard 子任务 | Baseline exact_match | Plugin exact_match |
| --- | ---: | ---: |
| `algebra` | `0.7` | `0.9` |
| `counting_and_probability` | `0.6` | `0.5` |
| `geometry` | `0.4` | `0.4` |
| `intermediate_algebra` | `0.2` | `0.2` |
| `number_theory` | `0.7` | `0.9` |
| `prealgebra` | `0.7` | `0.6` |
| `precalculus` | `0.7` | `0.4` |

## 运行时观察

高并发测试确认了 `local-completions + num_concurrent=8` 是可用的。相对早先单并发，MMLU-Pro 的请求吞吐显著提高；但生成型任务的 wall time 仍然受输出长度和长尾支配，IFEval 和 Math Hard 都明显慢于 MMLU-Pro。

plugin 在运行中多次触发 TurboQuant 的 paged-decompress dynamic fallback。MMLU-Pro 并发 probe 中出现过 `39 unique blocks exceed pre-allocated capacity (32 blocks)`，Math Hard 正式测试中持续出现 `122 unique blocks exceed pre-allocated capacity (32 blocks)`。这些警告没有直接导致 OOM 或任务失败，但说明 plugin 在较重负载下仍然走到了动态 fallback 路径，其运行时鲁棒性弱于 baseline。

## 结论

本轮三任务结果不能支持“TurboQuant 插件必然能力下降”的结论，也不能支持“TurboQuant 插件能力无损且性能更好”的结论。更准确的判断是：能力结果是混合的，运行时结果偏负面。

从能力看，plugin 在 `IFEval` 上优于 baseline，在 `MMLU-Pro` 和 `Math Hard` 上略低于 baseline。三个任务的差异都不是灾难性塌陷级别，更多表现为少量样本上的答案偏好漂移。从吞吐看，plugin 在 `MMLU-Pro` 和 `Math Hard` 上都慢于 baseline；尤其 Math Hard 这种长生成负载下，plugin 慢约 `11.5%`，并持续触发 paged-decompress fallback。

因此，当前更可靠的结论是：`turboquant-vllm` 适合继续作为学习和二次开发对象，也能完成三类能力 benchmark；但在这套 A100 + vLLM 0.19 + Qwen3.5-35B-A3B 的配置下，它还没有表现出相对固定官方 baseline 的稳定吞吐收益，也没有足够证据证明它是可直接替代 baseline 的更优实现。

## 结果文件

| 任务 | Baseline 输出 | Plugin 输出 |
| --- | --- | --- |
| MMLU-Pro | `results/plugin_v019_systematic/mmlu_pro_baseline_l200_c8/` | `results/plugin_v019_systematic/mmlu_pro_plugin_l200_c8/` |
| IFEval | `results/plugin_v019_systematic/ifeval_baseline_l50_c8/` | `results/plugin_v019_systematic/ifeval_plugin_l50_c8/` |
| Math Hard | `results/plugin_v019_systematic/math_hard_baseline_l10_c8/` | `results/plugin_v019_systematic/math_hard_plugin_l10_c8/` |
