# TurboQuant 深度集成 vLLM 验证记录

## Purpose

记录这条深度集成主线上的验证命令、结果与问题。

## Validation Table

| Time | Validation | Command | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-03-27 12:05 | Worktree baseline | `../..//../vllm/.venv/bin/python -m pytest -q test_turboquant.py::test_kv_cache` | Pass | 新工作树创建后，先确认基线测试可运行。 |
| 2026-03-27 14:32 | TQ server helper tests | `../..//../vllm/.venv/bin/python -m pytest -q test_tq_server.py` | Pass | `2 passed`。 |
| 2026-03-27 14:39 | TQ server status smoke | `../..//../vllm/.venv/bin/python tq_api_server.py ... --tq-enable --tq-mode active` + `curl http://127.0.0.1:8011/turboquant/status` | Pass | 单实例 TQ server 启动成功，`/turboquant/status` 返回 `48` hooks on both ranks。 |
| 2026-03-27 15:02 | Baseline multi-turn serving smoke | `../../.venv/bin/python benchmark_serving_multi_turn.py --served-model-name qwen30-base --url http://127.0.0.1:8010 --input-file /ceph/User/E01442/turboquant/doc/turboquant-deep-integration-vllm/generate_multi_turn_fit2048.json --num-clients 2 --max-active-conversations 4 --max-turns 6 --warmup-step --limit-min-tokens 64 --limit-max-tokens 64 --request-timeout-sec 300` | Pass | `runtime_sec=11.567`, `requests_per_sec=3.977`, `ttft_ms mean=45.81`, `tpot_ms mean=7.14`, `latency_ms mean=495.89`。 |
| 2026-03-27 15:10 | TQ multi-turn serving smoke | `../../.venv/bin/python benchmark_serving_multi_turn.py --served-model-name qwen30-tq --url http://127.0.0.1:8011 --input-file /ceph/User/E01442/turboquant/doc/turboquant-deep-integration-vllm/generate_multi_turn_fit2048.json --num-clients 2 --max-active-conversations 4 --max-turns 6 --warmup-step --limit-min-tokens 64 --limit-max-tokens 64 --request-timeout-sec 300` | Pass | `runtime_sec=11.567`, `requests_per_sec=3.977`, `ttft_ms mean=43.61`, `tpot_ms mean=7.14`, `latency_ms mean=493.61`。与 baseline 基本持平，当前在线 hooks 本身未见明显开销。 |
| 2026-03-27 17:32 | Long decode baseline | `PYTHONPATH=/ceph/User/E01442/turboquant/turboquant/.worktrees/deep-integration-vllm URL=http://127.0.0.1:8010 SERVED_MODEL_NAME=qwen30-base MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 INPUT_TOKENS=1400 MAX_TOKENS=512 FREE_AFTER_FIRST_CHUNK=0 vllm/.venv/bin/python doc/turboquant-deep-integration-vllm/benchmark_long_decode_stream.py` | Pass | `input_tokens=1419`, `output_tokens=512`, `ttft_ms=145.961`, `latency_ms=3245.683`, `tps_post_ttft=165.176`。 |
| 2026-03-27 17:15 | Long decode TQ no-free | `PYTHONPATH=/ceph/User/E01442/turboquant/turboquant/.worktrees/deep-integration-vllm URL=http://127.0.0.1:8011 SERVED_MODEL_NAME=qwen30-tq MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 INPUT_TOKENS=1400 MAX_TOKENS=512 FREE_AFTER_FIRST_CHUNK=0 vllm/.venv/bin/python doc/turboquant-deep-integration-vllm/benchmark_long_decode_stream.py` | Pass | `input_tokens=1419`, `output_tokens=512`, `ttft_ms=141.661`, `latency_ms=3251.585`, `tps_post_ttft=164.634`。与 baseline 基本一致。 |
| 2026-03-27 17:23 | Long decode TQ free-after-first-chunk | `PYTHONPATH=/ceph/User/E01442/turboquant/turboquant/.worktrees/deep-integration-vllm URL=http://127.0.0.1:8011 SERVED_MODEL_NAME=qwen30-tq MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 INPUT_TOKENS=1400 MAX_TOKENS=512 FREE_AFTER_FIRST_CHUNK=1 FREE_URL=http://127.0.0.1:8011/turboquant/free vllm/.venv/bin/python doc/turboquant-deep-integration-vllm/benchmark_long_decode_stream.py` | Pass with regression | `free_result.status=200`, `ttft_ms=38.671`, but `latency_ms=13884.698`, `tps_post_ttft=36.978`。说明当前量化-cache decode 路径在该长 decode 场景下显著慢于 flash attention。 |
