# TurboQuant Qwen3-30B-A3B-Instruct Experiment Record

## Purpose

This file records every experiment run, observed metric, failure, and
interpretation for the Qwen3-30B-A3B-Instruct evaluation.

## Experiment Table

| Time | Run Type | Command | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-03-27 10:51 | Structural occupancy smoke run | `CUDA_VISIBLE_DEVICES=0,1 VLLM_ENABLE_V1_MULTIPROCESSING=0 MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 TP=2 MAX_MODEL_LEN=8192 GPU_MEM=0.85 INPUT_TOKENS=2048 CONCURRENCY=1 MAX_TOKENS=16 MODE=accumulate vllm/.venv/bin/python doc/turboquant-qwen3-30b-a3b-instruct/task-kv-occupancy/inspect_kv_occupancy.py` | Success | After fixing `TurboQuantKVCache.append()` sequence-length accounting, TurboQuant hooked all `48` attention layers. Active baseline KV estimate per rank was `102,088,704` bytes, TQ side-cache bytes per rank were `20,470,912`, for an active-occupancy compression ratio of `4.99x`. |
| 2026-03-27 10:57 | Regression test for multi-token append | `../vllm/.venv/bin/python -m pytest test_turboquant.py::test_kv_cache_multi_token_append_updates_seq_len -q` | Success | Added a failing regression test first, then fixed `TurboQuantKVCache.append()` so multi-token appends increment `seq_len` correctly. |
| 2026-03-27 11:03 | TurboQuant proof on Qwen3-30B | `CUDA_VISIBLE_DEVICES=0,1 MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 TP=2 MAX_MODEL_LEN=8192 GPU_MEM=0.85 ../vllm/.venv/bin/python proof.py` | Success | `48` hooks installed. `nvidia-smi` dropped from `[70623, 70623] MB` after generation to `[32833, 32833] MB` after free. CUDA `allocated` dropped from `[71.69, 71.69] GB` to `[31.23, 31.23] GB`. |
| 2026-03-27 11:10 | Throughput benchmark on Qwen3-30B | `CUDA_VISIBLE_DEVICES=0,1 MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 TP=2 GPU_MEM=0.85 MAX_MODEL_LEN=8192 ../vllm/.venv/bin/python benchmark.py` | Success | Baseline `122.4 tok/s`, TurboQuant `128.8 tok/s` (`1.05x`). Outputs aligned at a high level. |
| 2026-03-27 11:23 | Colocation admission A/B | `CUDA_VISIBLE_DEVICES=0,1 FIRST_MODE=baseline|tq MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 TP=2 MAX_MODEL_LEN=8192 GPU_MEM=0.85 INPUT_TOKENS=4096 CONCURRENCY=1 MAX_TOKENS=64 SECOND_MAX_MODEL_LEN=1024 SECOND_GPU_MEM=0.55 SECOND_INPUT_TOKENS=512 SECOND_MAX_TOKENS=32 vllm/.venv/bin/python doc/turboquant-qwen3-30b-a3b-instruct/task-capacity/colocation_admission.py` | Internal probe only | With baseline resident, the second engine failed because only about `8.59 GiB` free remained. With TurboQuant resident, the second engine succeeded after wave 1 left about `45.57 GiB` free. This is recorded for debugging but excluded from the final evidence chain because it is not the target serving scenario. |
| 2026-03-27 17:28 | `lm_eval` MMLU-Pro smoke run | `CUDA_VISIBLE_DEVICES=2,3 VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_ENDPOINT=https://hf-mirror.com MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 MODE=baseline|tq_no_alloc TASKS=mmlu_pro TP=2 GPU_MEM=0.85 MAX_MODEL_LEN=16384 LM_EVAL_BATCH_SIZE=4 LIMIT=5 NUM_FEWSHOT=0 vllm/.venv/bin/python doc/turboquant-qwen3-30b-a3b-instruct/task-lm-eval/run_lm_eval.py` | Baseline/TQ diverged sharply | Baseline scored `71.43`, while `tq_no_alloc` scored `0.0` on the same `MMLU-Pro` smoke subset. This is the first benchmark-style evidence that the current compressed-decode path is not accuracy-safe. |

## Summary

- The new target model appears materially better for this experiment than the
  previous Qwen3.5-35B-A3B track because TurboQuant hooks all `48` attention
  layers rather than only a partial subset.
- The first structural smoke run already provides the primary evidence we
  actually need for the paper-level claim: TurboQuant compresses active
  hooked-layer KV occupancy by about `4.99x` at a `~2k`-token prompt length on
  `TP=2`.
- Limitations still remain:
  - online single-instance multi-user serving evidence is still pending
  - the colocation result above is deliberately excluded from the final claim
  - the active structural compression ratio was measured at a moderate pressure
    point; higher-pressure sweeps can still refine the story
  - the first `lm_eval` smoke run shows the current `tq_no_alloc` path causes
    severe accuracy degradation on `MMLU-Pro`
