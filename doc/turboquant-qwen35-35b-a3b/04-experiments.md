# TurboQuant Qwen3.5-35B-A3B Experiment Record

## Purpose

This file records every bring-up attempt, benchmark run, and observed metric for
the requested model.

## Experiment Table

| Time | Run Type | Command | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-03-26 22:39 | Baseline startup | `CUDA_VISIBLE_DEVICES=0,1 vllm/.venv/bin/python -m vllm.entrypoints.cli.main bench startup --model /share/models/official/Qwen3.5-35B-A3B --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.85 --num-iters-cold 1 --num-iters-warm 0 --num-iters-warmup 0 --trust-remote-code --skip-mm-profiling` | Partial success | Engine initialized fully, but the CLI crashed afterwards because `warm_startup_array` was empty when `--num-iters-warm 0` was used. |
| 2026-03-26 22:49 | Baseline generate | `CUDA_VISIBLE_DEVICES=0,1 VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm/.venv/bin/python doc/turboquant-qwen35-35b-a3b/baseline_generate.py` | Success | `load_s=121.282`, `gen_s=66.682`, `tok_per_s=0.96`, `blocks=3231`, memory after load `[69817, 69817] MB`, after gen `[69819, 69819] MB`. |
| 2026-03-26 22:52 | TurboQuant proof | `CUDA_VISIBLE_DEVICES=0,1 MODEL=/share/models/official/Qwen3.5-35B-A3B TP=2 MAX_MODEL_LEN=4096 GPU_MEM=0.85 ../vllm/.venv/bin/python proof.py` | Success with caveats | TurboQuant hooked `10` full-attention layers and reported `34,938 MB` freed per GPU. The script's context-capacity math is not trustworthy yet because it still uses a 27B-specific hard-coded block size. |
| 2026-03-26 23:01 | TurboQuant proof, corrected | `CUDA_VISIBLE_DEVICES=0,1 MODEL=/share/models/official/Qwen3.5-35B-A3B TP=2 MAX_MODEL_LEN=4096 GPU_MEM=0.85 ../vllm/.venv/bin/python proof.py` | Success with caveats | Corrected runtime block size to `1056`, observed `10` hooked layers, and recalculated theoretical context improvement to `2.00x`. `nvidia-smi` and CUDA allocator metrics still did not show a real drop after `free_kv_cache()`. |
| 2026-03-26 23:17 | Free behavior inspection | `CUDA_VISIBLE_DEVICES=0,1 VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm/.venv/bin/python doc/turboquant-qwen35-35b-a3b/inspect_tq_free.py` | Success | Before free, the 10 hooked layers each pointed to a distinct KV storage of about `3.49 GB`; after free, those layer and runner entries were replaced by shared `int8[1]` tensors, but CUDA `allocated` memory still did not decrease. |
| 2026-03-26 23:32 | Throughput benchmark | `CUDA_VISIBLE_DEVICES=0,1 MODEL=/share/models/official/Qwen3.5-35B-A3B TP=2 GPU_MEM=0.85 MAX_MODEL_LEN=4096 ../vllm/.venv/bin/python benchmark.py` | Success with caveats | Baseline `18.4 tok/s`, TurboQuant `17.9 tok/s` (`0.97x`), outputs and quality prompt matched at a high level, `10` full-attention hooks installed, theoretical context improvement `2.00x`, but allocator-visible memory remained unchanged after free. |

## Summary

- The requested model can run under `vllm==0.17.0` on `2 x A100 80GB` with a
  conservative configuration.
- TurboQuant's hook path is operational on the requested model and sees the
  expected `10` full-attention layers.
- The benchmark harness has been corrected to use the runtime block size rather
  than the stale 27B constant, producing a `2.00x` theoretical context estimate
  instead of the earlier incorrect value.
- The main unresolved blocker is memory reclamation proof:
  `free_kv_cache()` replaces the visible KV cache tensors, but on this stack we
  did not observe a corresponding decrease in CUDA `memory_allocated()` or
  `nvidia-smi` usage.
- Current evidence therefore supports:
  - baseline runnability
  - TurboQuant hook installation and output compatibility
  - near-baseline throughput (`0.97x`)
  - theoretical KV capacity gain based on replaced tensor storage
- Current evidence does **not** yet support claiming verified real-world GPU
  memory recovery after `free_kv_cache()` on this exact setup.
