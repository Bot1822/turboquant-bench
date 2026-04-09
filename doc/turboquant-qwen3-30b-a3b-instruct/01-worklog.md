# TurboQuant Qwen3-30B-A3B-Instruct Worklog

## Objective

Evaluate whether TurboQuant materially reduces KV-cache occupancy on
`/share/models/official/Qwen3-30B-A3B-Instruct-2507` under `2 x A100 80GB`, and
whether any reduction converts into improved multi-user long-dialog capacity.

## Constraints

- Documentation must precede implementation and experiments.
- Primary claim must be KV-cache reduction, not just downstream throughput.
- Experiments should emphasize memory pressure to make TurboQuant relevant.

## Chronological Log

### 2026-03-27 00:05 Asia/Shanghai

- Created documentation skeleton for the Qwen3-30B-A3B-Instruct experiment
  track.
- Confirmed the target model exists locally as
  `/share/models/official/Qwen3-30B-A3B-Instruct-2507`.
- User clarified target priority:
  1. prove KV-cache occupancy reduction
  2. then prove better multi-user long-dialog capacity
  3. use `2 x A100 80GB` with visible memory pressure

### 2026-03-27 10:50 Asia/Shanghai

- Completed a first design draft for the Qwen3-30B-A3B-Instruct experiment.
- Parallel review identified two important flaws in the initial design:
  - structural KV replacement alone was too weak to count as the primary proof
  - one-shot long-prompt batching could miss any TurboQuant benefit because the
    free path only matters after prefill
- Tightened the design:
  - primary claim now requires structural evidence plus at least one runtime
    reuse signal
  - added a same-engine `wave1 -> free -> wave2 admission` experiment concept
  - added explicit A/B control rules and model-wide coverage accounting

### 2026-03-27 10:59 Asia/Shanghai

- Added pure experiment helpers and deterministic long-dialog workload
  generation with tests:
  - `test_experiment_utils.py`
  - `test_dialog_workload.py`
- Verified helper tests pass:
  - `5 passed`

### 2026-03-27 11:00 Asia/Shanghai

- Built `inspect_kv_occupancy.py` for the new model track.
- First structural smoke run on
  `Qwen3-30B-A3B-Instruct-2507`, `TP=2`, `max_model_len=8192`,
  target input `2048` tokens, `concurrency=1` showed:
  - TurboQuant hooks all `48` attention layers on this model
  - active baseline KV estimate per rank: `102,088,704` bytes
  - TQ side-cache bytes per rank: `20,470,912` bytes
  - active-occupancy compression ratio: `4.99x`
  - allocated-backing ratio remains much larger because vLLM preallocates the
    full KV backing store up front
- While building this script, found and fixed a real bug in
  `TurboQuantKVCache.append()`:
  - multi-token append updated `seq_len` by `1` instead of the appended token
    count
  - added a failing test and fixed the root cause

### 2026-03-27 11:06 Asia/Shanghai

- Ran `proof.py` on the new model with `TP=2`, `max_model_len=8192`.
- Strong result:
  - TurboQuant hooked all `48` attention layers
  - `nvidia-smi` dropped from about `70.6 GB`/GPU after generation to about
    `32.8 GB`/GPU after `free_kv_cache()`
  - CUDA allocator `allocated` also dropped from about `71.69 GB` to
    `31.23 GB`
- This is materially stronger than the earlier Qwen3.5-35B-A3B case, where
  allocator-visible recovery did not show up.

### 2026-03-27 11:12 Asia/Shanghai

- Ran `benchmark.py` on the new model.
- Results:
  - baseline throughput: `122.4 tok/s`
  - TurboQuant throughput: `128.8 tok/s`
  - throughput ratio: `1.05x`
  - baseline and TurboQuant outputs matched at a high level

### 2026-03-27 11:23 Asia/Shanghai

- Built and ran a two-process colocation admission test:
  - process A loads the model, runs a long-history request, and stays resident
  - process B then attempts to start a second engine on the same two GPUs
- With first process in baseline mode:
  - second engine failed even at a reduced budget
- With first process in TurboQuant mode:
  - second engine succeeded at `second_max_model_len=1024`,
    `second_gpu_mem=0.55`, `second_input_tokens=512`
- This is not same-engine block admission, but it is valid system-level
  evidence that TurboQuant-recovered memory can be consumed by a second service
  process on the same GPUs.

### 2026-03-27 11:50 Asia/Shanghai

- Reframed the experiment hierarchy after a good methodological correction:
  - direct KV occupancy is the primary proof of the TurboQuant method
  - `free_kv_cache()` and allocator-visible memory recovery are secondary,
    integration-specific proofs
- Updated the design and experiment summary accordingly.

### 2026-03-27 12:00 Asia/Shanghai

- Added a new evaluation branch for accuracy.
- Goal:
  - test whether TurboQuant changes deterministic task accuracy
  - especially on long-context retrieval prompts, not just short factual QA
- Plan:
  - run the same prompt set through baseline and TurboQuant with
    `temperature=0`
  - score against exact known answers
  - report baseline accuracy, TurboQuant accuracy, and output agreement rate

### 2026-03-27 12:08 Asia/Shanghai

- Clarified an important technical constraint before running any accuracy
  benchmark:
  - in the current default integration path, if paged KV cache still exists,
    decode falls back to native flash attention
  - therefore a naive baseline-vs-TQ comparison may measure almost no quality
    difference simply because compressed decode is not actually active
- Next step: smoke test a true compressed-decode path before claiming any
  accuracy result.

### 2026-03-27 14:26 Asia/Shanghai

- Ran a no-alloc compressed-decode smoke test.
- Result:
  - the path does run
  - but a very short prompt produced many `no_alloc decode without TQ data`
    warnings and obviously bad output
- Interpretation:
  - short prompts are not a valid way to test compressed-decode accuracy
  - the accuracy benchmark should instead use long-context retrieval prompts so
    TQ cache is actually populated and the test matches the target scenario

### 2026-03-27 14:40 Asia/Shanghai

- Reorganized experiment records by task.
- New task directories:
  - `task-kv-occupancy/`
  - `task-capacity/`
  - `task-accuracy/`
- Subsequent scripts and task-specific notes should live under the matching
  task folder.

### 2026-03-27 14:50 Asia/Shanghai

- Started a new `task-lm-eval/` branch of the experiment.
- User requested benchmark-style accuracy evaluation with:
  - `lm_eval`
  - start from `MMLU-Pro`
  - target long-context setup around `16K`
- Next steps:
  - install requested dependencies into the current Python environment
  - inspect the `lm_eval` vLLM wrapper
  - build a baseline vs TurboQuant evaluation harness

### 2026-03-27 17:28 Asia/Shanghai

- Completed the first `lm_eval` smoke comparison on `MMLU-Pro`.
- Setup:
  - `CUDA_VISIBLE_DEVICES=2,3`
  - `TP=2`
  - `max_model_len=16384`
  - `batch_size=4`
  - `limit=5`
  - `num_fewshot=0`
  - `HF_ENDPOINT=https://hf-mirror.com`
- Results:
  - baseline: `71.43`
  - `tq_no_alloc`: `0.0`
- Interpretation:
  - the current true compressed-decode path is not accuracy-safe yet
  - this does not invalidate the structural compression result, but it is a
    serious blocker for any benchmark claim about quality preservation

### 2026-03-27 18:15 Asia/Shanghai

- Performed systematic debugging on the `tq_no_alloc` accuracy collapse.
- Current root-cause candidates are strong enough to treat as real
  implementation bugs in the compressed-decode path:
  1. all requests share a single `"default"` TQ cache instead of having
     request/sequence-separated caches
  2. no-alloc prefill uses a single causal attention pass over concatenated
     batched tokens and ignores per-sequence boundaries
  3. batched decode path appears to assume `num_actual == 1` when building the
     flattened query tensor
- These bugs explain why:
  - single-request proof-style demos can look reasonable
  - but benchmark-style batched evaluation collapses badly
