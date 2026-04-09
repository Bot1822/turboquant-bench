# TurboQuant Qwen3.5-35B-A3B Worklog

## Objective

Determine whether the existing TurboQuant implementation in this repository can
be run successfully on `/share/models/official/Qwen3.5-35B-A3B`, and if so,
produce benchmark evidence for its benefit relative to the baseline path.

## Constraints

- Documentation must be written before implementation and kept up to date.
- Work should continue autonomously unless a fatal blocker is encountered.
- Containers are allowed if they reduce bring-up risk or setup time.

## Initial Notes

- Top-level directories present at start: `doc/`, `turboquant/`, `vllm/`
- Existing TurboQuant paper PDF archived locally
- Existing implementation claims to target vLLM 0.17.0 and Qwen3.5-27B

## Chronological Log

### 2026-03-26 22:10 Asia/Shanghai

- Created documentation skeleton before further technical work.
- Next: inspect repo instructions, implementation layout, runtime entrypoints,
  and benchmark path.

### 2026-03-26 22:12 Asia/Shanghai

- Read `vllm/AGENTS.md`.
- Confirmed vLLM-specific constraints:
  - Python environment must be managed with `uv`
  - avoid bare `pip` and bare system `python3` for vLLM workflows
  - expected editable install path is via `uv pip install -e .`
- Confirmed `turboquant/README.md` documents an implementation tested only on
  Qwen3.5-27B, not the requested model.

### 2026-03-26 22:14 Asia/Shanghai

- Confirmed requested model path exists:
  `/share/models/official/Qwen3.5-35B-A3B`
- Confirmed local hardware:
  - GPUs 0,1,2,3,6,7 are effectively free A100 80GB
  - GPUs 4 and 5 are occupied by another vLLM workload and should be avoided
- Confirmed Docker is available if local environment setup becomes fragile.

### 2026-03-26 22:16 Asia/Shanghai

- Inspected target model config:
  - architecture: `Qwen3_5MoeForConditionalGeneration`
  - text backbone: `qwen3_5_moe_text`
  - 40 text layers
  - 16 Q heads, 2 KV heads, `head_dim=256`
  - hybrid layout with 10 `full_attention` layers and 30 `linear_attention`
    layers
- This matters because the current TurboQuant code only implements active decode
  for flash/full-attention paths and explicitly falls back on MLA-style paths.

### 2026-03-26 22:18 Asia/Shanghai

- Inspected `turboquant/proof.py` and `turboquant/benchmark.py`.
- Found model-specific assumptions that are not valid for the requested model:
  - default model path/name points to Qwen3.5-27B
  - hard-coded block size estimate `784`
  - benchmark model registry uses `/mnt/llm_models/...` paths
  - README claims 16 full-attention layers, while the requested model has 10
- Conclusion: baseline bring-up may work with configuration changes, but the
  benchmark and capacity-estimation path needs adaptation before results can be
  trusted.

### 2026-03-26 22:24 Asia/Shanghai

- Began environment bring-up with `uv` and Python 3.12 under `vllm/.venv`.
- Initial attempt followed `vllm/AGENTS.md` literally:
  `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
- Detailed logs showed an important mismatch:
  - the install path selected a precompiled upstream dev wheel around
    `0.18.1rc1.dev...`
  - this diverges from TurboQuant's stated target of `vLLM 0.17.x`
- Decided not to use that editable install path for runtime bring-up.

### 2026-03-26 22:27 Asia/Shanghai

- Pivoted runtime strategy:
  - keep local `vllm/` checkout for code reading and benchmark CLI references
  - install a pinned runtime `vllm==0.17.0` into the same `uv` environment for
    first bring-up
- Rationale:
  - closer to TurboQuant's documented compatibility window
  - avoids accidental testing against a newer upstream dev snapshot
  - reduces ambiguity when diagnosing hook breakage

### 2026-03-26 22:34 Asia/Shanghai

- Runtime environment validated from the repository root:
  - `torch 2.10.0+cu129`
  - `vllm 0.17.0` imported from the virtualenv site-packages path
  - local `turboquant` package import succeeded
- Important nuance:
  - if Python is launched from inside `vllm/`, the local source tree shadows the
    installed `vllm==0.17.0`
  - all runtime experiments must be launched from the top-level repo root or
    the `turboquant/` subrepo, not from `vllm/`

### 2026-03-26 22:45 Asia/Shanghai

- `vllm bench startup` with `TP=2`, `max_model_len=4096`, `gpu_mem=0.85` proved
  that the requested model can complete:
  - architecture resolution
  - NCCL TP worker initialization
  - weight loading
  - KV cache sizing
  - CUDA graph capture
- The CLI itself exposed two issues for our chosen invocation:
  - `--num-iters-warm 0` triggers a percentile calculation on an empty array
  - worker shutdown reporting is noisy after the benchmark completes
- Conclusion: the startup benchmark is useful for coarse bring-up, but a custom
  generation script is more reliable for correctness and performance data.

### 2026-03-26 22:52 Asia/Shanghai

- Added `doc/turboquant-qwen35-35b-a3b/baseline_generate.py` as a file-backed
  baseline driver because vLLM worker `spawn` cannot re-import `'<stdin>'`.
- Baseline generation succeeded on GPUs `0,1` with `TP=2`:
  - load time: `121.282s`
  - generation time for 64 tokens: `66.682s`
  - throughput: `0.96 tok/s`
  - KV blocks: `3231`
  - memory after load: `[69817, 69817] MB` on GPUs 0 and 1
  - memory after generation: `[69819, 69819] MB`
- The model produced coherent text for the KV cache compression prompt.

### 2026-03-26 22:58 Asia/Shanghai

- Ran `turboquant/proof.py` with the requested model path and conservative
  settings (`TP=2`, `max_model_len=4096`, `gpu_mem=0.85`).
- TurboQuant phase succeeded:
  - hooks installed on `10` full-attention layers
  - baseline and TurboQuant outputs matched at a high level
  - `free_kv_cache()` reported `34,938 MB` freed per GPU and `69.9 GB` total
- Credibility issue found in the current proof harness:
  - the script still uses a 27B-specific hard-coded block size (`784`)
  - the reported capacity improvement is therefore not trustworthy for
    Qwen3.5-35B-A3B
  - `nvidia-smi` only changed by a few MB after free, which means allocator
    behavior and metric choice need deeper inspection

### 2026-03-26 23:05 Asia/Shanghai

- Added `turboquant.report_utils` plus targeted tests to remove stale 27B math
  from `proof.py` and `benchmark.py`.
- Verified helper tests pass:
  - `4 passed` in `test_report_utils.py`
- Re-ran `proof.py` successfully with corrected runtime block size:
  - block size observed at runtime: `1056`
  - corrected theoretical baseline capacity: `3,411,936` tokens
  - corrected theoretical TurboQuant capacity: `6,823,872` tokens
  - corrected theoretical improvement: `2.00x`

### 2026-03-26 23:19 Asia/Shanghai

- Ran a structural KV-cache inspection script.
- Evidence gathered:
  - all `10` hooked full-attention layers point to distinct KV storage regions
    of about `3.49 GB` each per rank
  - after `free_kv_cache()`, the visible layer and runner cache tensors become
    tiny `int8[1]` placeholders
  - despite that replacement, CUDA `memory_allocated()` does not decrease on
    this stack
- Interpretation:
  - the current implementation definitely rewires the obvious Python-visible KV
    cache references
  - some additional live references or backend ownership path still keeps the
    underlying storage alive

### 2026-03-26 23:32 Asia/Shanghai

- Ran the corrected `benchmark.py` on the requested model path.
- Results:
  - baseline throughput: `18.4 tok/s`
  - TurboQuant throughput: `17.9 tok/s`
  - throughput ratio: `0.97x`
  - quality prompt outputs matched at a high level
- Current bottom line:
  - TurboQuant runs on this model
  - theoretical capacity gain is about `2.00x`
  - verified allocator-visible memory recovery has not yet been demonstrated
