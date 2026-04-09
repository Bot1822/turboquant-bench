# `turboquant-vllm` on vLLM 0.19 Comparison Track

Date: `2026-04-08`

## Goal

Evaluate `turboquant-vllm` inside a separate official vLLM `0.19` container and
compare its behavior against the previously recorded PR `#38479` results on the
same local model family.

## Design

This comparison is intentionally conservative.

It is not a perfect apples-to-apples benchmark because:

- PR `#38479` was benchmarked on a different vLLM code line
- `turboquant-vllm` uses a plugin-side `CUSTOM` attention backend rather than
  upstream-native `kv_cache_dtype=tq3/tq4`
- the plugin exposes a TQ4-style operating point rather than the PR's `tq3`
  path

The immediate question for this track is narrower:

1. Can `turboquant-vllm` actually initialize and run on
   `/share/models/official/Qwen3.5-35B-A3B` in an official vLLM `0.19` image?
2. If yes, how far is its minimal decode behavior from the already recorded PR
   `#38479` baseline / `tq4` numbers?
3. If no, the failure mode itself is a material gap relative to the PR branch.

## Environment

- Base image: official `vllm/vllm-openai:v0.19.0-x86_64-cu130`
- Container naming: `zgp-*`
- GPU target: first clean A100 (`gpu1` preferred, `gpu3` fallback)
- Model path: `/share/models/official/Qwen3.5-35B-A3B`

## Cases

### Case 1: Baseline Smoke

- vLLM `0.19`
- no plugin backend
- same local model
- short prompt + short decode

Purpose:

- verify the container and model load cleanly on the official `0.19` image

### Case 2: Plugin Smoke

- vLLM `0.19`
- `turboquant-vllm`
- `attention_backend=CUSTOM`
- default TQ4 config first

Purpose:

- determine whether the plugin can initialize on this Qwen3.5 model at all

### Case 3: Minimal Decode Comparison

- same prompt family as the existing benchmark harness
- one or two context lengths only
- compare:
  - `v0.19 baseline`
  - `turboquant-vllm CUSTOM`
  - existing PR `#38479` reference results

Purpose:

- estimate the practical gap without rebuilding a full new harness first

## Interpretation Rules

- If plugin initialization fails on the same model where PR `#38479` already
  ran, that is recorded as a first-order compatibility gap.
- If the plugin runs only on a shorter / easier case, that is recorded as a
  partial compatibility result, not as full parity.
- Any throughput comparison with the PR must be labeled approximate because the
  underlying vLLM versions and integration styles differ.

## Run Log

- `2026-04-08`: confirmed official Docker Hub tags exist for
  `v0.19.0-x86_64` / `v0.19.0-x86_64-cu130`.
- `2026-04-08`: confirmed host NVIDIA driver is `575.51.03`, so there is no
  obvious driver-side reason to avoid a direct official `0.19` image.
- `2026-04-08`: verified that the default official image
  `vllm/vllm-openai:v0.19.0-x86_64` is not usable on this host:
  - inside the container, `torch 2.10.0+cu129` reports
    `torch.cuda.is_available() == False`
  - the underlying failure is `cudaGetDeviceCount()` error `803`
- `2026-04-08`: verified that the official image
  `vllm/vllm-openai:v0.19.0-x86_64-cu130` is usable on this host:
  - inside the container, `torch 2.10.0+cu130` reports
    `torch.cuda.is_available() == True`
- `2026-04-08`: installed `turboquant-vllm==1.5.0` and its missing runtime
  dependencies inside the official `cu130` container.
- `2026-04-08`: official `0.19+cu130` baseline on
  `/share/models/official/Qwen3.5-35B-A3B`:
  - default backend + `float16`: model load succeeds but generation fails with
    `query and key must have the same dtype`
  - `TRITON_ATTN` + `float16`: model load succeeds but generation fails in
    Triton unified attention with an `fp16` vs `uint8` operand mismatch
- `2026-04-08`: `turboquant-vllm` `CUSTOM` backend + `float16` successfully
  loaded and generated on the same model in the same official `0.19+cu130`
  container.
- `2026-04-09`: root-cause follow-up for the official baseline found that
  forcing `kv_cache_dtype=fp8` fixes the broken generation path.
- `2026-04-09`: fixed official baseline measurements:
  - smoke case: load `71.82 s`, TTFT `3.11 s`, decode `13.36 tok/s`
  - `ctx4096` case (`3984` prompt tokens, `32` output):
    - load `67.65 s`
    - TTFT `0.28 s`
    - decode `13.36 tok/s`
    - output throughput `12.32 tok/s`
- `2026-04-08`: minimal plugin decode case at `ctx4096` completed:
  - prompt tokens: `3984`
  - TTFT: `0.75 s`
  - decode throughput: `12.78 tok/s`
  - output throughput: `10.08 tok/s`
- `2026-04-09`: moved the plugin repository to the non-temporary path
  `/ceph/User/E01442/turboquant/turboquant-vllm` and re-ran the same plugin
  `ctx4096` case on `gpu0`.
- `2026-04-09`: rerun results after the path move:
  - same-container rerun on empty `gpu0`:
    - TTFT `0.48 s`
    - decode `12.53 tok/s`
    - output throughput `10.83 tok/s`
  - fresh official container rerun on empty `gpu0`:
    - TTFT `0.89 s`
    - decode `12.16 tok/s`
    - output throughput `9.31 tok/s`
  - second run in that same fresh official container on the same `gpu0`:
    - TTFT `0.36 s`
    - decode `12.78 tok/s`
    - output throughput `11.49 tok/s`

## Interim Conclusion

At this stage the most important gap is not plugin-versus-PR performance but
plugin-versus-official-baseline operability on this specific model.

On the official `0.19+cu130` image and local `Qwen3.5-35B-A3B` model:

- the official baseline `auto` path is broken
- the official baseline becomes usable when `kv_cache_dtype=fp8` is forced
- `turboquant-vllm` `CUSTOM` backend also generates successfully

Against the fixed official baseline at `ctx4096`, the plugin is slower on the
two metrics that matter here:

- baseline `fp8` TTFT `0.28 s` vs plugin TTFT roughly `0.48` to `0.89 s`
- baseline `fp8` decode `13.36 tok/s` vs plugin decode roughly
  `12.16` to `12.78 tok/s`

The follow-up reruns also answer a narrower operational question: moving the
repository out of `.tmp` does not eliminate the plugin TTFT penalty. The plugin
still trails the fixed baseline on TTFT after the move, and the fresh-container
rerun suggests that TTFT is not only worse than baseline but also somewhat
variable across runs.

The strongest local evidence for the variance source is that the plugin did not
reserve a stable KV budget across nominally equivalent `ctx4096` runs and that
the first run inside a fresh container is worse than the second. On the same
GPU and same nominal settings, `num_gpu_blocks` changed across `70`, `226`,
and then `187`; TTFT moved across `0.48 s`, `0.89 s`, and then `0.36 s`. That
pattern points much more strongly to initialization-time memory profiling /
KV-budget instability and first-run warmup state than to sustained third-party
GPU contention or a large change in steady-state decode behavior.

An explicit process check was also performed for the
`zgp-vllm019-cu130-dev` and `zgp-vllm019-cu130-fresh` containers. Both
containers only had `sleep infinity` running when inspected, with no residual
`EngineCore`, `vllm serve`, or benchmark Python process. The GPUs that remained
heavily used after some runs were occupied by external host-side jobs, mainly
`ray::WorkerDict` actors and unrelated long-lived VLLM workers, rather than by
our own stale container processes.

Against the previously recorded PR `#38479` `tq4@4096` result, the plugin also
does not look faster on A100. Its `ctx4096` approximate decode throughput
(`12.78 tok/s`) is slightly below the PR's `tq4` decode throughput
(`13.04 tok/s`), and its TTFT (`0.75 s`) is materially worse than the PR
steady-state `tq4` reading (`0.26 s`). All cross-stack comparisons remain
approximate because the vLLM versions, cache dtypes, and integration styles
differ.
