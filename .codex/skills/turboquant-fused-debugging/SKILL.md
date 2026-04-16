---
name: turboquant-fused-debugging
description: Use when TurboQuant fused-path startup, KV cache sizing, or fp8-vs-TurboQuant performance results look wrong, especially when investigating CUDA graph capture failures, hybrid-model page-size anomalies, or cases where fused is expected to beat unfused but does not.
---

# TurboQuant Fused Debugging

## Overview

Use this skill for the project’s two recurring TurboQuant failure classes: KV cache size not matching theory, and fused path startup/performance not matching expectations. The key rule is to separate capacity bugs, capture-safety bugs, and kernel-speed bugs instead of treating them as one problem.

## When to Use

- `GPU KV cache size` is unexpectedly close to `fp8`
- `TQ4_USE_FUSED_PAGED=1` fails during startup
- logs mention `operation not permitted when stream is capturing`
- fused starts but is not meaningfully faster than unfused
- you need to determine whether a failure is model-specific or fused-path specific
- you need to pick a clean GPU and host before reproducing the issue

## Debug Order

### 1. Separate the problem class first

Decide which bucket you are in:

- KV cache sizing anomaly
- fused startup failure
- fused performance regression

Do not mix them.

Before reproducing any of them, use `$gpu-selection-runbook` to pick the
emptiest local or remote GPU so contention does not contaminate the diagnosis.

### 2. KV cache sizing anomaly

Trace this chain end to end:

1. packed slot bytes
2. per-token attention page bytes
3. hybrid/dense block-size calculation
4. final `GPU KV cache size`

For hybrid models, always inspect:

- `num_key_value_heads`
- `head_dim`
- whether the model has linear/mamba/gated-delta-net state
- `Setting attention block size ...`
- `Padding mamba page size ...`

Project-learned root cause:

- if TQ4 slot bytes are over-padded, the per-token page size can collapse to `fp8` parity and erase the expected cache advantage

### 3. Fused startup failure

Use a non-`--rm` debug container and capture full logs.

Look for:

- `operation not permitted when stream is capturing`
- `cudaErrorStreamCaptureInvalidated`
- `Engine core initialization failed`
- stack frames in:
  - `determine_available_memory()`
  - `profile_cudagraph_memory()`
  - `_warmup_and_capture()`
  - `torch.cuda.graph(...).capture_end()`

If those appear, treat the issue as capture-safety, not as a generic model incompatibility.

### 4. Model-specific vs path-specific

When a model fails, rerun the same experiment on a simpler dense model.

Project pattern:

- hybrid Qwen3.5 explained KV cache sizing anomalies
- but dense Qwen3 also reproduced the fused+cudagraph failure

So:

- hybrid can explain sizing complexity
- hybrid alone does not prove the fused path is the root cause

### 5. Fused performance regression

Once fused can start, compare:

- `fp8`
- `tq-unfused`
- `tq-fused`

at the same:

- image tag
- model
- endpoint
- prompt length
- output length
- request concurrency

If fused only improves by about `1%` over unfused while both remain far behind `fp8`, the bottleneck is not cache sizing anymore. It is online compute cost in the TQ path.

## Quick Reference

### Key log signatures

| Symptom | Meaning |
| --- | --- |
| `Setting attention block size to ...` | hybrid block-size logic changed |
| `GPU KV cache size: ... tokens` | final capacity truth |
| `operation not permitted when stream is capturing` | fused path not capture-safe |
| `Engine core initialization failed` | startup aborted after worker failure |

### Decision table

| Observation | Next step |
| --- | --- |
| TQ cache size ≈ fp8 | inspect slot/page/block calculations |
| fused dies before ready | rerun in debug container without `--rm` |
| fused dies in capture stack | disable or downgrade cudagraph support |
| fused starts but not faster | benchmark against unfused and fp8 under same load |

## Common Mistakes

- Assuming hybrid architecture explains every fused failure
- Benchmarking fused vs unfused before verifying final KV cache size
- Looking only at throughput and ignoring startup/capture logs
- Treating eager-only fused numbers as proof of non-eager production viability
- Changing multiple things at once: model, image, backend mode, and workload

## References

See `references/checklist.md` for concrete debug steps and the key formula chain.
