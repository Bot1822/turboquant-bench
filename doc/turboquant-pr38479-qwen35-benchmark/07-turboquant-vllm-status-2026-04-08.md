# `turboquant-vllm` Detailed Status

Date: `2026-04-08`

## Executive Summary

`turboquant-vllm` is currently the most active public community implementation
of TurboQuant for the vLLM ecosystem, but it is not the same thing as upstream
native TurboQuant support in vLLM.

Its current identity is:

- a pip-installable Python package
- a `vllm.general_plugins` plugin that registers a custom attention backend
- a HuggingFace `DynamicCache` compression library
- a fast-moving, single-maintainer project with aggressive release cadence

It looks good for external smoke tests and exploratory reproduction, but it is
not yet the cleanest proxy for the upstream-native A100 path we have been
testing in PR `#38479`.

## 1. Packaging and Release State

### What exists today

- The package is published on PyPI as `turboquant-vllm`.
- The latest public PyPI release is `1.5.0`, released on `2026-04-08`.
- Python requirement is `>=3.12`.
- The package currently declares a direct dependency on `vllm>=0.19`.
- Its optional extra `vllm` is slightly looser and declares `vllm>=0.18`.
- The package advertises itself as `Development Status :: 5 - Production/Stable`
  on PyPI.
- Installation is intentionally simple:
  - `pip install turboquant-vllm[vllm]`
  - or `uv add turboquant-vllm --extra vllm`

Practical interpretation:

- the safest stated target today is `vLLM 0.19.x` or newer
- I would not assume compatibility with older `0.8` / `0.9` era vLLM branches
- the `>=0.18` versus `>=0.19` metadata split suggests recent packaging churn,
  so if reproducing today, align to `vLLM 0.19.x` first

### What the release cadence says

PyPI release history shows the project moved from `0.1.0` and `1.0.0` on
`2026-03-27` to `1.5.0` on `2026-04-08`, with ten published versions in about
twelve days.

Interpretation:

- positive signal: the maintainer is shipping fixes quickly
- risk signal: the project is still stabilizing and APIs or behavior may move
  quickly

I would treat it as operationally promising but still early-stage, even though
the PyPI classifier says `Production/Stable`.

## 2. What It Actually Implements

### It is not just a vLLM patch

The repository is broader than a single vLLM backend:

- `src/turboquant_vllm/kv_cache.py`
  - HuggingFace `DynamicCache` integration
- `src/turboquant_vllm/triton/`
  - Triton compress, decompress, and fused attention kernels
- `src/turboquant_vllm/vllm/`
  - vLLM plugin-side serving integration

The repo documentation now explicitly frames the project as the reference
implementation for HuggingFace `DynamicCache` workflows, while pointing users to
upstream vLLM PR `#38479` for the native production vLLM path.

Interpretation:

This is now best understood as a hybrid project:

- practical HF reference implementation first
- community vLLM plugin second
- not the canonical upstream-native backend

## 3. How the vLLM Integration Works

### Integration model

The package registers through vLLM's plugin mechanism:

- entry point: `vllm.general_plugins`
- backend name: `CUSTOM`
- expected launch style:
  - `vllm serve <model> --attention-backend CUSTOM`

This is an important distinction from upstream PR `#38479`:

- PR `#38479` adds native TurboQuant handling inside vLLM's internal attention
  stack
- `turboquant-vllm` plugs in from outside through the custom backend extension
  point

### Current serving shape

The implemented vLLM backend is named `TQ4AttentionBackend` /
`TQ4AttentionImpl`.

The current public code path does three main things:

1. compresses incoming KV to packed bytes
2. stores packed bytes in a smaller KV layout
3. decompresses back to FP16 and delegates to Flash Attention

There is also a more optimized fused paged decode path in the codebase, but it
is feature-gated:

- requires successful Triton kernel import
- requires `TQ4_USE_FUSED_PAGED`
- requires symmetric K/V bits

An INT8 prefill path also exists but is separately gated by
`TQ4_USE_INT8_PREFILL`.

Interpretation:

The plugin already has a useful robust fallback path, which is one reason it
looks more operationally stable than PR `#38479`. But the fastest path is still
optional rather than the obvious always-on default.

## 4. Compression Mode Reality

### It is centered on a TQ4 serving backend, not upstream-style `tq3`/`tq4`

The public vLLM backend is explicitly a `TQ4` backend.

The README exposes asymmetric K/V settings with:

- `TQ4_K_BITS`
- `TQ4_V_BITS`

But the backend's packed storage layout remains nibble-packed and its page-size
logic states that storage is independent of bit-width in the vLLM path.

Interpretation:

- this is not the same user model as upstream vLLM `kv_cache_dtype=tq3/tq4`
- it is better thought of as a fixed packed serving format with configurable
  quantizer codebooks
- if your next experiments are specifically about `tq3` operating points, this
  plugin is not a direct apples-to-apples replacement

## 5. Claimed Validation Surface

### Publicly claimed validated models

The README currently lists per-layer cosine checks on these eight model
families:

- Llama 3.1 8B
- Qwen2.5 3B
- Mistral 7B
- Phi-3-mini
- Phi-4
- Gemma 2 2B
- Gemma 3 4B
- Molmo2 4B

The docs and release notes also show recent hardening for:

- sliding-window Gemma behavior
- heterogeneous `head_dim`
- hybrid model page alignment

### What I did not find

I did not find public evidence in the repo README, release notes, or docs that
the plugin has already been validated on:

- Qwen3.5
- Qwen3.5-35B-A3B
- A100 as a primary benchmark target

The public performance and quality reporting is centered mostly on RTX 4090 and
consumer-GPU workflows, with additional AMD ROCm experimentation.

Interpretation:

For your current workload, `turboquant-vllm` is relevant but not directly
pre-validated. There is still real uncertainty around:

- A100-specific kernel behavior
- large-Qwen compatibility beyond the published Qwen2.5 example
- whether its consumer-GPU tuning transfers well to your setup

## 6. Testing and Hardening Signals

### Positive signals

- The repo contains a substantial test suite, including dedicated vLLM backend
  tests and GPU-oriented Triton tests.
- The documentation describes a high coverage bar and explicit test markers.
- Recent releases show bug-fix activity on real serving edge cases instead of
  only marketing updates.

Examples from recent releases:

- `2026-03-29` `1.2.0`
  - fused paged TQ4 decode kernel
  - INT8 prefill kernel
  - CUDA-graph buffer pre-allocation
- `2026-03-30` `1.2.1` and `1.2.2`
  - bounded scratch buffers for prefill and decode
- `2026-03-31` `1.3.0`
  - sliding-window Gemma bypass
  - support for `head_dim` `64/96`
  - added Gemma and Qwen2.5 validation
- `2026-04-08` `1.5.0`
  - full-attention-layer bypass in sliding-window models
  - per-layer `head_dim` support
  - hybrid-model page-alignment fix

### Negative / caution signals

- The repository is very new.
- Public issue volume is small, but the project has not yet had time to prove
  long-term stability.
- It still appears heavily maintainer-driven rather than battle-tested by a
  wide production user base.

As of the open issues page on `2026-04-08`, the visible open issues were four
low-priority test / housekeeping items rather than major runtime failures. That
is encouraging, but it is not enough by itself to prove production maturity.

## 7. Performance Positioning

### What the project currently claims

The README's serving example for `Llama-3.1-8B-Instruct` on RTX 4090 at `200`
concurrent requests says:

- request throughput: `-7.3%`
- output tok/s: `-7.3%`
- median TTFT: `-25.2%`
- median TPOT: `+201%`

The README explicitly interprets this as:

- better TTFT from smaller cache pages
- worse decode latency from online decompression
- best fit for memory-bound regimes

Interpretation:

This is broadly consistent with what we already observed on PR `#38479`:
TurboQuant can be useful because it changes the memory operating point, not
because it is guaranteed to make decode faster in every regime.

## 8. Relationship to PR `#38479`

The project documentation now explicitly says that descriptions of planned
production vLLM serving integration are superseded by upstream PR `#38479`.

That is a strong signal about intended positioning:

- `turboquant-vllm` is no longer claiming to be the final native vLLM answer
- it is positioning itself as the practical external implementation around
  HuggingFace and plugin-based integration
- upstream native-kernel work is still expected to live in vLLM itself

## 9. Bottom-Line Assessment for Our Work

For your current use case:

### Good reasons to test it

- easy to install and reproduce
- faster iteration than rebuilding a custom vLLM branch
- likely better out-of-box robustness for smoke tests
- useful as a second implementation to compare against PR `#38479`

### Reasons not to treat it as the main benchmark target

- not a direct replacement for upstream `tq3`
- public validation is not centered on Qwen3.5 or A100
- vLLM integration is plugin-side and architecturally different from PR
  `#38479`
- public docs themselves defer native production vLLM claims to upstream PR
  `#38479`

## Recommendation

Use `turboquant-vllm` as:

- a community baseline
- a robustness cross-check
- a quick install / quick smoke-test path

Do not use it as the only source of truth for:

- A100 TurboQuant behavior
- upstream-native vLLM TurboQuant performance
- `tq3` versus `tq4` operating-point analysis

## Sources

- PyPI project page:
  - <https://pypi.org/project/turboquant-vllm/>
- GitHub repository:
  - <https://github.com/Alberto-Codes/turboquant-vllm>
- README:
  - <https://github.com/Alberto-Codes/turboquant-vllm/blob/main/README.md>
- Releases:
  - <https://github.com/Alberto-Codes/turboquant-vllm/releases>
- Open issues:
  - <https://github.com/Alberto-Codes/turboquant-vllm/issues>
- Raw `pyproject.toml`:
  - <https://raw.githubusercontent.com/Alberto-Codes/turboquant-vllm/main/pyproject.toml>
- Raw vLLM plugin entry point:
  - <https://raw.githubusercontent.com/Alberto-Codes/turboquant-vllm/main/src/turboquant_vllm/vllm/__init__.py>
- Raw backend implementation:
  - <https://raw.githubusercontent.com/Alberto-Codes/turboquant-vllm/main/src/turboquant_vllm/vllm/tq4_backend.py>
- Raw architecture doc:
  - <https://raw.githubusercontent.com/Alberto-Codes/turboquant-vllm/main/docs/ARCHITECTURE.md>
- Raw roadmap doc:
  - <https://raw.githubusercontent.com/Alberto-Codes/turboquant-vllm/main/docs/ROADMAP.md>
- Raw project overview:
  - <https://raw.githubusercontent.com/Alberto-Codes/turboquant-vllm/main/docs/project-overview.md>
- Raw source-tree analysis:
  - <https://raw.githubusercontent.com/Alberto-Codes/turboquant-vllm/main/docs/source-tree-analysis.md>
