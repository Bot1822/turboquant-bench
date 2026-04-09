# TurboQuant Qwen3.5-35B-A3B Bring-up And Benchmark Design

## Scope

This document captures the design for validating and benchmarking the existing
TurboQuant integration against the requested model
`/share/models/official/Qwen3.5-35B-A3B`.

## Working Assumptions

- The implementation is likely tuned for a different model and may require
  adaptation.
- The fastest path is to preserve the author's execution flow first, then
  extend it to the requested model.
- Benchmark evidence should include both correctness sanity checks and resource
  or throughput improvements.

## Confirmed Constraints

- The target model is not the README-tested Qwen3.5-27B path. It is
  `/share/models/official/Qwen3.5-35B-A3B`.
- The target model is multimodal at the wrapper level:
  `Qwen3_5MoeForConditionalGeneration`.
- The text stack is a hybrid model with 40 layers:
  - 30 `linear_attention`
  - 10 `full_attention`
- TurboQuant's active decode path is implemented only for flash/full-attention
  layers in `turboquant/turboquant/vllm_attn_backend.py`.
- Existing benchmark scripts contain model-specific constants copied from the
  27B experiment and cannot be used as-is for trustworthy claims.

## Success Criteria

The work is considered successful if all of the following are achieved:

1. A baseline vLLM run completes on the requested model with a text-only prompt.
2. TurboQuant hooks install and a TurboQuant run completes on the same model.
3. The TurboQuant path demonstrates at least one measurable benefit with logged
   evidence:
   - reduced resident KV memory after free, or
   - increased feasible context/cache capacity estimate, or
   - improved sustained throughput in a comparable setup.
4. All commands, assumptions, failures, fixes, and experiment outputs are
   recorded in this directory.

## Approach Options

### Option A: Direct local editable setup on the checked-out vLLM tree

Install vLLM into a local `uv` virtual environment, install the local
TurboQuant package into the same environment, then adapt and run the existing
bring-up scripts.

Pros:
- Fastest path to using the already checked-out code
- Simplest for live debugging and patching
- Easiest to verify that local source modifications are actually used

Cons:
- Initial vLLM setup may take time
- If the local environment is inconsistent, debugging packaging issues can be
  noisy

### Option B: Run inside a container with mounted source trees

Use a CUDA-enabled container, mount `vllm/`, `turboquant/`, and the model path,
then install editable packages and run benchmarks inside the container.

Pros:
- Cleaner dependency isolation
- Easier to reproduce once stabilized

Cons:
- More setup overhead before first signal
- Container image choice and CUDA compatibility still need care

### Option C: Start from baseline-only validation and defer TurboQuant until the
model architecture is proven compatible

Run only baseline vLLM on the requested model first, then analyze whether the
hybrid full-attention layers are enough to justify the TurboQuant path before
patching scripts.

Pros:
- Lowest-risk first step
- Quickly distinguishes environment issues from TurboQuant issues

Cons:
- Delays discovery of TurboQuant-specific integration breakage
- Does not directly satisfy the end-to-end goal

## Recommended Approach

Use a staged version of Option A:

1. Create a local `uv` environment in `vllm/`
2. Install local `vllm` and local `turboquant` into that environment
3. Prove baseline inference on the requested model with text-only prompts
4. Adapt the TurboQuant scripts to remove 27B-specific assumptions
5. Run TurboQuant bring-up on the requested model
6. Run benchmark comparisons and log evidence
7. Fall back to Option B only if local editable setup fails for environment
   reasons rather than model/code reasons

This is the best trade-off between speed, debuggability, and confidence that
the final benchmark actually exercises the checked-out source code.

## Planned Technical Changes

- Parameterize model path, TP degree, and GPU set for the requested hardware.
- Replace hard-coded model metadata in benchmark/proof scripts with model-driven
  values or explicit configuration.
- Make capacity estimation aware of the requested model's hybrid layer layout.
- Keep baseline and TurboQuant runs in separate processes to reduce allocator
  cross-talk.
- Record VRAM before generation, after generation, and after `free_kv_cache()`.
- Prefer text-only prompts for the first successful run to avoid multimodal
  processor complexity while still exercising the requested model.
