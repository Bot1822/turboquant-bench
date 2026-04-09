# TurboQuant Qwen3.5-35B-A3B Execution Plan

## Goal

Produce a reproducible run procedure and at least one benchmark comparison
between baseline vLLM and TurboQuant using the requested model if the model and
hardware support the current implementation.

## Execution Plan

### Phase 1: Environment Bring-up

1. Create a `uv` virtual environment under `vllm/.venv`.
2. Install local `vllm` in editable mode following repo guidance.
3. Install the local `turboquant` package into the same environment.
4. Verify imports for `vllm`, `torch`, and `turboquant`.

### Phase 2: Baseline Validation

1. Select free GPUs and avoid the two GPUs already occupied by another vLLM
   workload.
2. Run a minimal text-only baseline on
   `/share/models/official/Qwen3.5-35B-A3B`.
3. Capture:
   - success/failure
   - load-time VRAM footprint
   - generation-time VRAM footprint
   - generated text

### Phase 3: TurboQuant Bring-up

1. Inspect hook installation count on the requested model.
2. Confirm whether only full-attention layers are hooked.
3. Run a minimal TurboQuant generation path.
4. If it fails, patch the smallest possible surface to restore compatibility and
   document each change.

### Phase 4: Benchmark Adaptation

1. Remove hard-coded 27B model names and paths from benchmark scripts.
2. Replace hard-coded full-attention assumptions with target-model values.
3. Ensure the benchmark reports only metrics that are valid for this model.

### Phase 5: Benchmark Execution

1. Run baseline and TurboQuant in separate processes on the same GPU set.
2. Compare:
   - tokens/s
   - memory per GPU after generation
   - memory freed by `free_kv_cache()`
   - qualitative output sanity
3. If feasible, attempt a higher `max_model_len` or capacity estimate to show
   the practical effect of freed KV memory.

### Phase 6: Verification And Reporting

1. Re-run the final commands from a clean shell invocation.
2. Write exact commands and observed results into `04-experiments.md`.
3. Summarize what works, what was changed, and what remains limited.
