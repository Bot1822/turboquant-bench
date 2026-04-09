# TurboQuant Qwen3-30B-A3B-Instruct Experiment Design

## Scope

Design and run a reproducible experiment suite for TurboQuant on
`Qwen3-30B-A3B-Instruct-2507` under `2 x A100 80GB`, with:

1. KV-cache reduction as the primary claim
2. multi-user long-dialog capacity as the secondary claim

## Confirmed Context

- Target model: `/share/models/official/Qwen3-30B-A3B-Instruct-2507`
- Model family: `Qwen3MoeForCausalLM`
- Size / topology from local model card and config:
  - 48 layers
  - 32 query heads
  - 4 KV heads
  - head dimension 128
  - native context length 262,144
- This is a text-only causal LM model, which makes it cleaner than the
  previously-tested multimodal Qwen3.5-35B-A3B path for isolating KV-cache
  behavior.

## Goals

### Primary Goal

Prove that TurboQuant reduces effective KV-cache occupancy on this model.

This must be shown with two distinct evidence types:

1. **Direct KV-occupancy evidence**
   - What baseline KV tensors exist for TurboQuant-hooked layers
   - How many bytes the active baseline KV state occupies for the same requests
   - How many bytes the compressed TurboQuant KV representation occupies
   - The implied compression ratio as a function of active sequence length and
     concurrency

2. **Integration/runtime evidence**
   - Whether allocator-visible memory drops after the original KV tensors are
     replaced
   - If allocator-visible memory does not drop, whether the supposedly-freed
     capacity can still be consumed by subsequent requests inside the same live
     engine after the first wave remains resident

Primary-goal rule:
- direct compressed-KV occupancy reduction is the **main proof** of the paper
  idea
- runtime reuse is the **main proof** of the current vLLM integration quality
- if runtime-reuse evidence is absent, the honest result is:
  - “TurboQuant compresses KV representation successfully”
  - but “this integration does not yet fully convert that into reusable runtime
    memory”

### Secondary Goal

Show that the KV reduction, if real, converts into higher multi-user long-dialog
capacity under `2 x A100 80GB`.

### Tertiary Goal

Measure the cost of the method:
- throughput impact
- latency impact
- output quality drift

## What We Need To Avoid

The previous Qwen3.5-35B-A3B track already showed a failure mode:
- visible KV tensor replacement succeeded
- internal `freed_bytes` numbers were large
- allocator-visible memory did not drop

Therefore, this new experiment must **not** rely on a single metric like
`freed_bytes` or `nvidia-smi`. Any one metric alone can mislead.

## Approach Options

### Option A: Direct proof by memory counters only

Run baseline and TurboQuant, compare:
- `nvidia-smi`
- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`

Pros:
- Fast and simple

Cons:
- Too weak by itself
- Previous experiments already showed that these counters may not reflect the
  implementation's own claimed freeing behavior

### Option B: Structural proof only

Inspect model runner state and count:
- baseline KV tensor sizes for hooked layers
- compressed TQ cache sizes
- replaced placeholder tensor sizes after free

Pros:
- Strong evidence that the implementation changes the cache representation

Cons:
- Does not prove runtime capacity is actually reusable

### Option C: Two-layer proof with capacity stress follow-up

1. Structural proof that TurboQuant replaces large hooked-layer KV storage with
   a compressed representation
2. Runtime proof attempt using allocator counters
3. If runtime counters stay flat, use a stress test to ask the real system
   question:
   can TurboQuant admit longer or more concurrent long-history requests before
   failure?

Pros:
- Strongest design
- Separates implementation truth from metric ambiguity
- Produces an honest result even if allocator counters remain stubborn

Cons:
- More work
- Requires custom workload harnesses

## Recommended Approach

Use **Option C**.

This aligns with the user's stated priority:
- first prove KV-cache occupancy reduction
- then prove improved multi-user long-dialog capacity

## Workload Model

The multi-user long-dialog scenario will be approximated using **batched
concurrent requests with long chat histories**.

This is a deliberate approximation:
- each prompt will be a multi-turn chat transcript built with the model's chat
  template
- multiple such prompts will be submitted together in one generation batch
- this stresses active KV-cache footprint in the same direction as many live
  users carrying long histories

Limitation:
- it does not preserve server-side session state across turns
- it is therefore a proxy for concurrent long-history inference, not a full
  online-chat stateful serving benchmark

This limitation is acceptable for the current goal because the main quantity of
  interest is active KV occupancy under long histories.

## Experiment Groups

### Group 1: Compatibility And Hook Coverage

Purpose:
- confirm the model runs under baseline vLLM
- confirm TurboQuant hooks install
- measure how many layers are actually hooked

Outputs:
- run success/failure
- hook count
- basic output sanity

### Group 2: Structural KV Occupancy Proof

Purpose:
- quantify baseline hooked-layer active KV occupancy
- quantify compressed TQ active KV occupancy
- compute compression ratio across pressure points
- quantify the hooked fraction versus total model KV footprint

Metrics:
- hooked layer count
- total attention layer count
- active baseline KV bytes per rank
- TQ compressed key bytes per rank
- TQ compressed value bytes per rank
- total TQ active bytes per rank
- direct occupancy compression ratio
- hooked fraction of model KV layers

### Group 3: Runtime Memory-Reuse Proof

Purpose:
- test whether the compressed KV representation actually replaces or bypasses
  the original paged KV storage in a way that creates reusable runtime memory

Metrics:
- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`
- `nvidia-smi`
- success/failure of a follow-up request wave after `free_kv_cache()`
- whether wave 1 is still resident when wave 2 is admitted

Interpretation rule:
- if counters drop, that is direct runtime evidence
- if counters do not drop but higher follow-up load succeeds only under TQ,
  that is indirect runtime evidence
- if neither happens, we must not claim verified runtime memory recovery, even
  if direct occupancy compression is strong

Required structure:
1. admit wave 1 of long-history requests
2. keep wave 1 resident or otherwise prevent normal teardown from explaining the
   result
3. call `free_kv_cache()`
4. attempt wave 2 admission in the same engine
5. compare baseline and TQ under the same sequence

This avoids the false-positive case where extra admission happens only because
wave 1 has already completed and released resources naturally.

The cross-process colocation experiment may still be useful as an internal
debugging probe, but it is **not part of the final evidence chain** because it
is not the target serving scenario.

### Group 4: Multi-User Long-Dialog Capacity

Purpose:
- show whether TurboQuant increases the number of concurrent long-history
  requests the system can sustain

Metrics:
- max successful concurrency at fixed long-history length in one-shot mode
- max successful history length at fixed concurrency in one-shot mode
- max additional admitted concurrency in the steady-state same-engine wave-2
  test
- throughput and latency at each point
- failure mode when the system crosses the limit

### Group 5: Cost And Quality

Purpose:
- ensure the method is not buying capacity via unacceptable quality or speed
  regressions

Metrics:
- output comparisons on fixed sanity prompts
- throughput ratio
- latency ratio

## Proposed Pressure Points

The pressure should be introduced in two dimensions:

1. **History length ladder**
   - start with moderate long-context prompts
   - increase to the point where baseline approaches failure

2. **Concurrency ladder**
   - at each chosen history length, increase concurrent requests until failure,
     timeout, or severe collapse

This avoids relying on one cherry-picked point.

But it is not sufficient by itself for the secondary claim, because one-shot
batched long histories mainly stress initial prefill. Since TurboQuant only
tries to release the original KV tensors after data has already entered the TQ
store, one-shot admission can understate or entirely miss the benefit.

Therefore, the final design must contain:
- one-shot pressure sweeps for coarse limits
- plus a steady-state same-engine wave-2 admission test for reusable-capacity
  proof

## Initial Ladder Recommendation

These are starting points, not fixed commitments. The actual pressure levels
should be refined after the first calibration run.

- History length targets: `16k`, `32k`, `48k`, `64k`, `96k` input tokens
- Output length targets: `32` or `64` tokens
- Concurrency ladder: `1`, `2`, `4`, `8`, `12`, `16`

If baseline fails too early, the ladder moves downward. If baseline is too
comfortable, the ladder moves upward.

## Success Criteria

We can claim success only if all three statements are defended by evidence:

1. TurboQuant reduces direct KV-cache occupancy on this model.
2. That reduction improves concurrent long-history serving capacity or
   stability under memory pressure on `2 x A100 80GB`, or we explicitly report
   that the current integration fails to convert occupancy reduction into
   system-level benefit.
3. The performance or quality trade-off remains acceptable and explicitly
   quantified.

## Failure Criteria

Any of the following blocks the higher-level claim:

- TurboQuant cannot install hooks or generate correct output on this model
- hook coverage is too small to matter
- structural compression is real but no runtime-reuse signal appears
- runtime-reuse signal appears, but no measurable system-level gain follows
- runtime memory counters and capacity stress both fail to show any benefit
- quality or throughput degradation dominates the capacity gain

## A/B Control Rules

Every baseline vs TurboQuant comparison must hold constant:

- model path
- tensor parallel size
- `gpu_memory_utilization`
- `max_model_len`
- prompt set and token lengths
- output token budget
- process isolation between baseline and TQ

Forbidden shortcuts:
- changing memory budget between baseline and TQ
- using stale block-size constants instead of runtime values
- using one-shot prefill failure as the only “capacity” result
