# PR #38479 Qwen3.5-35B-A3B Benchmark Design

## Goal

Design a reproducible benchmark suite for vLLM PR `#38479` that measures the
practical value of TurboQuant on A100 hardware along three axes:

1. GPU memory savings
2. Decode efficiency
3. Long-context detail retrieval quality

The benchmark must be usable for iterative local debugging, not just one-off
reporting. It must therefore produce structured machine-readable outputs as
well as human-readable experiment logs.

## Target System

- Workspace root: `/ceph/User/E01442/turboquant`
- vLLM PR worktree:
  `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`
- Primary model:
  `/share/models/official/Qwen3.5-35B-A3B`
- Primary hardware:
  NVIDIA A100-SXM4-80GB
- Primary benchmark environment:
  official-image-derived container with local PR source mounted

## Benchmark Question

TurboQuant changes the KV-cache representation. The benchmark should therefore
answer:

- Does `tq3` materially reduce effective KV-cache footprint?
- What does `tq3` cost in decode throughput and latency?
- When context becomes long, does `tq3` reduce the model's ability to retrieve
  exact details from a document?

## Compared Modes

### Primary comparison

- `baseline`
  Standard vLLM path with non-TurboQuant KV cache
- `tq3`
  PR `#38479` TurboQuant 3-bit mode

### Secondary comparison

- `tq4`
  PR `#38479` TurboQuant 4-bit mode

### Why this ordering

`tq3` is the user's requested focus and should be the default experimental
track. `tq4` is included because it is the most natural quality-efficiency
tradeoff checkpoint once `tq3` is characterized.

## Experimental Axes

### 1. Memory axis

Measure memory from three perspectives:

- vLLM-reported KV capacity
- CUDA allocator-visible memory
- `nvidia-smi` resident GPU memory

These do not always move together. The benchmark should explicitly separate
them rather than collapse them into a single number.

#### Primary memory metrics

- model load memory per GPU
- memory after engine initialization per GPU
- memory after a generation run per GPU
- KV-cache token capacity reported by vLLM
- available KV cache memory reported by engine logs

#### Secondary memory metrics

- torch `memory_allocated`
- torch `memory_reserved`
- delta vs baseline

### 2. Decode efficiency axis

TurboQuant's value is limited if it destroys decode efficiency. The benchmark
should therefore isolate decode-time cost, not just total request wall time.

#### Methodology refinement

The benchmark must distinguish between two regimes:

- cold isolated execution
  Useful for reproducing user-visible first-run cost, including kernel
  compilation and engine bootstrap overhead
- warmed steady-state execution
  Useful for evaluating the actual decode path after the runtime kernels are
  already compiled and the engine instance is hot

For the final comparison, the steady-state decode benchmark is the primary
signal. Cold isolated numbers are kept as auxiliary evidence because PR
`#38479` currently depends on runtime-compiled CUDA paths.

#### Primary decode metrics

- TTFT
- decode tokens/s
- mean and median inter-token latency

#### Secondary decode metrics

- prefill tokens/s
- total wall time
- latency percentiles for repeated runs

### 3. Long-document detail retrieval axis

This is the primary "intelligence degradation" metric for the suite.

The benchmark should not use subjective open-ended judging as the main score.
It should use exact or near-exact answer matching so that `baseline`, `tq3`,
and `tq4` are compared on the same deterministic rubric.

## Quality Task Design

### Core task family

Use long-document synthetic retrieval tasks with controlled answer keys.

Each task consists of:

- one long document
- one or more embedded facts
- one question whose answer is an exact span, short string, or number

### Task tiers

#### Tier A: Single-detail retrieval

One target fact embedded in a long noisy document.

Purpose:
- detect whether TurboQuant loses simple recall at long context

Examples:
- order ID
- date
- version string
- person-role mapping

Scoring:
- exact match

#### Tier B: Multi-detail retrieval

Several target facts embedded at distant positions, with the question asking
for one specific field or a small tuple.

Purpose:
- measure detail confusion and interference

Scoring:
- exact match on each requested field
- optional all-fields-correct metric

#### Tier C: Adversarial near-match retrieval

The document includes multiple highly similar distractors:

- similar dates
- similar IDs
- repeated entity names with different attributes
- nearby lexical overlap

Purpose:
- distinguish real retrieval from fuzzy paraphrase behavior

Scoring:
- exact match or regex-normalized match

#### Tier D: Cross-reference retrieval

The answer requires combining two distant facts from the same long document.

Purpose:
- test long-context reasoning over retrieved details, not just copying

Scoring:
- exact structured answer

### Context-length ladder

Quality should be measured across a length ladder rather than a single long
context.

Recommended lengths:

- `4k`
- `8k`
- `16k`
- `32k`
- `64k`

If runtime allows, add:

- `96k`
- `128k`

### Answer formatting

To reduce evaluation noise, prompts should require a constrained answer style:

- `Answer with the exact ID only.`
- `Answer with one date in YYYY-MM-DD format only.`
- `Answer as JSON with keys x and y only.`

This sharply reduces judge ambiguity and makes EM-based comparison viable.

## Workload Matrix

### Matrix dimensions

- mode: `baseline`, `tq3`, `tq4`
- context length: `4k`, `8k`, `16k`, `32k`, `64k`, optional `96k`, `128k`
- prompt type:
  - throughput prompt
  - retrieval prompt
- run repetition:
  - at least `3` runs for throughput
  - at least `N` seeded documents per retrieval tier

### Minimal required matrix

For the first complete benchmark report:

- Memory:
  - `baseline` vs `tq3`
- Decode:
  - `baseline` vs `tq3`
  - lengths `4k`, `16k`, `32k`, `64k`
- Quality:
  - `baseline` vs `tq3`
  - Tiers A/B/C
  - lengths `8k`, `32k`, `64k`

### Extended matrix

After the minimal report is stable:

- add `tq4`
- add Tier D cross-reference retrieval
- add repeated latency runs

## Benchmark Approaches

### Approach A: Reuse and adapt the existing local benchmark harness

Use the existing benchmark scripts in `turboquant-test/` as scaffolding for:

- process isolation
- metric collection
- result serialization

Pros:
- fastest path to a working harness
- already aligned with this workspace's history

Cons:
- existing scripts were written for a different integration style and may carry
  assumptions that do not match PR `#38479`

### Approach B: New benchmark harness directly against PR `#38479`

Build a new benchmark package under `doc/...` or a dedicated helper module that

### Retrieval execution strategy refinement

Retrieval quality should not be measured with one model load per example. That
would mostly benchmark engine startup variance and make long-context sweeps too
slow to iterate on. Instead:

- generate all examples for a target context length up front
- launch one container / one engine per `mode x context`
- run every example in that batch through the same hot engine
- record both per-example exact-match scores and per-batch aggregate accuracy

This keeps quality comparison fair while making long-context experiments
practical on local A100 hardware.
uses the source-built PR container as the only execution backend.

Pros:
- cleaner measurement model
- fewer inherited assumptions

Cons:
- slower to build
- more code to validate

### Approach C: Hybrid approach

Reuse useful pieces from `turboquant-test/benchmark.py` and supporting helpers,
but build a new PR-`#38479`-specific harness and dataset generator.

Pros:
- fastest path with better control
- good balance of reuse and correctness

Cons:
- requires careful separation of reusable helpers from legacy assumptions

## Recommended Approach

Use Approach C.

Reasoning:

- We already have evidence that the PR-backed vLLM path can run end-to-end on
  local Qwen3.5 with `tq3`.
- Existing helper scripts in `turboquant-test/` are useful for subprocess
  isolation and result structuring.
- The benchmark target here is not the legacy hook-based TurboQuant code path;
  it is the PR `#38479` source-backed vLLM implementation. Reusing too much of
  the old harness without adaptation would blur that distinction.

## Benchmark Architecture

### Components

#### 1. Environment runner

Responsibilities:

- launch the PR-backed benchmark commands in the validated container
- inject the required `LD_LIBRARY_PATH`
- select GPU placement
- capture stdout/stderr and structured JSON outputs

#### 2. Retrieval dataset generator

Responsibilities:

- generate seeded long documents with deterministic fact placement
- emit exact answer keys
- support multiple tiers and context sizes

#### 3. Benchmark executor

Responsibilities:

- run baseline / `tq3` / `tq4`
- collect metrics
- retry or mark failures cleanly
- separate quality and throughput runs

#### 4. Result aggregator

Responsibilities:

- compute summary statistics
- emit JSON/CSV/markdown tables
- compare modes side by side

#### 5. Experiment recorder

Responsibilities:

- append every command and observed result to `04-experiments.md`
- keep one-line outcome summaries for each run

## Success Criteria

The benchmark project is successful when all of the following are true:

1. A reproducible baseline and `tq3` benchmark can be run from documented
   commands.
2. Memory metrics are captured in a structured format.
3. Decode metrics are captured in a structured format.
4. Long-document retrieval accuracy is captured with deterministic scoring.
5. A final report can answer:
   - how much memory does `tq3` save?
   - how much decode speed does it lose or gain?
   - how much retrieval accuracy changes at each context length?

## Risks

- Runtime-compiled CUDA fast paths may introduce high first-run latency.
- GPU memory pressure can vary depending on which GPU is selected.
- Some FA3/TQ runtime compilation is architecture-driven and may not respect
  narrow build-intent expectations.
- `Qwen3.5-35B-A3B` hybrid attention means only part of the stack pays or gains
  from TurboQuant, so gains may be bounded.

## Design Decision Summary

- Primary quality metric:
  long-document exact detail retrieval
- Primary scoring style:
  exact/rules-based
- Primary model:
  `Qwen3.5-35B-A3B`
- Primary mode under study:
  `tq3`
- Secondary comparison:
  `tq4`
- Primary implementation strategy:
  adapt existing local helper scripts but build a PR-`#38479`-specific harness
