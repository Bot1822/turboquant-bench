# PR #38479 Qwen3.5-35B-A3B Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use structured execution and keep all changes and results documented in this benchmark directory. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible benchmark suite that compares baseline vLLM, `tq3`, and `tq4` on memory, decode efficiency, and long-document detail retrieval for `/share/models/official/Qwen3.5-35B-A3B`.

**Architecture:** A PR-`#38479`-specific benchmark harness will run inside the validated container environment, generate deterministic long-context retrieval datasets, execute baseline/TurboQuant runs, and emit structured results plus markdown experiment logs.

**Tech Stack:** Python 3.12, source-built vLLM PR `#38479`, local containers, local Qwen3.5 model, JSONL/CSV/markdown reporting.

---

## Task 1: Create benchmark document skeleton and result files

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/04-experiments.md`
- Modify: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/01-worklog.md`

- [ ] Add experiment log template with run table columns for mode, context length, task tier, GPU, command, outcome, and key metrics.
- [ ] Add initial worklog entries describing benchmark implementation start.
- [ ] Verify both docs render correctly with plain markdown headings and tables.

## Task 2: Extract reusable helper logic from existing local scripts

**Files:**
- Inspect: `/ceph/User/E01442/turboquant/turboquant-test/benchmark.py`
- Inspect: `/ceph/User/E01442/turboquant/turboquant-test/test_experiment_utils.py`
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/benchmark_helpers.py`

- [ ] Identify reusable subprocess/result helpers from `turboquant-test/benchmark.py`.
- [ ] Copy only generic utilities needed for PR-`#38479` execution into a new helper module.
- [ ] Do not carry forward legacy hook-based TurboQuant assumptions.
- [ ] Add docstrings explaining which parts are PR-specific and which are generic.

## Task 3: Implement deterministic long-document dataset generator

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/retrieval_dataset.py`
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/generated/README.md`

- [ ] Implement seeded document generation for Tier A single-detail retrieval.
- [ ] Implement seeded document generation for Tier B multi-detail retrieval.
- [ ] Implement seeded document generation for Tier C adversarial near-match retrieval.
- [ ] Emit prompt, answer key, tier, seed, and approximate context length in structured JSONL or Python dict form.
- [ ] Keep answer format constrained to exact string/number/span outputs.

## Task 4: Add failing tests for dataset and scoring logic

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/test_retrieval_dataset.py`

- [ ] Write a failing test for deterministic dataset generation with a fixed seed.
- [ ] Write a failing test for exact-match scoring on Tier A.
- [ ] Write a failing test for multi-field scoring on Tier B.
- [ ] Run only these tests and confirm they fail for the expected reason before implementing the scoring code.

## Task 5: Implement scoring logic

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/scoring.py`
- Modify: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/test_retrieval_dataset.py`

- [ ] Implement exact-match normalization for string answers.
- [ ] Implement numeric and JSON-like field comparison helpers where needed.
- [ ] Make the tests from Task 4 pass with the smallest scoring surface.
- [ ] Re-run the tests and record the command and result in the worklog.

## Task 6: Implement PR-`#38479` runner for baseline and TQ modes

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/pr38479_runner.py`

- [ ] Implement a runner that launches vLLM in the validated container environment.
- [ ] Support modes:
  - `baseline`
  - `tq3`
  - `tq4`
- [ ] Always set:
  - model path
  - `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu`
- [ ] Allow explicit GPU selection, context length, and prompt payload.
- [ ] Return structured JSON with success/failure, stdout/stderr, and parsed metrics.

## Task 7: Implement memory metric collection

**Files:**
- Modify: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/pr38479_runner.py`
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/memory_metrics.py`

- [ ] Capture `nvidia-smi` memory before startup, after load, after generate.
- [ ] Capture `torch.cuda.memory_allocated()` and `memory_reserved()` where possible.
- [ ] Capture vLLM-reported KV cache capacity and available KV cache memory from logs.
- [ ] Normalize these into one structured result object per run.

## Task 8: Implement decode-efficiency benchmark path

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/decode_benchmark.py`

- [ ] Create a decode benchmark runner using fixed prompt lengths and fixed output lengths.
- [ ] Measure TTFT.
- [ ] Measure decode tokens/s.
- [ ] Measure per-token latency where possible.
- [ ] Support multiple context lengths in a single sweep.

## Task 9: Implement retrieval benchmark path

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/retrieval_benchmark.py`

- [ ] Generate retrieval documents for target context lengths.
- [ ] Run baseline / `tq3` / `tq4` with the same documents and questions.
- [ ] Score outputs with deterministic exact/rule-based scoring.
- [ ] Emit per-example and aggregate accuracy metrics.

## Task 10: Add a top-level benchmark CLI

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/run_benchmark.py`

- [ ] Add subcommands or flags for:
  - `memory`
  - `decode`
  - `retrieval`
  - `all`
- [ ] Add mode selection for baseline / `tq3` / `tq4`.
- [ ] Add output path arguments for JSON and markdown summaries.
- [ ] Print concise progress logs suitable for long-running local jobs.

## Task 11: Record experiments automatically

**Files:**
- Modify: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/04-experiments.md`
- Modify: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/run_benchmark.py`

- [ ] Append one markdown row per benchmark run with command, result, and key metrics.
- [ ] Include failures, not just successes.
- [ ] Include run timestamp, GPU selection, and mode.

## Task 12: Run minimum benchmark matrix

**Files:**
- Modify: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/04-experiments.md`
- Modify: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/01-worklog.md`

- [ ] Run memory benchmark for `baseline` and `tq3`.
- [ ] Run decode benchmark for `baseline` and `tq3` at `4k`, `16k`, `32k`, `64k` if feasible.
- [ ] Run retrieval benchmark for `baseline` and `tq3` at `8k`, `32k`, `64k`.
- [ ] If runtime permits, add `tq4` to the same matrix.
- [ ] Record every run and result in markdown and machine-readable output.

## Task 13: Write benchmark summary

**Files:**
- Create: `/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/05-summary.md`

- [ ] Summarize memory savings for each mode.
- [ ] Summarize decode efficiency deltas for each mode.
- [ ] Summarize retrieval accuracy deltas by context length and task tier.
- [ ] Call out any remaining caveats:
  - runtime-compiled CUDA path sensitivity
  - GPU memory variance by selected GPU
  - hybrid-model bounded benefit
