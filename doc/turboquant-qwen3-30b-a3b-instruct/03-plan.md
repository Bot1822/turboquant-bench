# TurboQuant Qwen3-30B-A3B-Instruct Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a documented experiment suite that first proves KV-cache occupancy reduction for TurboQuant on Qwen3-30B-A3B-Instruct, then tests whether that reduction improves concurrent long-history capacity on `2 x A100 80GB`.

**Architecture:** Reuse the existing baseline / TurboQuant harnesses where possible, but add focused experiment utilities for long-history prompt generation, structural KV accounting, and concurrency-capacity sweeps. Report structural and runtime memory evidence separately so the final conclusion remains honest even if allocator counters do not move.

**Tech Stack:** local `uv` Python environment, `vllm==0.17.0`, local `turboquant`, PyTorch CUDA metrics, `nvidia-smi`, pytest, JSON experiment logs, markdown records.

---

## Chunk 1: Documented Calibration Setup

### Task 1: Record target model and hardware assumptions

**Files:**
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/01-worklog.md`
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Write the initial experiment assumptions into the worklog**

Include:
- target model path
- target GPU budget
- target proof hierarchy: KV reduction first, capacity second

- [ ] **Step 2: Record the first calibration commands before running them**

Add empty experiment rows for:
- baseline compatibility
- TurboQuant compatibility
- structural KV accounting
- concurrency ladder

### Task 2: Verify baseline compatibility on the new model

**Files:**
- Reuse: `doc/turboquant-qwen3-30b-a3b-instruct/baseline_generate.py` (create if needed)
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Create or adapt a minimal baseline generation driver**
- [ ] **Step 2: Run it on `2 x A100 80GB` with conservative settings**
- [ ] **Step 3: Record load time, generation time, block size, and memory metrics**

## Chunk 2: Structural KV Accounting

### Task 3: Add pure helper logic for model metadata and storage accounting

**Files:**
- Create: `turboquant/turboquant/experiment_utils.py`
- Test: `turboquant/test_experiment_utils.py`

- [ ] **Step 1: Write failing tests for model metadata parsing and byte accounting**

Test:
- counting hookable/full-attention layers when available
- computing unique storage bytes from synthetic tensor-metadata inputs
- summarizing compression ratio calculations

- [ ] **Step 2: Run the new tests and verify they fail for missing helpers**

Run: `../vllm/.venv/bin/python -m pytest test_experiment_utils.py -q`
Expected: FAIL with import or missing symbol errors

- [ ] **Step 3: Implement the minimal helper functions**

- [ ] **Step 4: Re-run the tests and verify they pass**

Run: `../vllm/.venv/bin/python -m pytest test_experiment_utils.py -q`
Expected: PASS

### Task 4: Build a structural KV inspection driver

**Files:**
- Create: `doc/turboquant-qwen3-30b-a3b-instruct/inspect_kv_occupancy.py`
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Implement a diagnostic script that reports**

At minimum:
- hook count
- total attention layer count or explicit model-wide coverage statement
- baseline hooked-layer KV tensor bytes
- TQ side-cache bytes
- placeholder bytes after free
- allocator metrics before and after free

- [ ] **Step 2: Run the diagnostic script for baseline and TurboQuant**
- [ ] **Step 3: Record structural compression evidence in the experiment log**

## Chunk 3: Long-History Workload Generator

### Task 5: Build deterministic long-dialog prompt generation

**Files:**
- Create: `turboquant/turboquant/dialog_workload.py`
- Test: `turboquant/test_dialog_workload.py`

- [ ] **Step 1: Write failing tests for chat-history prompt generation**

Test:
- prompt builder creates multi-turn chat history
- token length target is approximately met
- batched prompt generation is deterministic

- [ ] **Step 2: Run tests and verify they fail**

Run: `../vllm/.venv/bin/python -m pytest test_dialog_workload.py -q`
Expected: FAIL

- [ ] **Step 3: Implement minimal prompt-generation helpers**

- [ ] **Step 4: Re-run tests and verify they pass**

Run: `../vllm/.venv/bin/python -m pytest test_dialog_workload.py -q`
Expected: PASS

## Chunk 4: Capacity Benchmark Harness

### Task 6: Build a long-history concurrency benchmark

**Files:**
- Create: `doc/turboquant-qwen3-30b-a3b-instruct/concurrency_capacity.py`
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Implement a harness that runs baseline and TurboQuant in separate processes**

Inputs:
- model path
- tensor parallel size
- history length target
- concurrency target
- output length

Outputs:
- success/failure
- total latency
- total generated tokens
- tokens/s
- `nvidia-smi` snapshot
- CUDA allocator metrics
- hook count for TurboQuant

- [ ] **Step 2: Verify the harness on a low-pressure point**
- [ ] **Step 3: Record the successful smoke run**

## Chunk 5: Pressure Sweep

### Task 7: Calibrate the pressure ladder

**Files:**
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Run single-request history-length calibration**

Suggested ladder:
- `16k`, `32k`, `48k`, `64k`, `96k`

- [ ] **Step 2: Pick one or two high-pressure lengths based on the results**
- [ ] **Step 3: Record the chosen lengths and the reason**

### Task 8: Run concurrency sweeps at high pressure

**Files:**
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Run baseline concurrency ladder**

Suggested ladder:
- `1`, `2`, `4`, `8`, `12`, `16`

- [ ] **Step 2: Run TurboQuant on the same ladder**
- [ ] **Step 3: Continue until failure, timeout, or severe throughput collapse**
- [ ] **Step 4: Record exact failure modes**

### Task 9: Build a steady-state wave-2 admission test

**Files:**
- Create: `doc/turboquant-qwen3-30b-a3b-instruct/steady_state_admission.py`
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Implement a same-engine two-wave harness**

Sequence:
- admit wave 1 long-history requests
- keep wave 1 resident
- call `free_kv_cache()` in TurboQuant mode
- attempt wave 2 admission

- [ ] **Step 2: Run baseline and TurboQuant with identical wave shapes**
- [ ] **Step 3: Record whether wave 2 admission changes under TQ**

### Task 10: Integrate TurboQuant with a single online serving instance

**Files:**
- Create: `doc/turboquant-qwen3-30b-a3b-instruct/tq_api_server.py`
- Create: `doc/turboquant-qwen3-30b-a3b-instruct/generate_multi_turn_long.json`
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Build a server launcher that installs TurboQuant hooks into one live vLLM server**
- [ ] **Step 2: Create a long-history multi-turn workload configuration**
- [ ] **Step 3: Run baseline vs TurboQuant using `vllm/benchmarks/multi_turn` against one server instance**
- [ ] **Step 4: Record per-request latency and throughput statistics**

## Chunk 6: Quality And Throughput Trade-off

### Task 11: Run fixed sanity prompts for baseline vs TurboQuant

**Files:**
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Use a fixed small quality prompt set**
- [ ] **Step 2: Compare outputs for obvious drift**
- [ ] **Step 3: Record the outcome honestly**

### Task 12: Summarize the benchmark trade-off

**Files:**
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/01-worklog.md`

- [ ] **Step 1: Summarize**
- structural compression evidence
- runtime memory evidence
- concurrency gain
- throughput cost

- [ ] **Step 2: Explicitly call out unresolved gaps**

## Chunk 7: Verification

### Task 13: Re-run final verification commands

**Files:**
- Modify: `doc/turboquant-qwen3-30b-a3b-instruct/04-experiments.md`

- [ ] **Step 1: Re-run the final baseline command**
- [ ] **Step 2: Re-run the final TurboQuant proof command**
- [ ] **Step 3: Re-run the final concurrency benchmark command**
- [ ] **Step 4: Re-run the final steady-state admission command**
- [ ] **Step 5: Re-run the final single-instance multi-turn serving benchmark**
- [ ] **Step 6: Confirm the logged numbers match the actual output**
