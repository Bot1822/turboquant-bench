# PR #38479 TurboQuant Benchmark Pack

This directory contains the benchmark design, execution plan, work log, and
experiment record for evaluating vLLM PR `#38479` on
`/share/models/official/Qwen3.5-35B-A3B`.

## Files

- `01-worklog.md`
  Ongoing engineering log for benchmark design, implementation, debugging, and
  benchmark execution.
- `02-design.md`
  Benchmark design document.
- `03-plan.md`
  Concrete execution plan for building and running the benchmark harness.
- `04-experiments.md`
  Structured experiment log and result table.
- `05-summary.md`
  Rolling benchmark summary and interpretation document.
- `06-community-status-2026-04-08.md`
  Community-status survey for TurboQuant support in the vLLM ecosystem as of
  `2026-04-08`.
- `07-turboquant-vllm-status-2026-04-08.md`
  Detailed status survey for the community `turboquant-vllm` plugin as of
  `2026-04-08`.
- `08-code-evaluation-pr38479-vs-plugin-2026-04-08.md`
  Chinese code-level evaluation comparing PR `#38479` with
  `turboquant-vllm`.
- `09-turboquant-vllm-v019-comparison-2026-04-08.md`
  Experiment design and results for testing `turboquant-vllm` in a separate
  vLLM `0.19` container and comparing it with PR `#38479`.

## Scope

The benchmark suite is designed to answer three questions for TurboQuant:

1. How much GPU memory does it save in practice?
2. What is the decode-time cost or benefit?
3. Does long-context detail retrieval quality degrade relative to baseline?
