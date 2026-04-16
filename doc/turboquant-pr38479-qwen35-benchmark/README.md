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
- `10-systematic-capability-report-2026-04-10.md`
  Systematic `lm_eval` slice report for baseline vs plugin.
- `11-full-benchmark-record-2026-04-10.md`
  Full benchmark execution record for the plugin track.
- `12-fused-kernel-benchmark-design-2026-04-10.md`
  Fused-kernel benchmark design note.
- `13-fused-kernel-execution-plan-2026-04-10.md`
  Execution plan for the fused-kernel debugging and benchmark track.
- `14-fused-kernel-worklog-2026-04-10.md`
  Chronological work log for fused-kernel debugging.
- `15-fused-kernel-report-2026-04-10.md`
  Summary report for fused-kernel findings.
- `16-biweekly-report-2026-04-13.md`
  Two-week summary covering research, experiments, and implementation.
- `17-qwen3-fused-cudagraph-debug-2026-04-14.md`
  Detailed debugging record for getting the fused path through CUDA graph.
- `18-attention-operator-comparison-2026-04-15.md`
  Chinese research note comparing plugin attention, PR attention, and paper attention.
- `19-pr-latest-vs-plugin-qwen3-2026-04-15.md`
  Latest PR vs plugin comparison log, including A100 source-build and smoke debugging.
- `20-h200-handoff-2026-04-16.md`
  H200 migration handoff document for continuing the PR track on a newer GPU platform.

## Scope

The benchmark suite is designed to answer three questions for TurboQuant:

1. How much GPU memory does it save in practice?
2. What is the decode-time cost or benefit?
3. Does long-context detail retrieval quality degrade relative to baseline?
