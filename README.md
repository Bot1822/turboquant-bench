# TurboQuant Workspace

This repository is the top-level workspace for local TurboQuant reproduction,
benchmarking, and notes.

## Layout

- `doc/`
  Experiment design, worklogs, benchmark harnesses, results, and summaries.
- `vllm/`
  vLLM source tree, tracked as a submodule.
- `turboquant-vllm/`
  Community `turboquant-vllm` source tree, tracked as a submodule.

## Submodules

The root repository records these submodules:

- `vllm` → `https://github.com/vllm-project/vllm.git`
- `turboquant-vllm` → `https://github.com/Alberto-Codes/turboquant-vllm.git`

To clone and initialize:

```bash
git clone <this-repo>
cd turboquant
git submodule update --init --recursive
```

To refresh submodules after pulling:

```bash
git submodule update --init --recursive
```

## Notes

- Generated cache artifacts under
  `doc/turboquant-pr38479-qwen35-benchmark/cache/` are intentionally ignored.
- `turboquant-test/` is a separate local workspace and is not managed by this
  root repository.
