---
name: turboquant-vllm-experiment-runbook
description: Use when running, extending, or reproducing TurboQuant/vLLM experiments in this repo, especially when choosing between local and remote machines, selecting idle GPUs first, launching official vLLM containers, benchmarking from a separate client, recording every step in project docs, and cleaning up services afterward.
---

# TurboQuant vLLM Experiment Runbook

## Overview

Use this skill for the project-standard experiment workflow around `vllm`, `turboquant-vllm`, and local benchmark harnesses. It codifies the stable pattern from this repo: official image first, use `$gpu-selection-runbook` to choose the machine and GPU, prefer server/client split when helpful, document everything, and never leave benchmark containers behind.

## When to Use

- Running any new TurboQuant serving experiment on local or remote GPUs
- Reproducing a previous benchmark from the project docs
- Comparing baseline `fp8` / `bf16` against TurboQuant plugin modes
- Switching models but keeping the same containerized serving workflow
- Cleaning up remote experiment servers after a run

Do not use this skill for pure code reading or algorithm-only analysis without any service launch or benchmark execution.

## Core Workflow

### 1. Fix the experiment contract first

Before launching anything, write down:

- model path
- image tag
- GPU allocation
- ports
- baseline modes
- TurboQuant modes
- benchmark shape
- result directory

Use the project docs as the source of truth:

- `doc/turboquant-pr38479-qwen35-benchmark/01-worklog.md`
- `doc/turboquant-pr38479-qwen35-benchmark/04-experiments.md`
- `doc/turboquant-pr38479-qwen35-benchmark/11-full-benchmark-record-2026-04-10.md`
- `doc/turboquant-pr38479-qwen35-benchmark/14-fused-kernel-worklog-2026-04-10.md`

### 2. Prefer official vLLM images first

Default image:

- `vllm/vllm-openai:v0.19.0-x86_64-cu130`

Rules:

- Always prefer the official image first.
- For plugin experiments, mount the local `turboquant-vllm` repo and `pip install` it inside the container.
- If the official image has a default entrypoint, use `--entrypoint bash` and then `-lc "vllm serve ..."` to avoid the `vllm: error: unrecognized arguments: -lc ...` failure.

### 3. Choose the machine and GPU first

Always invoke `$gpu-selection-runbook` before launch if the user did not pin a
specific machine and card.

The current project commonly uses:

- local machine
- remote host `10.90.24.4`

### 4. Use the standard server/client split when it helps

Project-default pattern:

- server on remote host
- benchmark client on local host

Why:

- cleaner GPU isolation
- easier benchmark orchestration
- easier log collection

The common pattern is still:

- server on the chosen machine
- benchmark client on local host

For remote services on `.4`:

- use `ssh guipeng@10.90.24.4`
- use shared model path mounts from `/share/models/official`
- use shared project path mounts from `/ceph/User/E01442/turboquant`

For local services:

- use the same official image and naming rules
- still record GPU and port ownership explicitly

### 5. Naming and lifecycle rules

- Container names must start with `zgp-`
- Use one container per mode
- Record GPU and port ownership explicitly
- Do not use `tmux`
- Use background process/session mechanisms instead

Recommended naming pattern:

- `zgp-vllm019-<track>-<mode>`

Examples:

- `zgp-vllm019-fusedstudy-fp8`
- `zgp-vllm019-qwen3-fused-debug`

### 6. Benchmarking rules

- Mount tokenizer/model path into the client container when using `vllm bench serve`
- For local model paths, always pass `--tokenizer /share/models/official/<model>` explicitly
- Use `/v1/completions` for stable throughput testing unless the task truly needs chat formatting
- Keep one result directory per case and save raw JSONs

### 7. Documentation is mandatory

Every experiment must update at least one of:

- `01-worklog.md` for chronological trace
- a dedicated track-specific report/worklog
- `04-experiments.md` or equivalent summary doc

Record:

- exact image
- ports and GPUs
- key env vars
- startup failures
- benchmark parameters
- cleanup status

### 8. Cleanup is part of completion

After the run:

- remove `zgp-` containers on the remote host
- confirm ports are gone
- confirm no experiment service containers remain

## Quick Reference

### Machine/GPU selection

See `$gpu-selection-runbook` for the actual local/remote GPU choice logic and
check commands.

### Standard remote launch shape

```bash
ssh guipeng@10.90.24.4 'docker run -d --rm \
  --name zgp-vllm019-<track>-<mode> \
  --gpus device=<gpu> \
  --ipc=host \
  -p <host_port>:8000 \
  -v /share/models/official:/share/models/official:ro \
  --entrypoint bash \
  vllm/vllm-openai:v0.19.0-x86_64-cu130 \
  -lc "vllm serve /share/models/official/<model> ... "'
```

### Plugin launch shape

```bash
ssh guipeng@10.90.24.4 'docker run -d --rm \
  --name zgp-vllm019-<track>-plugin \
  --gpus device=<gpu> \
  --ipc=host \
  -p <host_port>:8000 \
  -v /share/models/official:/share/models/official:ro \
  -v /ceph/User/E01442/turboquant/turboquant-vllm:/workspace/turboquant-vllm:ro \
  -e TQ4_K_BITS=4 \
  -e TQ4_V_BITS=4 \
  --entrypoint bash \
  vllm/vllm-openai:v0.19.0-x86_64-cu130 \
  -lc "pip install /workspace/turboquant-vllm && vllm serve ... --attention-backend CUSTOM"'
```

### Client benchmark shape

```bash
docker run --rm --network host \
  -v /share/models/official:/share/models/official:ro \
  -v /ceph/User/E01442/turboquant/doc/.../results:/results \
  --entrypoint bash vllm/vllm-openai:v0.19.0-x86_64-cu130 \
  -lc 'vllm bench serve \
    --backend openai \
    --endpoint /v1/completions \
    --model /share/models/official/<model> \
    --tokenizer /share/models/official/<model> \
    --host 10.90.24.4 --port <port> ...'
```

## Common Mistakes

- Picking a GPU by index before checking whether it is actually idle
- Looking only at `memory.used` and ignoring `utilization.gpu`
- Assuming experiments must always run on `.4`
- Forgetting `--entrypoint bash` on the official image
- Using a local model path in `vllm bench serve` without also passing `--tokenizer`
- Mixing chat endpoints into pure throughput experiments
- Leaving remote containers alive after the benchmark
- Recording conclusions without saving the raw JSON result
- Running plugin and baseline on the same GPU when you actually need isolation

## References

See `references/commands.md` for reusable launch and benchmark command patterns.
