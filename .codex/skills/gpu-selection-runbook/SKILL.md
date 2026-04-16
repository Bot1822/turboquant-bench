---
name: gpu-selection-runbook
description: Use when an experiment, benchmark, or debugging task can run on either the local machine or remote hosts and you need to pick the emptiest GPU first, based on both memory usage and current utilization instead of guessing by card index.
---

# GPU Selection Runbook

## Overview

Use this skill before any GPU experiment that is not explicitly pinned to a machine or card. The goal is to choose the least-contended GPU across available hosts, record that decision, and avoid noisy results caused by hidden contention.

## When to Use

- Launching a new benchmark or service on local or remote GPUs
- Picking between this machine and `.4`
- Re-running an experiment and wanting a cleaner device
- Verifying whether a surprising result may be due to GPU contention

Do not use this skill when the user has already explicitly pinned the machine and GPU and there is no ambiguity.

## Core Rule

Never choose a GPU by index alone.

Always inspect:

- `memory.used`
- `utilization.gpu`
- experiment-specific residual containers or processes

## Workflow

### 1. Enumerate candidate machines

In this project, the common candidates are:

- local machine
- remote host `10.90.24.4`

If the user names a machine, treat that as the candidate set unless there is a compelling reason to verify another host for diagnosis.

### 2. Check GPU state on every candidate

Local:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Remote `.4`:

```bash
ssh guipeng@10.90.24.4 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
ssh guipeng@10.90.24.4 'docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
```

### 3. Rank GPUs

Preferred order:

1. lowest `memory.used`
2. then lowest `utilization.gpu`
3. then fewest relevant residual experiment containers

Useful heuristic:

- `memory.used <= 1024 MiB` and `utilization.gpu == 0%` is usually clean enough
- if two GPUs are both clean, choose the one with fewer active experiment remnants

### 4. Record the choice before launch

Before starting containers or servers, write down:

- machine
- GPU index
- competing containers/processes checked
- reason for choosing this card

This belongs in the experiment worklog.

## Quick Reference

### One-line local view

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

### One-line `.4` view

```bash
ssh guipeng@10.90.24.4 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
```

### Residual container check

```bash
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
ssh guipeng@10.90.24.4 'docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
```

## Common Mistakes

- Choosing `gpu0` or the “last four cards” without checking actual usage
- Looking only at memory and ignoring `utilization.gpu`
- Ignoring residual `zgp-*` containers from earlier runs
- Forgetting to record which machine/GPU was chosen
- Treating “small memory use” as safe while utilization is still high

## References

See `references/host-checks.md` for reusable command snippets and decision examples.
