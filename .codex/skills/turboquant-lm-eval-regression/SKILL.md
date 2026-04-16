---
name: turboquant-lm-eval-regression
description: Use when running or extending TurboQuant capability regressions with lm-eval, especially for the project-standard task set IFEval, leaderboard_mmlu_pro, and leaderboard_math_hard, or when choosing between local-completions, loglikelihood, and generate-until evaluation paths.
---

# TurboQuant LM-Eval Regression

## Overview

Use this skill for the capability-evaluation branch of the TurboQuant project. It encodes the stable choices already validated here: which lm-eval tasks to use, which endpoint style to prefer, how to split remote server and local client, and how to interpret small score gaps versus runtime cost.

## When to Use

- Running a new capability regression after a serving or kernel change
- Comparing baseline and TurboQuant on instruction following, common knowledge, or math
- Deciding whether to use `mmlu_pro` or `leaderboard_mmlu_pro`
- Preparing subsample, systematic, or full evaluation runs
- Selecting the cleanest machine/GPU before launching evaluation services

## Task Set

The project-standard ability matrix is:

- `leaderboard_ifeval`
- `leaderboard_mmlu_pro`
- `leaderboard_math_hard`

Interpretation:

- `leaderboard_ifeval`: instruction-following generation task
- `leaderboard_mmlu_pro`: multiple-choice leaderboard path using prompt scoring
- `leaderboard_math_hard`: generation-heavy math group task

## Workflow

### 1. Check the task shape before running

Do not assume task semantics from the task name.

Verify:

- whether the task is `multiple_choice` or `generate_until`
- whether `limit=N` means total samples or per-subtask samples
- whether the endpoint should be chat or completions

Project-learned rule:

- prefer `leaderboard_mmlu_pro` over `mmlu_pro` for fast regression checks
- `leaderboard_mmlu_pro` is the better path when you want prompt loglikelihood scoring instead of long answer generation

### 2. Prefer `local-completions`

Default:

- server on remote host
- client on local host
- `--model local-completions`

Why:

- keeps tokenization and benchmarking under local control
- avoids long chat-format reasoning output when you only need regression evidence

Before launching the server, use `$gpu-selection-runbook` to choose the
machine and GPU.

### 3. Standard model args

Use the local tokenizer path and explicit concurrency:

```text
model=<label>,tokenizer=/share/models/official/<model>,max_length=<len>,num_concurrent=<n>,max_retries=10,tokenizer_backend=huggingface,tokenized_requests=False
```

### 4. Run in three layers

- smoke: tiny slice to confirm endpoint and task wiring
- systematic: modest, fixed slices across all tasks
- full: complete or near-complete production comparison

### 5. Compare both aggregate and sample-level behavior

Always extract:

- aggregate score
- stderr when available
- wall time
- sample-level agreement / baseline-only wins / plugin-only wins

The project repeatedly found cases where aggregate accuracy stayed flat but sample-level predictions diverged.

## Quick Reference

### Suggested tasks

| Goal | Task |
| --- | --- |
| Instruction following | `leaderboard_ifeval` |
| Common knowledge MC | `leaderboard_mmlu_pro` |
| Math generation | `leaderboard_math_hard` |

### Suggested endpoint

| Task type | Preferred endpoint |
| --- | --- |
| Multiple-choice scoring | `/v1/completions` |
| Generation tasks | `/v1/completions` unless chat template is required |

### Suggested reporting

- score
- stderr
- elapsed seconds
- concurrency
- max length
- sample-level disagreement counts

## Common Mistakes

- Using `mmlu_pro` chat generation when the real goal is fast regression checking
- Forgetting that `leaderboard_math_hard` fans out by subject, so `limit` is per subtask
- Comparing only aggregate score and ignoring changed predictions
- Running full eval without verifying the endpoint on a tiny slice first
- Forgetting to document task semantics in the report

## References

See `references/tasks.md` for task-specific notes and command patterns.
