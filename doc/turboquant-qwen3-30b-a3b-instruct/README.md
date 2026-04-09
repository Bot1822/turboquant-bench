# TurboQuant on Qwen3-30B-A3B-Instruct

This directory records the end-to-end experiment design, execution, and results
for evaluating TurboQuant on:

- `/share/models/official/Qwen3-30B-A3B-Instruct-2507`
- `2 x A100 80GB`

Primary goal:
- prove KV-cache occupancy reduction first

Secondary goal:
- show whether that reduction translates into better multi-user long-dialog
  capacity and throughput under memory pressure

Files:
- `01-worklog.md`: chronological work log
- `02-design.md`: experiment design and rationale
- `03-plan.md`: execution plan
- `04-experiments.md`: concrete runs and metrics
- `05-实验说明.md`: Chinese write-up of the experiment flow and conclusions
- `task-kv-occupancy/`: direct KV occupancy measurement
- `task-capacity/`: capacity and admission experiments
- `task-accuracy/`: accuracy / quality experiments
- `task-lm-eval/`: benchmark evaluation with `lm_eval`
