# lm_eval Results

## Scope

This file records benchmark-style accuracy evaluation using `lm_eval`.

## MMLU-Pro Smoke Run

### Environment

- Model: `/share/models/official/Qwen3-30B-A3B-Instruct-2507`
- GPUs: `CUDA_VISIBLE_DEVICES=2,3`
- TP: `2`
- `gpu_memory_utilization=0.85`
- `max_model_len=16384`
- `batch_size=4`
- `num_fewshot=0`
- `limit=5`
- Mirror: `HF_ENDPOINT=https://hf-mirror.com`

### Baseline Command

```bash
CUDA_VISIBLE_DEVICES=2,3 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
HF_ENDPOINT=https://hf-mirror.com \
MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 \
MODE=baseline \
TASKS=mmlu_pro \
TP=2 \
GPU_MEM=0.85 \
MAX_MODEL_LEN=16384 \
LM_EVAL_BATCH_SIZE=4 \
LIMIT=5 \
NUM_FEWSHOT=0 \
vllm/.venv/bin/python \
doc/turboquant-qwen3-30b-a3b-instruct/task-lm-eval/run_lm_eval.py
```

### Baseline Result

- Metric: `exact_match,custom-extract`
- Score: `71.43`
- Elapsed: `190.615s`

### TurboQuant no_alloc Command

```bash
CUDA_VISIBLE_DEVICES=2,3 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
HF_ENDPOINT=https://hf-mirror.com \
MODEL=/share/models/official/Qwen3-30B-A3B-Instruct-2507 \
MODE=tq_no_alloc \
TASKS=mmlu_pro \
TP=2 \
GPU_MEM=0.85 \
MAX_MODEL_LEN=16384 \
LM_EVAL_BATCH_SIZE=4 \
LIMIT=5 \
NUM_FEWSHOT=0 \
vllm/.venv/bin/python \
doc/turboquant-qwen3-30b-a3b-instruct/task-lm-eval/run_lm_eval.py
```

### TurboQuant no_alloc Result

- Metric: `exact_match,custom-extract`
- Score: `0.0`
- Elapsed: `338.52s`

## Interpretation

- This is the first benchmark-style evidence that **true compressed-decode
  mode** currently causes severe quality loss on a real evaluation task.
- Important nuance:
  - this result is for `MODE=tq_no_alloc`
  - it is not the same as the default current integration path, which often
    falls back to native flash attention while paged KV is still present
- Therefore the correct reading is:
  - **TurboQuant compression works structurally**
  - but the current compressed-decode integration is not yet accuracy-safe
    enough for benchmark use
