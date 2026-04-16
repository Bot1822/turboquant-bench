# TurboQuant LM-Eval Task Notes

## `leaderboard_mmlu_pro`

- Use for regression-friendly multiple-choice comparison
- Prefer `local-completions`
- Faster and cleaner than long chat-generation `mmlu_pro`

## `leaderboard_ifeval`

- Generation task
- Sensitive to answer formatting and instruction-following behavior
- Report prompt-level and instruction-level metrics separately

## `leaderboard_math_hard`

- Group task over multiple math subjects
- `limit=N` means `N` per subtask, not `N` total
- Always inspect per-subtask deltas, not only aggregate exact match

## Command Pattern

```bash
lm_eval \
  --model local-completions \
  --model_args model=<label>,tokenizer=/share/models/official/<model>,max_length=<len>,num_concurrent=<n>,max_retries=10,tokenizer_backend=huggingface,tokenized_requests=False \
  --tasks <task> \
  --output_path <dir> \
  --trust_remote_code
```
