# PR #38479 Qwen3.5-35B-A3B Benchmark Summary

## Status

- Decode cold-isolated results: recorded
- Decode steady-state results: recorded
- Retrieval quality results: recorded
- Final interpretation: recorded

## Memory

Current finding: TurboQuant changes effective KV capacity far more than it
changes observed `nvidia-smi` resident memory in this single-GPU A100 setup.

Steady-state decode runs show:

| Mode | Block Size | GPU Blocks | Effective KV Capacity | Relative to Baseline |
| --- | --- | --- | --- | --- |
| baseline | 1056 | 345 | 364,320 tokens | 1.00x |
| tq3 | 4192 | 348 | 1,458,816 tokens | 4.00x |
| tq4 | 2096 | 348 | 729,408 tokens | 2.00x |

At `ctx8192`, `nvidia-smi` on the active GPU stayed very close across modes:

- baseline: `75,415 MB` after load, `76,607 MB` after generate
- tq3: `75,447 MB` after load, `76,637 MB` after generate
- tq4: `75,447 MB` after load, `76,637 MB` after generate

Interpretation so far:

- TurboQuant is not reducing model-resident memory in a way that is visible in
  `nvidia-smi` here.
- TurboQuant is increasing usable KV capacity substantially via larger cache
  pages, which is the meaningful memory-side benefit for this PR.
- Front-end `torch.cuda` allocator counters remained `0` in this harness,
  because the actual allocations live in vLLM worker processes rather than the
  benchmark front-end process. These counters are not useful evidence here.

## Decode Efficiency

Two decode regimes matter and they tell different stories.

### Cold isolated runs

The early isolated decode runs from 2026-04-03 showed `tq3` / `tq4` near
`0.10 tok/s` while baseline was about `3.5 tok/s`. Those numbers were dominated
by first-use runtime CUDA compilation inside fresh containers and should not be
used as the main decode comparison.

### Warmed steady-state runs

After full-length warmup and three measured trials per case, the decode gap is
small rather than catastrophic.

| Context | Mode | Load s | Wall tok/s | TTFT s | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 1024 | baseline | 202.37 | 13.25 | 0.19 | 13.97 |
| 1024 | tq3 | 532.90 | 12.82 | 0.20 | 13.52 |
| 1024 | tq4 | 206.40 | 12.54 | 0.21 | 13.23 |
| 4096 | baseline | 206.16 | 13.26 | 0.25 | 14.32 |
| 4096 | tq3 | 206.13 | 12.51 | 0.26 | 13.50 |
| 4096 | tq4 | 207.44 | 12.14 | 0.26 | 13.04 |
| 8192 | baseline | 204.27 | 11.92 | 0.44 | 13.83 |
| 8192 | tq3 | 208.94 | 11.68 | 0.44 | 13.50 |
| 8192 | tq4 | 206.25 | 11.43 | 0.45 | 13.21 |

Relative to baseline steady-state decode:

- `tq3` decode throughput was lower by about `3.2%` at `1024`, `5.8%` at
  `4096`, and `2.4%` at `8192`.
- `tq4` decode throughput was lower by about `5.3%` at `1024`, `8.9%` at
  `4096`, and `4.5%` at `8192`.
- TTFT deltas stayed small, roughly `0.4%` to `6.9%` across the measured
  contexts.

Important caveat:

- The first `tq3@1024` steady-state load still paid a one-time runtime compile
  cost, so its load time was `+330.5 s` relative to baseline. Once the cache
  was warm, `tq3` load time at `4096` and `8192` was essentially at parity.

### Decode-heavy follow-up

To test the hypothesis that the original benchmark was still too prefill-heavy,
an additional single-request decode-heavy case was run with about `7.1k`
prompt tokens, `1024` output tokens, and `max_model_len=16384`.

| Case | Mode | Mean Wall s | Mean TTFT s | Mean Decode tok/s |
| --- | --- | --- | --- | --- |
| prompt `~8192`, output `1024` | baseline | 75.59 | 0.383 | 13.60 |
| prompt `~8192`, output `1024` | tq3 | 76.76 | 0.385 | 13.39 |
| prompt `~8192`, output `1024` | tq4 | 79.47 | 0.404 | 12.94 |

This follow-up matters because it removes the easiest objection to the earlier
steady-state result: even when decode dominates the wall time, `tq3` still does
not overtake baseline on single-request A100 decode. It remains about `1.5%`
slower in decode throughput, and `tq4` remains about `4.9%` slower.

That strongly suggests the current non-winning result is not just an artifact
of short output length.

### Batched decode probe

A follow-up `B=4` batched decode probe was run to test whether a more
memory-bound multi-request setting would finally expose a TurboQuant advantage.

For `baseline`, with about `7.6k` prompt tokens per request and `512` output
tokens per request:

- aggregate output throughput: `49.41 tok/s`
- mean per-request decode throughput: `12.67 tok/s`
- mean per-request TTFT: `1.02 s`

`tq3` did not reach a comparable throughput measurement in the same setting.
Instead, it failed with a CUDA OOM in mixed-batch attention inside
`turboquant_attn.py` / `scaled_dot_product_attention` before the comparative
decode run completed.

Lowering `gpu_memory_utilization` partially changed that result. With the same
`B=4` case and `gpu_memory_utilization=0.88`, `tq3` no longer OOMed, but it
still did not win:

- baseline aggregate throughput: `49.41 tok/s`
- `tq3 @ 0.88` aggregate throughput: `43.06 tok/s`
- baseline mean per-request TTFT: `1.02 s`
- `tq3 @ 0.88` mean per-request TTFT: `2.75 s`

This is important because it means the current A100 bottleneck is not merely
"we have not yet tested enough concurrent decode." Once concurrency is raised,
the current TurboQuant path first hits a stability boundary, and even after
that boundary is relaxed by reserving less KV cache, it still trails baseline.

## Retrieval Quality

The corrected retrieval benchmark now has two qualitatively different regimes.

For the nominal `ctx8192` batch, the real prompt lengths were about `11.0k`
tokens, and the harness therefore raised `effective_max_model_len` to `11118`.

| Requested Context | Mode | Exact Match | Max Prompt Tokens | Effective Max Len | Mean Wall s | Mean TTFT s |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | baseline | 3/3 | 11,030 | 11,118 | 2.08 | 0.90 |
| 8192 | tq3 | 3/3 | 11,030 | 11,118 | 2.26 | 1.07 |
| 8192 | tq4 | 0/3 | 11,030 | 11,118 | 2.80 | 1.06 |

Interpretation so far:

- `tq3` shows no retrieval accuracy drop on the first corrected long-document
  batch.
- `tq4` fails all three strict exact-match cases on this batch, but the raw
  outputs show a specific failure mode: it spends the entire `24`-token answer
  budget inside a verbose `<think>` preamble and never reaches the final exact
  answer span.
- This means the current `tq4` failure is not merely a formatting difference;
  it is a practical answer-budget regression under the same constrained prompt.

For the nominal `ctx32768` batch, the real prompt lengths were about `43.8k`
tokens, and the harness therefore raised `effective_max_model_len` to `43878`.

| Requested Context | Mode | Result | Max Prompt Tokens | Effective Max Len | Mean Wall s | Mean TTFT s |
| --- | --- | --- | --- | --- | --- | --- |
| 32768 | baseline | 3/3 exact match | 43,790 | 43,878 | 4.50 | 3.34 |
| 32768 | tq3 | failure: `EngineDeadError` | 43,790 | 43,878 | na | na |
| 32768 | tq4 | failure: `EngineDeadError` | 43,790 | 43,878 | na | na |

The underlying failure for both `tq3` and `tq4` at the longer context is a
prefill-time CUDA OOM inside `turboquant_attn.py` during
`scaled_dot_product_attention`, while baseline finishes successfully on the
same prompt length.

Lowering `gpu_memory_utilization` also changes this story for `tq3`. With
`gpu_memory_utilization=0.86`, the same `ctx32768` corrected retrieval case no
longer OOMed, but the tradeoff was obvious:

- KV blocks dropped to `31`
- exact match became `2/3` instead of baseline `3/3`
- mean TTFT rose to `6.14 s`
- mean wall time rose to `7.37 s`

Interpretation:

- `baseline` is stable on both corrected retrieval lengths in this suite.
- `tq3` preserves exact-match retrieval quality on the first corrected long
  batch (`~11k` prompt tokens), but it is not robust at the longer corrected
  batch (`~43.8k` prompt tokens) on A100 at the default `gpu_memory_utilization`
  because the TurboQuant prefill path OOMs before generation.
- lowering `gpu_memory_utilization` can recover `tq3` stability at the longer
  corrected batch, but it does so by giving back KV reservation headroom, and
  the recovered run is slower and only `2/3` exact-match.
- `tq4` is weaker than `tq3` even before the OOM regime: it already fails the
  strict exact-answer task at `~11k` prompt tokens by burning the answer budget
  inside verbose reasoning text, and it also OOMs at `~43.8k` prompt tokens.

## Final Interpretation

For the user's stated goals on A100, the current PR-`#38479` `tq3` path is
useful but not fully stable.

- Good news:
  - warmed decode cost is small, roughly low-single-digit percent versus
    baseline
  - effective KV capacity gain is large, about `4x`
  - detail retrieval accuracy is intact on the first corrected long-document
    batch (`~11k` prompt tokens)
- Bad news:
  - there is still a one-time runtime compile penalty on the first `tq3`
    engine startup
  - even with a decode-heavy `1024`-token generation follow-up, `tq3` still
    does not beat baseline on single-request A100 decode
  - when the workload is pushed toward a more memory-bound multi-request regime
    (`B=4`), baseline can run but the current `tq3` path OOMs before a fair
    throughput comparison is even possible; lowering `gpu_memory_utilization`
    enough to make it fit still leaves `tq3` slower than baseline
  - the longer corrected retrieval batch (`~43.8k` prompt tokens) crashes with
    CUDA OOM in TurboQuant prefill while baseline succeeds; lowering
    `gpu_memory_utilization` to make it fit trades that OOM for slower latency
    and a weaker `2/3` retrieval result
  - `tq4` currently looks less practical than `tq3` because it both regresses
    strict answer emission at moderate long context and also OOMs at the longer
    batch

Practical conclusion:

- If the next step is to test and reproduce TurboQuant on A100, `tq3` is the
  right focus.
- If the next step is to claim a broadly stable long-context implementation,
  this PR is not there yet because the long-prefill TurboQuant path still fails
  on the larger retrieval case where baseline remains healthy.
- If the next step is to prove a decode-speed win rather than a capacity win,
  the next benchmark should move away from single-request `B=1` and test a more
  memory-bound serving regime, because the current implementation does not win
  even after making decode much longer.

## Official 0.19 Plugin Comparison

A separate comparison track was run for `turboquant-vllm` on official vLLM
`0.19`.

The first important result is environmental: the default official image
`vllm/vllm-openai:v0.19.0-x86_64` is not usable on this host because its
`torch 2.10.0+cu129` stack fails at `cudaGetDeviceCount()` with error `803`.
The official image `v0.19.0-x86_64-cu130` does work on this host and was used
for the plugin tests.

On `/share/models/official/Qwen3.5-35B-A3B`, the official `0.19+cu130`
baseline path initially looked broken:

- default backend with `float16` loads the model, but generation fails with
  `query and key must have the same dtype`
- explicit `TRITON_ATTN` also loads, but generation fails in the Triton
  unified-attention compile path with an `fp16` vs `uint8` operand mismatch

Root-cause follow-up showed that the issue is not "official baseline cannot run
this model at all." The practical fix is to force `kv_cache_dtype=fp8`. With
that override, the official baseline path succeeds on the same model.

Three official/plugin measurements matter:

- fixed official baseline smoke case (`float16`, `kv_cache_dtype=fp8`):
  - load `71.82 s`
  - TTFT `3.11 s`
  - decode `13.36 tok/s`

- short smoke case:
  - load `83.15 s`
  - TTFT `4.40 s`
  - decode `12.70 tok/s`
- fixed official baseline `ctx4096` case (`3984` prompt tokens, `32` output):
  - load `67.65 s`
  - TTFT `0.28 s`
  - decode `13.36 tok/s`
  - output throughput `12.32 tok/s`
- approximate `ctx4096` decode case (`3984` prompt tokens, `32` output tokens):
  - load `87.41 s`
  - TTFT `0.75 s`
  - decode `12.78 tok/s`
  - output throughput `10.08 tok/s`

Approximate interpretation versus the fixed official baseline and the existing
PR `#38479` `tq4@4096` steady-state result:

- official baseline `0.19+cu130` with `kv_cache_dtype=fp8`:
  - TTFT `0.28 s`
  - decode `13.36 tok/s`
  - wall throughput `12.32 tok/s`
- `turboquant-vllm` reruns after moving the repository out of `.tmp`:
  - rerun on `gpu0`: TTFT `0.48 s`, decode `12.53 tok/s`, wall `10.83 tok/s`
  - rerun in a fresh official container on `gpu0`: TTFT `0.89 s`, decode
    `12.16 tok/s`, wall `9.31 tok/s`
  - second rerun in that same fresh official container on `gpu0`: TTFT
    `0.36 s`, decode `12.78 tok/s`, wall `11.49 tok/s`
  - remote built-in `bench serve` on `10.90.24.4`:
    - baseline mean TTFT `366.6 ms`, mean TPOT `80.43 ms`, mean E2EL `2860.0 ms`
    - plugin mean TTFT `427.2 ms`, mean TPOT `89.87 ms`, mean E2EL `3213.1 ms`

- PR `#38479` `tq4@4096`:
  - TTFT `0.26 s`
  - decode `13.04 tok/s`
  - wall throughput `12.14 tok/s`
- `turboquant-vllm` official `0.19+cu130` plugin `ctx4096`:
  - TTFT `0.75 s`
  - decode `12.78 tok/s`
  - wall throughput `10.08 tok/s`

This is only an approximate comparison because the vLLM versions, backend
integration styles, and dtype choices differ. Even so, two conclusions are
already robust:

- official `0.19+cu130` baseline is usable on this model only after forcing
  `kv_cache_dtype=fp8`; the `auto` path is not stable here
- once that fixed baseline is available, `turboquant-vllm` does not show a
  performance win on this A100 setup: compared with the fixed baseline it is
  slightly slower on decode (about `12.16` to `12.78` vs `13.36 tok/s`) and
  much slower on TTFT (about `0.48` to `0.89 s` vs `0.28 s`)
- moving the plugin repository out of `.tmp` does not remove the TTFT penalty;
  the reruns still show elevated TTFT and add evidence that the plugin TTFT is
  also somewhat variable across runs / container states
- the strongest evidence for that variance source is the plugin's unstable KV
  reservation during initialization plus first-run state: under nominally the
  same `ctx4096` configuration on `gpu0`, `num_gpu_blocks` moved across `70`,
  `226`, and then `187`, while TTFT moved across `0.48 s`, `0.89 s`, and then
  `0.36 s`
- that pattern suggests the largest variance source is initialization-time
  memory profiling / KV budget selection plus first-run warmup state in the
  plugin stack, not sustained third-party GPU contention or a large change in
  steady-state decode speed itself
- the remote repeated online benchmark removes most of the single-run
  ambiguity: under `20` serialized requests at `3984/32`, the plugin still
  trails the fixed baseline on mean TTFT, mean TPOT, and mean end-to-end
  latency
- relative to PR `#38479` `tq4@4096`, the plugin remains slightly slower on
  decode and materially slower on TTFT

## Ability Evaluation

Capability follow-up on `2026-04-10` used a different path from the earlier
`mmlu_pro` chat-completions attempt.

The important distinction is that, in the current `lm_eval` package,
`mmlu_pro` resolves to the Llama-style instruct task group under
`tasks/llama3/instruct/mmlu_pro/`, which is `generate_until` and therefore
turns even a small `--limit 0.02` probe into a long-running reasoning-style
generation benchmark. For a tighter regression smoke, the evaluation was moved
to `leaderboard_mmlu_pro` over `local-completions`, using the remote
OpenAI-compatible `/v1/completions` endpoint with prompt logprobs enabled.

That path produced two useful subsample comparisons on the same remote
baseline/plugin pair:

- `limit=20`:
  - baseline: `acc=0.70 ± 0.1051`
  - plugin: `acc=0.70 ± 0.1051`
  - sample-level comparison:
    - same correctness on `20/20`
    - same predicted option on `18/20`
- `limit=100`:
  - baseline: `acc=0.48 ± 0.0502`
  - plugin: `acc=0.48 ± 0.0502`
  - sample-level comparison:
    - same correctness on `96/100`
    - same predicted option on `87/100`
    - baseline-only correct on `2` items
    - plugin-only correct on `2` items

The current evidence therefore supports a narrow claim: on these
`leaderboard_mmlu_pro` subsamples, no aggregate capability drop is visible for
`turboquant-vllm` relative to the fixed official baseline. At the same time,
the sample-level prediction path is not identical, so the plugin is not
behaviorally equivalent item by item. The measured slices are still too small
to rule out a modest true delta, but they do rule out an obvious or
catastrophic regression on this benchmark path.

## Systematic Three-Benchmark Pass

A more systematic pass was then run on `2026-04-10` with a unified remote
service pair on `10.90.24.4`, `max_model_len=16384`, and `num_concurrent=8`.
The three benchmark families were:

- `leaderboard_mmlu_pro`
- `leaderboard_ifeval`
- `leaderboard_math_hard`

Because the per-sample cost differs sharply across these tasks, the run used a
cost-balanced budget rather than a shared `limit`:

- `leaderboard_mmlu_pro`: `limit=200`
- `leaderboard_ifeval`: `limit=50`
- `leaderboard_math_hard`: `limit=10` per subtask (`70` math questions total)

The main results were:

- `leaderboard_mmlu_pro`
  - baseline: `0.48 ± 0.0354`
  - plugin: `0.47 ± 0.0354`
  - throughput:
    - baseline about `1.146 samples/s`, `11.34 requests/s`
    - plugin about `1.038 samples/s`, `10.27 requests/s`
  - sample-level comparison:
    - same correctness `192/200`
    - baseline-only correct `5`
    - plugin-only correct `3`
- `leaderboard_ifeval`
  - baseline:
    - prompt strict `0.44`
    - prompt loose `0.48`
    - inst strict `0.5526`
    - inst loose `0.5789`
  - plugin:
    - prompt strict `0.52`
    - prompt loose `0.56`
    - inst strict `0.6184`
    - inst loose `0.6447`
  - throughput:
    - baseline about `0.0866 samples/s`
    - plugin about `0.0916 samples/s`
  - prompt-level strict comparison:
    - same label `46/50`
    - plugin-only wins `4`
    - baseline-only wins `0`
- `leaderboard_math_hard`
  - baseline: `0.5714 ± 0.0579`
  - plugin: `0.5571 ± 0.0545`
  - throughput:
    - baseline about `0.1300 samples/s`
    - plugin about `0.1150 samples/s`
  - sample-level comparison:
    - same exact-match label `53/70`
    - baseline-only correct `9`
    - plugin-only correct `8`
  - subtask trend:
    - plugin higher on `algebra` and `num_theory`
    - baseline higher on `counting_and_probability`, `prealgebra`, and
      especially `precalculus`

The runtime behavior in this systematic pass is also important. Under the
heavier plugin runs, TurboQuant repeatedly emitted paged-decompress fallback
warnings, including:

- `39 unique blocks exceed pre-allocated capacity (32 blocks)` during the
  high-concurrency `leaderboard_mmlu_pro` probe
- `122 unique blocks exceed pre-allocated capacity (32 blocks)` during the
  formal `leaderboard_math_hard` run

That pattern strengthens the earlier performance conclusion: even when the
aggregate capability gap is small or mixed, the plugin's runtime path remains
less robust than the fixed official baseline.

Taking the three tasks together, the current evidence is mixed rather than
uniform:

- common-knowledge / MC path: plugin slightly worse and slower
- instruction-following path: plugin better on this slice and slightly faster
- math path: plugin slightly worse and slower

So the best current interpretation is:

- there is no evidence of a catastrophic capability collapse from the plugin
- there is also no evidence of a consistent quality win
- the most stable negative signal remains runtime robustness and throughput,
  not a single large aggregate accuracy drop

## Caveats

Partially confirmed caveats:

Expected caveats to confirm or reject:

- TurboQuant runtime kernels still depend on JIT / extension cache state
- eager-mode execution may understate best-case throughput for all modes
- A100 measurements here do not imply Hopper behavior
- retrieval dataset targets were originally specified as approximate lengths;
  on 2026-04-07 the harness was updated to record real prompt token counts and
  to increase `max_model_len` to an effective safe budget per batch
