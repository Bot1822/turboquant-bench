# PR #38479 Qwen3.5 Benchmark Worklog

## 2026-03-31

- Decided benchmark focus:
  - model: `/share/models/official/Qwen3.5-35B-A3B`
  - main TurboQuant mode: `tq3`
  - comparison groups: baseline `bf16 kv`, `tq3`, `tq4`
- Decided quality focus:
  - long-document detail retrieval is the primary intelligence metric
  - exact-match / rules-based scoring is the primary evaluation method
  - open-ended scoring may be added only as a secondary qualitative signal
- Verified current PR state:
  - PR `#38479` builds from source in an official-image-based container
  - local Qwen3.5 model can initialize with `kv_cache_dtype=tq3`
  - real text generation completed on A100
  - TQ CUDA store and decode kernels can compile at runtime after local fixes
- Began benchmark design and plan documentation in this directory.
- Created benchmark documentation skeleton:
  - `README.md`
  - `02-design.md`
  - `03-plan.md`
  - `04-experiments.md`
- Implemented first-pass benchmark harness files:
  - `benchmark_helpers.py`
  - `container_case_runner.py`
  - `pr38479_runner.py`
  - `decode_benchmark.py`
  - `retrieval_dataset.py`
  - `scoring.py`
  - `retrieval_benchmark.py`
  - `run_benchmark.py`
- Added a local retrieval dataset generator and deterministic exact-match scorer.
- Started the first decode sweep from the host with:
  - `python3 doc/turboquant-pr38479-qwen35-benchmark/run_benchmark.py --area decode --gpu 1`
- Confirmed the sweep progressed past the baseline case and reached at least the `tq4` case inside:
  - `zgp-vllm-pr38479-share`
- Found and fixed multiple harness issues during early decode runs:
  - result JSON names originally did not include context length
  - experiment log rows originally could not distinguish success from per-case error payloads
  - long-lived shared containers polluted later runs with stale GPU memory
  - fixed runner to launch one fresh `zgp-` container per benchmark case
  - added a clean-GPU preflight gate to skip dirty GPUs
  - updated decode prompt generation to leave token headroom and to scale with target context length
- Current decode sweep is now running in isolated per-case containers with automatic clean GPU selection.

## 2026-04-03

- Reviewed the first isolated decode sweep results and confirmed that the raw
  `generation_seconds` numbers still mixed in first-use TurboQuant runtime
  compilation effects, making them unsuitable as the only decode-efficiency
  signal.
- Added a pure helper module for benchmark-side aggregation and scoring:
  - `benchmark_logic.py`
- Added and ran unit coverage for the helper logic with:
  - `python3 -m unittest test_benchmark_logic`
- Added static syntax verification for the benchmark harness with:
  - `python3 -m py_compile benchmark_logic.py container_case_runner.py container_retrieval_runner.py decode_benchmark.py pr38479_runner.py retrieval_benchmark.py run_benchmark.py`
- Refined decode methodology in the harness:
  - full-length warmup is now separate from measured trials
  - measured decode runs now record TTFT, prefill time, decode time, wall
    throughput, and decode-only throughput
  - steady-state decode runs now support repeated trials in one engine instance
- Refined retrieval methodology in the harness:
  - retrieval examples are now grouped by target context length
  - one container / one engine instance now serves all examples for the same
    `mode x context` batch
  - retrieval runs now emit per-example exact-match scores and per-batch
    aggregates instead of paying one model load per example
- Started the steady-state decode sweep with:
  - `python3 /ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/run_benchmark.py --area decode_steady --gpu -1`
- Aborted the first steady-state decode attempt after confirming that the
  harness was still constructing `LLM(..., disable_log_stats=True)` and would
  therefore fail to surface per-request TTFT / decode metrics.
- Patched the harness to force `disable_log_stats=False` before restarting the
  steady-state decode sweep.

## 2026-04-07

- Confirmed that the steady-state decode sweep had fully completed and that the
  earlier `0.10 tok/s` TurboQuant readings were a cold-start artifact rather
  than a real warmed decode regression.
- Confirmed from the completed steady-state sweep that:
  - `tq3` delivers about `4.00x` effective KV capacity versus baseline
  - `tq4` delivers about `2.00x` effective KV capacity versus baseline
  - warmed decode throughput loss is only a few percent, not catastrophic
- During retrieval reruns, found that the synthetic dataset's requested
  `target_tokens` were only approximate and undercounted true tokenizer length:
  - nominal `8192` prompts were actually about `11,0xx` tokens
  - nominal `32768` prompts were actually about `43,7xx` tokens
- Added effective retrieval context budgeting in the container runner so that
  each batch now computes:
  - `max_prompt_tokens`
  - `mean_prompt_tokens`
  - `effective_max_model_len`
- Found that Qwen emits an empty leading `<think>...</think>` wrapper even when
  the answer span is correct.
- Added a failing test for that scorer mismatch, then updated
  `normalize_answer()` to strip leading `think` wrappers before exact-match
  comparison.
- Preserved the invalid `gpu1` retrieval artifacts as superseded evidence and
  restarted the retrieval benchmark on a fresh GPU (`gpu3`).
- Final corrected retrieval outcomes on `gpu3`:
  - nominal `ctx8192` / actual `~11.0k` prompt tokens:
    - baseline: `3/3`
    - `tq3`: `3/3`
    - `tq4`: `0/3`, because the model spent the answer budget inside verbose
      reasoning text and never emitted the exact answer span
  - nominal `ctx32768` / actual `~43.8k` prompt tokens:
    - baseline: `3/3`
    - `tq3`: failed with TurboQuant prefill CUDA OOM
    - `tq4`: failed with TurboQuant prefill CUDA OOM
- Started a follow-up decode-heavy investigation to answer why TurboQuant is
  not faster on A100 despite the expected KV-bandwidth advantage.
- Investigation hypothesis:
  - current benchmark cases are too prefill-heavy and too short on decode to
    expose a decode-side bandwidth win
  - the currently implemented PR path may not be using the fastest intended TQ
    decode strategy for this workload
  - single-request `B=1` execution may be under-occupying the GPU and hiding
    any bandwidth-side advantage
- Ran a decode-heavy follow-up on `gpu1` with roughly `7.1k` prompt tokens,
  `1024` output tokens, and `max_model_len=16384`.
- Decode-heavy result:
  - baseline: `13.60 decode tok/s`
  - `tq3`: `13.39 decode tok/s`
  - `tq4`: `12.94 decode tok/s`
- This follow-up rules out the simplest explanation that the original benchmark
  was merely too short on output length. Even when decode dominates request
  wall time, `tq3` still does not beat baseline on single-request A100 decode.
- Added a dedicated `batch_decode_probe.py` file after confirming that
  container-side `spawn` multiprocessing cannot safely execute a heredoc
  `python - <<PY` benchmark body.
- Ran a `B=4` batched decode probe on `gpu1`:
  - baseline succeeded with aggregate throughput `49.41 tok/s`
  - `tq3` failed with CUDA OOM in mixed-batch attention before a comparative
    throughput measurement completed
- This shows that pushing the workload toward a more memory-bound serving shape
  does not currently reveal a clean TQ decode win on A100; instead it exposes a
  new stability limit.
- Re-ran the key OOM cases with lower `gpu_memory_utilization`.
- `tq3` nominal `ctx32768` retrieval:
  - at `0.88`, OOM improved from a multi-GB gap to roughly a `1 GiB` request
    with about `0.88 GiB` free, but still failed
  - at `0.86`, the run succeeded
  - tradeoff after recovery: `2/3` exact-match, `mean_ttft=6.14 s`, and KV
    blocks dropped to `31`
- `tq3` `B=4` batched decode probe:
  - at `0.94`, OOMed in mixed-batch attention
  - at `0.88`, completed successfully but still trailed baseline
  - baseline aggregate throughput: `49.41 tok/s`
- `tq3 @ 0.88` aggregate throughput: `43.06 tok/s`
  - baseline mean per-request TTFT: `1.02 s`
  - `tq3 @ 0.88` mean per-request TTFT: `2.75 s`

## 2026-04-08

- Performed a community-status survey for TurboQuant support in the vLLM
  ecosystem after the follow-up question on whether a more robust vLLM-side
  implementation already exists.
- Survey focus:
  - current upstream status of vLLM native TurboQuant support
  - abandoned / superseded upstream TurboQuant PRs
  - current state of the most active out-of-tree community implementation
- Recorded the detailed survey and recommendation in:
  - `06-community-status-2026-04-08.md`
- High-level conclusion from the survey:
  - no newer fully merged official TurboQuant implementation was found in
    upstream vLLM
  - PR `#38479` is still the main native-kernel upstream path, but it remains
    open and under architectural review
  - the most mature community option at this time is the out-of-tree
    `turboquant-vllm` plugin, which appears operationally more robust than the
    upstream PR but is still a young side project rather than official vLLM
    support
- Performed a deeper repository-level survey of `turboquant-vllm` to clarify:
  - its release cadence and packaging state
  - whether it is a true native vLLM replacement or a plugin-side serving path
  - what parts of the TurboQuant stack are actually implemented today
  - whether it is validated for our A100 and Qwen3.5-oriented workflow
- Recorded the detailed assessment in:
  - `07-turboquant-vllm-status-2026-04-08.md`
- Looked up the currently declared vLLM compatibility range for
  `turboquant-vllm` from PyPI metadata and the repository `pyproject.toml`.
- Recorded the result in:
  - `07-turboquant-vllm-status-2026-04-08.md`
- Verified the authorship relationship between upstream PR `#38479` and the
  `turboquant-vllm` repository, then compared recent activity to judge where
  development focus currently sits.
- Recorded the result in:
  - `06-community-status-2026-04-08.md`
- Performed a code-level comparison between the local PR `#38479` worktree and
  the current `turboquant-vllm` repository snapshot, focusing on:
  - implementation completeness
  - robustness / fallback design
  - ease of secondary development
  - learning value for future contributors
- Recorded the Chinese assessment in:
  - `08-code-evaluation-pr38479-vs-plugin-2026-04-08.md`
- Rewrote the same assessment into a normal prose-style research report after
  the first draft was too bullet-heavy for the intended use.
- Started a separate `vLLM 0.19` comparison track for `turboquant-vllm`.
- Scope of the new comparison track:
  - use an official `vllm/vllm-openai:v0.19.0-*` image rather than reusing the
    PR container
  - install `turboquant-vllm` inside an isolated `zgp-` prefixed container
  - run same-model smoke and minimal decode comparisons against the existing PR
    benchmark results
- Added a dedicated document for this track:
  - `09-turboquant-vllm-v019-comparison-2026-04-08.md`
- Verified that the default official image `vllm/vllm-openai:v0.19.0-x86_64`
  is not usable on this host:
  - inside the container, `torch 2.10.0+cu129` reported
    `cudaGetDeviceCount` error `803`
  - the issue reproduced even before loading any model
- Verified that the official image
  `vllm/vllm-openai:v0.19.0-x86_64-cu130` is compatible with the host at the
  container-runtime level:
  - inside the container, `torch 2.10.0+cu130` reports
    `torch.cuda.is_available() == True`
- Installed `turboquant-vllm==1.5.0` and its missing runtime dependencies
  (`scipy`, `molmo-utils`, etc.) inside the official `cu130` container.
- Ran official-`0.19` Qwen3.5 smoke tests on `gpu3`:
  - baseline default backend with `float16` loads but generation fails with
    `query and key must have the same dtype`
  - baseline `TRITON_ATTN` with `float16` loads but generation fails with a
    Triton compilation dtype mismatch (`fp16` vs `uint8`)
  - `turboquant-vllm` `CUSTOM` backend with `float16` successfully loads and
    generates on the same model
- Ran a minimal `ctx4096` plugin decode case on `gpu3` and recorded the first
  approximate plugin-vs-PR comparison metrics.
- Root-cause follow-up for official baseline:
  - the failure is not that official `0.19+cu130` cannot run Qwen3.5 at all
  - baseline becomes operational when `kv_cache_dtype=fp8` is set explicitly
  - with that override, both smoke and `ctx4096` decode cases succeed on
    official `0.19+cu130`
- Updated the plugin-vs-baseline comparison after the fixed official baseline
  results were available.
- Moved `turboquant-vllm` out of `.tmp` into the main workspace at:
  - `/ceph/User/E01442/turboquant/turboquant-vllm`
- Repointed editable installation in the official `0.19+cu130` container to
  the new path and re-ran the plugin `ctx4096` case on an otherwise empty GPU.
- Follow-up finding after the path move:
  - moving the repository path does not remove the plugin TTFT penalty
  - repeated runs on `gpu0` still show materially worse TTFT than the fixed
    official baseline, with noticeable run-to-run variance
- Continued investigation showed that this variance is dominated by first-run
  state rather than ongoing GPU contention:
  - a second run in the same fresh official container on the same `gpu0`
    improved TTFT from `0.89 s` to `0.36 s`
  - `num_gpu_blocks` also moved from `226` to `187`
- Explicitly checked for leftover processes from the `zgp-vllm019-*`
  experiment containers:
  - both live `zgp-vllm019-cu130-*` containers only had `sleep infinity`
    running
  - no residual `EngineCore`, `vllm serve`, or benchmark Python process was
    found inside those containers
  - current heavy GPU usage is attributable to external host-side jobs
    (`ray::WorkerDict`, another `VLLM::Worker_TP*`, and unrelated Python
    processes), not to our stale experiment workers
- Organized the top-level `/ceph/User/E01442/turboquant` repository:
  - added a root `.gitignore`
  - added a root `README.md`
  - recorded `vllm` and `turboquant-vllm` as git submodules in `.gitmodules`
  - left `turboquant-test/` out of the root repository on purpose as a
    separate local workspace
