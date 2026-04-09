# PR #38479 vLLM TurboQuant Reproduction Log

Date: 2026-03-31
Owner: Codex with user supervision
Primary goal: reproduce, validate, and experiment on vLLM PR #38479 in an isolated environment

## Requirements

- All development, testing, experiment records, and code changes must be documented in this log.
- Container names created for this effort must use the `zgp-` prefix.

## Workspace Layout

- Outer workspace root: `/ceph/User/E01442/turboquant`
- vLLM repo used for PR reproduction: `/ceph/User/E01442/turboquant/vllm`
- PR worktree path: `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`
- TurboQuant side repo in this workspace: `/ceph/User/E01442/turboquant/turboquant`

## Initial Repository Safety Work

- Confirmed the outer workspace root is only a local collection repo and should not be used as the main PR checkout.
- Confirmed the real upstream-tracking repo is `/ceph/User/E01442/turboquant/vllm`.
- Added `.worktrees/` to `/ceph/User/E01442/turboquant/vllm/.gitignore` to prevent project-local worktrees from polluting git status.
- Committed that safety fix on `vllm/main` with commit:
  - `49673aa7b` `Ignore project-local worktrees`

## PR Checkout Work

- Fetched upstream state from `origin`.
- Fetched PR head:
  - `refs/pull/38479/head -> pr-38479-upstream`
- Created dedicated worktree and branch:
  - branch: `pr-38479-turboquant`
  - path: `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`
- Verified worktree head:
  - `daaf633e4` `Merge branch 'main' into feature/turboquant-kv-cache`

## Local venv Attempt

- Created local `.venv` inside the PR worktree using `uv venv --python 3.12`.
- Started local dependency installation with:
  - `uv pip install -r requirements/test.in`
  - `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
- Stopped relying on this path after deciding to move the real reproduction workflow into containers for stronger environment control and reproducibility.

## Source-Build Container Attempt From Dockerfile

- Attempted to build a dedicated `test` image from the PR worktree using `docker/Dockerfile`.
- First build failed because the local Docker daemon was configured to use a broken registry mirror:
  - `mirror.ccs.tencentyun.com`
- Switched strategy to use a locally available CUDA base image to bypass the registry mirror.

### Custom-base build command attempted

- Target: `test`
- Intended tag: `vllm:pr-38479-test`
- Custom base image override:
  - `BUILD_BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`
  - `FINAL_BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`

### Root cause found for custom Docker build failure

- Build later failed at `docker/Dockerfile:548-559`, the `libnccl-dev` install step.
- Evidence gathered inside local NVIDIA base images showed:
  - the images already contain matching installed NCCL packages such as `libnccl-dev` and `libnccl2` for `+cuda12.9`
  - after `apt-get update`, the CUDA repo exposes newer candidates such as `libnccl-dev 2.29.7-1+cuda13.2`
  - apt chooses the highest candidate version, causing `libnccl-dev` to depend on `libnccl2 (= 2.29.7-1+cuda13.2)` while the resolver tries to keep `libnccl2 +cuda12.9`
- Conclusion:
  - the failure was caused by package-resolution drift in the CUDA apt repo, not by PR #38479 source changes

## Official vLLM Image Investigation

- Checked local official images already present on the host.
- Important finding:
  - local `vllm/vllm-openai:latest` is stale and reports `vllm 0.12.0`
  - local `vllm/vllm-openai:v0.17.0` is much closer to the PR base and is suitable as the dev container base

### Verified `vllm/vllm-openai:v0.17.0` toolchain contents

- Python version: `3.12.13`
- `nvcc` present at `/usr/local/cuda/bin/nvcc`
- `gcc` and `g++` present
- `libpython3.12-dev` present
- `libnccl-dev` present
- Preinstalled vLLM version: `0.17.0`

## Official-image Dev Container

- Started an official-image-based development container with the PR worktree bind-mounted.
- Important runtime note:
  - official `vllm/vllm-openai` images define a `vllm` entrypoint
  - the first attempt failed because `bash -lc ...` was passed to the image entrypoint instead of replacing it
  - corrected by explicitly setting `--entrypoint bash`

### Active development container

- Container name: `zgp-vllm-pr38479-dev`
- Image: `vllm/vllm-openai:v0.17.0`
- Mount:
  - host `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`
  - container `/workspace/vllm`
- Working directory inside container: `/workspace/vllm`

## Current Status

- Worktree is ready.
- Official-image dev container is ready.
- Next task is to install PR #38479 from source inside `zgp-vllm-pr38479-dev`, then run smoke tests and TurboQuant-specific checks.

## Official-container Source Install Attempt 1

### Actions

- Installed `git` inside `zgp-vllm-pr38479-dev`.
- Uninstalled the image-bundled `vllm 0.17.0`.
- Attempted source install from the mounted PR worktree with:
  - `TORCH_CUDA_ARCH_LIST=8.0`
  - `MAX_JOBS=16`
  - `NVCC_THREADS=8`
  - `uv pip install --system -e . --torch-backend=auto --no-build-isolation --no-deps`

### Result

- Install failed during editable metadata preparation, before the real CUDA compile stage.

### Root Cause

- `setuptools-scm` could not detect repository metadata for the mounted worktree.
- Evidence:
  - worktree `.git` file contains:
    - `gitdir: /ceph/User/E01442/turboquant/vllm/.git/worktrees/pr-38479-turboquant`
  - inside the current container, only the worktree itself was bind-mounted at `/workspace/vllm`
  - the absolute host path under `/ceph/User/E01442/turboquant/vllm/.git/worktrees/...` was not mounted into the container
- Conclusion:
  - the failure was caused by incomplete bind-mount coverage for git metadata in a worktree checkout
  - the correct fix is to recreate the container with a bind mount that preserves the host absolute path used by the worktree metadata, rather than patching around version detection

## Official-container Source Install Attempt 2 Preparation

### Actions

- Recreated `zgp-vllm-pr38479-dev` with the outer workspace bind-mounted at the same absolute host path:
  - host and container path: `/ceph/User/E01442/turboquant`
- Set container working directory to:
  - `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`
- Verified inside the container that:
  - `.git` still points to `/ceph/User/E01442/turboquant/vllm/.git/worktrees/pr-38479-turboquant`
  - that absolute path is now accessible
- Reinstalled `git` inside the recreated container.
- Uninstalled the image-bundled `vllm 0.17.0` again.

### New blocker found

- Running `git status` inside the mounted worktree now fails with:
  - `fatal: detected dubious ownership in repository`
- Root cause:
  - the bind-mounted repo files are owned by the host UID, while the container is running as root
  - git therefore rejects the repository until it is marked as a trusted `safe.directory`

### Next fix

- Add the PR worktree path to container git `safe.directory`
- Retry the editable install from source after that

## Official-container Source Install Attempt 2

### Actions

- Added container git trust entries:
  - `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`
  - `/ceph/User/E01442/turboquant/vllm`
- Verified `git status` works inside the container worktree.
- Retried editable install with the same compile settings:
  - `TORCH_CUDA_ARCH_LIST=8.0`
  - `MAX_JOBS=16`
  - `NVCC_THREADS=8`
  - `uv pip install --system -e . --torch-backend=auto --no-build-isolation --no-deps`

### Result

- Install progressed further than before:
  - metadata generation succeeded
  - editable wheel build started
  - `build_ext` started
- The install then failed with:
  - `RuntimeError: Cannot find CMake executable`

### Root Cause

- The official runtime image `vllm/vllm-openai:v0.17.0` is usable as a development base, but it does not include all build-time tools required to rebuild vLLM from source.
- Confirmed missing build dependency:
  - `cmake`
- Conclusion:
  - the worktree metadata issue is fixed
  - the next blocker is ordinary build-tool provisioning inside the official image

### Next fix

- Install `cmake` inside `zgp-vllm-pr38479-dev`
- Re-run editable install

## Official-container Source Install Attempt 3

### Actions

- Installed missing build tools in the official image:
  - `cmake`
  - `ninja-build`
- Retried editable install from the PR worktree with the same compile settings.

### Result

- Install progressed further again:
  - editable metadata built
  - `build_ext` ran
  - CMake configure started
- New failure:
  - repository now requires `cmake >= 3.26`
  - Ubuntu 22.04 apt package only supplied `cmake 3.22.1`

### Root Cause

- The official runtime image is still missing a sufficiently new CMake for current source builds.
- Exact error:
  - `CMake 3.26 or higher is required. You are running version 3.22.1`

### Additional observation

- PyTorch still prints a CUDA initialization warning in the container:
  - `Error 803: system has unsupported display driver / cuda driver combination`
- At this stage that warning has not blocked the build itself; the hard failure remains the CMake version gate.

### Next fix

- Upgrade container CMake to `>= 3.26`
- Retry editable install again

## Official-container Source Install Attempt 4

### Actions

- Upgraded CMake inside the container from Ubuntu apt `3.22.1` to Python wheel `3.31.10`.
- Retried editable install from source.

### Result

- Build progressed through:
  - editable metadata
  - `build_ext`
  - CMake configure
  - CMake generate
- It still failed during `cmake --build`.

### Follow-up debugging

- Switched from pip-wrapped build to manual CMake reproduction for better error visibility.
- Saved raw configure/build logs under:
  - `/ceph/User/E01442/turboquant/doc/raw/pr38479-manual-cmake-configure.log`
  - `/ceph/User/E01442/turboquant/doc/raw/pr38479-manual-build-fa2.log`
  - `/ceph/User/E01442/turboquant/doc/raw/pr38479-manual-build-cumem.log`
  - `/ceph/User/E01442/turboquant/doc/raw/pr38479-manual-build-moe.log`
- Manual findings:
  - `cumem_allocator` builds successfully
  - `_vllm_fa2_C` starts compiling normally and is not an immediate failure point
  - `_moe_C` fails immediately on a missing CUDA development header

### Root Cause Identified

- First concrete failing target: `_moe_C`
- First concrete compile failure:
  - `fatal error: cusparse.h: No such file or directory`
- Conclusion:
  - the official `vllm-openai` runtime image still lacks at least one CUDA development package required for rebuilding current vLLM source
  - the next required package is the CUDA cuSPARSE development package matching CUDA 12.9

### Additional note

- During manual configure without explicit `TORCH_CUDA_ARCH_LIST`, CMake picked a much broader architecture list than desired.
- For actual rebuild attempts, the preferred constraint remains `TORCH_CUDA_ARCH_LIST=8.0` on this A100 host.

### Next fix

- Install the missing CUDA sparse development headers for 12.9
- Retry the source build

## Official-container Source Install Attempt 5

### Additional root-cause tracing

- Inspected PyTorch CUDA header dependencies in:
  - `torch/include/ATen/cuda/CUDAContextLight.h`
- Confirmed that it directly includes:
  - `cusparse.h`
  - `cublas_v2.h`
  - `cublasLt.h`
  - `cusolverDn.h` when `CUDART_VERSION` is defined

### CUDA dev packages added after target-level diagnosis

- Already added:
  - `cuda-nvrtc-dev-12-9`
  - `libcublas-dev-12-9`
- Then added after `_moe_C` failure:
  - `libcusparse-dev-12-9`
  - `libcusolver-dev-12-9`

### Manual-target diagnosis result

- `_moe_C` was the first target with a concrete compile failure before the new CUDA dev packages were installed.
- Exact first hard compiler error:
  - `fatal error: cusparse.h: No such file or directory`
- After installing cuSPARSE/cuSOLVER development packages, `_moe_C` progressed past that point and continued compiling.

### Current install status

- Re-ran the official editable install after adding the CUDA development packages.
- The install is now in a long-running real compile phase inside `zgp-vllm-pr38479-dev`.
- Confirmed active compilation of:
  - `_C`
  - `_moe_C`
  - `_vllm_fa2_C`
- No new immediate hard failure has surfaced yet after adding:
  - `cuda-nvrtc-dev-12-9`
  - `libcublas-dev-12-9`
  - `libcusparse-dev-12-9`
  - `libcusolver-dev-12-9`

### Raw records added

- `/ceph/User/E01442/turboquant/doc/raw/pr38479-manual-build-moe-2.log`

### Current working hypothesis

- The official `vllm-openai` runtime image can be converted into a usable PR #38479 source-build container, but only after layering in several missing CUDA development packages.
- The remaining issue, if any, is now likely to be a deeper compile/link incompatibility rather than missing basic build prerequisites.

## Continuing compile status

### Additional container packages installed during source-build conversion

- `git`
- `cmake` via Python wheel:
  - `cmake==3.31.10`
- CUDA development packages added on top of the runtime image:
  - `cuda-nvrtc-dev-12-9`
  - `libcublas-dev-12-9`
  - `libcusparse-dev-12-9`
  - `libcusolver-dev-12-9`

### Verified files after package installation

- `/usr/local/cuda/lib64/libnvrtc.so`
- `/usr/local/cuda/include/cublas_v2.h`
- `/usr/local/cuda/include/cusparse.h`
- `/usr/local/cuda/include/cusolverDn.h`

### Current status of the official-image editable install

- The active install command remains:
  - `uv pip install --system -e . --torch-backend=auto --no-build-isolation --no-deps`
- Compile constraints used:
  - `TORCH_CUDA_ARCH_LIST=8.0`
  - `MAX_JOBS=16`
  - `NVCC_THREADS=8`
- The build is no longer failing immediately on missing tools or missing CUDA headers.
- As of the latest check, the build is still actively compiling CUDA code inside:
  - `_moe_C`
  - `_C`
  - `_vllm_fa2_C`
- The currently most expensive visible work is `sm80` MoE/Marlin CUDA kernel compilation.

### Build-directory evidence

- The active pip/setuptools build is running from:
  - `/tmp/tmpjldf2hye.build-temp`
- The active Ninja state files there include:
  - `.ninja_log`
  - `.ninja_deps`
  - `build.ninja`
- Latest `.ninja_log` entries confirm forward progress rather than a deadlock.
- Recent completed objects include multiple FlashAttention 2 `sm80` objects such as:
  - `flash_fwd_split_hdim128_*_sm80.cu.o`
  - `flash_fwd_split_hdim192_*_sm80.cu.o`
  - `flash_fwd_split_hdim256_*_sm80.cu.o`
- This indicates the official-image source build has advanced past setup failures and is spending time in real CUDA compilation work.

### Latest monitored status

- Latest `.ninja_log` line count observed:
  - `126` completed Ninja entries
- More recent completions after that checkpoint include additional FlashAttention 2 split-forward `sm80` objects such as:
  - `flash_fwd_split_hdim32_bf16_sm80.cu.o`
  - `flash_fwd_split_hdim32_fp16_causal_sm80.cu.o`
- Latest live process checks still show active CUDA compilation inside the main editable install, including:
  - `_vllm_fa2_C`
  - `_moe_C`
- No new immediate hard compiler or linker error has surfaced after the CUDA dev package fixes.

## Why Hopper objects are being built on an A100 host

### User question

- Why is the build compiling Hopper (`sm90`) objects on an A100 machine?

### Root cause from source inspection

- The flash-attention subproject does **not** decide targets from the currently attached GPU model.
- Instead, it decides from the configured CUDA architecture list (`CUDA_ARCHS`).
- Relevant source:
  - `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/.deps/vllm-flash-attn-src/CMakeLists.txt`
- In that file:
  - `FA3_ENABLED` is `ON`
  - FA3 target architectures are selected via:
    - `cuda_archs_loose_intersection(FA3_ARCHS "9.0a;" "${CUDA_ARCHS}")`
- Therefore:
  - if `CUDA_ARCHS` contains `9.0a`, Hopper/FA3 sources are compiled
  - this is true regardless of the actual host GPU being A100

### Evidence from this session

- The generated build is actively producing `_vllm_fa3_C` `sm90` object files.
- The active Ninja log already contains many completed Hopper instantiations.
- This proves the current build configuration still includes a Hopper-capable architecture set.

### Current interpretation

- The issue is not incorrect runtime GPU detection.
- The issue is that the effective architecture list used by this editable build has not been reduced to `sm80`-only.
- The intended `TORCH_CUDA_ARCH_LIST=8.0` constraint has not fully propagated into all relevant CMake/flash-attn architecture decisions for this build configuration.

## Local Qwen3.5 TurboQuant smoke result

### Model path used

- `/share/models/official/Qwen3.5-35B-A3B`

### Runtime environment adjustments needed

- Required corrected CUDA runtime path:
  - `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu`
- Required local code fixes already applied:
  - hybrid/mamba config import regression
  - TurboQuant centroids runtime `scipy` removal

### Smoke outcome

- A real single-GPU eager smoke run with:
  - `dtype=bfloat16`
  - `kv_cache_dtype=tq3`
  - `max_model_len=128`
  - `gpu_memory_utilization=0.94`
- successfully initialized the engine, loaded the Qwen3.5 model, created KV cache, and generated output.

### Key observed logs

- `Using TURBOQUANT attention backend out of potential backends: ['TURBOQUANT']`
- `GPU KV cache size: 11,136 tokens`
- successful output generation from the prompt:
  - `"Write one short sentence about TurboQuant."`

### Generated output observed

- Output began with:
  - `'\n\n<think>\nThinking Process:\n\n1.  **An'`

### Additional issue discovered during successful smoke

- TurboQuant still fell back from the intended CUDA kernels to Triton because the runtime compiler looked for non-existent source paths:
  - missing store kernel path:
    - `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/vllm/v1/attention/ops/tq_store_cuda.cu`
  - missing decode kernel path:
    - `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/vllm/v1/attention/ops/tq_decode_warp_per_head.cu`
- The real sources in this PR are under `vllm/v1/attention/ops/csrc/`.
- Therefore:
  - end-to-end `tq3` generation works
  - but the CUDA fast paths for TQ store/decode are not being found, so the run used Triton fallback instead of the intended CUDA kernels

## Local Qwen3.5 TurboQuant smoke success

### Final successful smoke configuration

- Container:
  - `zgp-vllm-pr38479-share`
- Model:
  - `/share/models/official/Qwen3.5-35B-A3B`
- Environment:
  - `CUDA_VISIBLE_DEVICES=1`
  - `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu`
- Runtime config:
  - `dtype=bfloat16`
  - `kv_cache_dtype=tq3`
  - `max_model_len=128`
  - `gpu_memory_utilization=0.94`
  - `enforce_eager=True`

### Successful outcome

- Engine initialized successfully
- Model loaded successfully
- TurboQuant backend selected successfully
- Decode CUDA kernel compiled successfully:
  - `TQ WPH CUDA kernel compiled (smem=True)`
- A real generation completed

### Generated text observed

- `'\n\n<think>\n\n</think>\n\nTurboQuant is a high-performance'`

### Updated interpretation

- PR #38479 is now validated to the point of:
  - source build succeeds in an official-image-based dev container
  - local Qwen3.5 model can initialize with `kv_cache_dtype=tq3`
  - generation can complete on A100

### Remaining issue still observed

- TQ CUDA **decode** kernel compiled successfully in the final smoke run.
- TQ CUDA **store** kernel still failed and fell back to Triton, due a compile error in `tq_store_cuda.cu`:
  - macro `LAUNCH` is defined with 3 parameters
  - several call sites pass only 2 parameters
- This is the main remaining clear TurboQuant fast-path bug discovered in the session.

## TQ CUDA store fast-path fix

### Root cause

- `vllm/v1/attention/ops/csrc/tq_store_cuda.cu` defined:
  - `#define LAUNCH(NR, NQ, VQ)`
- but the call sites passed only two arguments:
  - `LAUNCH(true, 8)` and similar
- The actual kernel template only takes two template parameters:
  - `<bool NO_QJL, int VQB>`

### Code change

- Fixed the launch macro to match the real kernel template arity:
  - `#define LAUNCH(NQ, VQ)`
  - `tq_fused_store_kernel<NQ, VQ>(...)`

### File changed

- `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/vllm/v1/attention/ops/csrc/tq_store_cuda.cu`

## End-to-end local success on Qwen3.5

### Final successful configuration

- Container:
  - `zgp-vllm-pr38479-share`
- Model:
  - `/share/models/official/Qwen3.5-35B-A3B`
- Environment:
  - `CUDA_VISIBLE_DEVICES=1`
  - `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu`
- Runtime settings:
  - `dtype=bfloat16`
  - `kv_cache_dtype=tq3`
  - `max_model_len=128`
  - `gpu_memory_utilization=0.94`
  - `enforce_eager=True`

### Final observed result

- Engine initialization succeeded
- Model loading succeeded
- KV cache initialization succeeded
- TQ CUDA store kernel compiled successfully:
  - `TQ CUDA store kernel compiled for D=256`
- TQ CUDA decode kernel compiled successfully:
  - `TQ WPH CUDA kernel compiled (smem=True)`
- Real generation completed successfully

### Final generated text observed

- `'\n\n<think>\n\n</think>\n\nTurboQuant is a high-performance'`

### Current validated conclusion

- On this A100 machine, with the fixes applied during this session, PR #38479 can:
  - build from source in an official-image-based container
  - initialize `Qwen3.5-35B-A3B` with `kv_cache_dtype=tq3`
  - compile the TQ CUDA store and decode fast paths at runtime
  - complete real text generation

## Editable install completion

### Result

- The official-image editable install finally completed successfully.
- Installed package now reports:
  - `vllm==0.18.1rc1.dev258+gdaaf633e4`
  - editable project location:
    - `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`

### Note

- The install required a long compile but did not surface a new hard build failure after the CUDA development packages were added.

## Runtime smoke attempt 1

### Test

- Attempted a minimal eager single-GPU generation with:
  - model: `sshleifer/tiny-gpt2`
  - `kv_cache_dtype=\"tq3\"`
  - `dtype=\"float16\"`
  - `enforce_eager=True`

### Result

- Model/config parsing started normally.
- Engine initialization failed before generation began.

### Root Cause Seen So Far

- Failure is at CUDA runtime initialization inside the container, not at TurboQuant config parsing.
- Exact runtime error:
  - `RuntimeError: Unexpected error from cudaGetDeviceCount() ... Error 803: system has unsupported display driver / cuda driver combination`

### Interpretation

- This new blocker is a container GPU runtime compatibility problem.
- It occurs after the PR build/install succeeds.
- It prevents meaningful serving/inference smoke tests until resolved.

### Next debugging step

- Check `nvidia-smi` and basic torch CUDA availability inside the active container.
- Determine whether the issue is intrinsic to the official image on this host or caused by subsequent package changes inside the container.

## Runtime smoke attempt 1 diagnostics

### Evidence collected

- `nvidia-smi` works inside `zgp-vllm-pr38479-dev` and sees all 8 A100 GPUs.
- `torch` inside the same container reports:
  - `torch 2.10.0+cu129`
  - `torch.version.cuda == 12.9`
  - `torch.cuda.device_count() == 8`
  - `torch.cuda.is_available() == False`
- The warning/error remains:
  - `Error 803: system has unsupported display driver / cuda driver combination`

### Interpretation

- GPU device exposure into the container is working.
- The problem is now narrowed to CUDA runtime library resolution from PyTorch, not Docker GPU passthrough itself.
- A likely next suspect is incorrect runtime loading of CUDA driver stubs versus real driver libraries.

## Runtime smoke attempt 1 root cause refinement

### Additional evidence

- With the container's default runtime library search path, PyTorch reported:
  - `torch.cuda.is_available() == False`
  - `torch.cuda.device_count() == 8`
  - `Error 803`
- After forcing a reduced runtime library path:
  - `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu`
  - or even `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu`
- PyTorch then reported:
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 8`
  - device 0 visible as `NVIDIA A100-SXM4-80GB`

### Refined root cause

- The failure is caused by incorrect CUDA runtime library resolution in the container's default environment.
- The container can see the GPUs, but the default library path causes PyTorch CUDA initialization to use an incompatible library combination.
- Overriding `LD_LIBRARY_PATH` to prefer the host driver libraries fixes the CUDA runtime initialization path.

### Next step

- Re-run the minimal `kv_cache_dtype=tq3` generation smoke test with the corrected `LD_LIBRARY_PATH`

## Hybrid-config regression fix

### Root cause

- `vllm/model_executor/models/config.py` in the PR worktree had regressed relative to the base vLLM file:
  - multiple imports required by `HybridAttentionMambaModelConfig.verify_and_update_config` were missing
  - the saved local variable `mamba_block_size` had also been removed, but later code still referenced it

### TDD-style repro

- A minimal one-off test script was executed in the container using stubbed dependencies and a fake hybrid-model config with:
  - `cache_dtype=\"tq3\"`
  - `mamba_cache_mode=\"all\"`
- Before the fix, it failed with:
  - `NameError: STR_DTYPE_TO_TORCH_DTYPE is not defined`

### Code changes applied

- Restored imports in:
  - `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/vllm/model_executor/models/config.py`
  - added back:
    - `lcm`
    - `ModelRegistry`
    - `cdiv`
    - `STR_DTYPE_TO_TORCH_DTYPE`
    - `AttentionBackendEnum`
    - `FullAttentionSpec`
    - `MambaSpec`
    - `MLAAttentionSpec`
- Restored local preservation of:
  - `mamba_block_size = cache_config.mamba_block_size`

### Post-fix check

- Re-ran the one-off repro script after the code change.
- The script completed successfully and produced:
  - `block_size == 256`
  - `mamba_block_size == 256`

### Files changed

- `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/vllm/model_executor/models/config.py`
- `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/tests/test_config.py`

## TurboQuant centroids runtime dependency fix

### Root cause

- `vllm/turboquant/centroids.py` called `from scipy import integrate` inside
  `solve_lloyd_max()`
- In the runtime container, `scipy` was unavailable or incompatible with the
  active `numpy`, causing TurboQuant initialization to fail before model load
  completed

### TDD-style repro

- A minimal one-off script calling:
  - `from vllm.turboquant.centroids import get_centroids`
  - `get_centroids(128, 2)`
- failed before the fix with an import error originating from SciPy

### Code change

- Replaced the runtime numerical integration dependency with analytic formulas
  for a zero-mean Gaussian:
  - denominator via Gaussian CDF
  - numerator via `sigma^2 * (pdf(a) - pdf(b))`
- This removed the runtime `scipy` requirement from
  `vllm/turboquant/centroids.py`

### Post-fix check

- Re-ran the one-off centroid script successfully
- Observed:
  - shape: `torch.Size([4])`
  - dtype: `torch.float32`

### File changed

- `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant/vllm/turboquant/centroids.py`
- `/ceph/User/E01442/turboquant/doc/raw/pr38479_qwen35_tq3_smoke.py`

### Known non-blocking observation

- The container process list contains some defunct `cicc/nvcc/ninja` children from earlier interrupted diagnostic builds.
- The active main build is still the pip-driven editable install process started from the PR worktree.

## Code Changes Recorded So Far

- `/ceph/User/E01442/turboquant/vllm/.gitignore`
  - added `.worktrees/`
- `/ceph/User/E01442/turboquant/doc/2026-03-31-pr38479-vllm-repro-log.md`
  - created this log
