# PR 最新版与插件方案对比实验

日期：`2026-04-15`

## 目标

本轮实验的目标是把 GitHub upstream 上 PR `#38479` 的最新状态拉到共享工作区，并与当前 `turboquant-vllm` 插件方案在同一模型、同一机器、同一 benchmark 和同一 `lm_eval` 抽样任务上做直接比较。比较维度分为两类：一类是 serving 侧的 `KV cache`、`TTFT`、`TPOT` 与吞吐；另一类是 `lm_eval` 抽样能力回归。

## 实验约定

模型统一使用 `/share/models/official/Qwen3-32B-FP8`。服务端统一放在 `guipeng@10.90.24.4`。PR 方案使用 GitHub upstream 最新 PR 分支 `pr-38479-upstream-latest`，当前提交为 `ac46a983e`。插件方案使用当前工作区中的 `turboquant-vllm`。性能测试统一使用 `vllm bench serve` 的 `/v1/completions` 路径，能力测试统一使用 `lm_eval --model local-completions`。

## 执行记录

### 本机 `zgp-vllm-pr38479-snap` 快照镜像验证

在远端 `.4` 上直接把最新版 PR 塞进 `vllm/vllm-openai:v0.19.0-x86_64-cu130` 官方 runtime 镜像后，连续暴露出一串构建环境问题：editable install 需要可写源码目录、镜像缺 `cmake`、系统 `cmake` 版本低于 `3.26`、缺 `git`、随后又卡在 `CUDA::nvrtc` 等开发库。用户因此要求先回到本机已有的 `zgp-vllm-pr38479-snap:latest` 快照镜像验证。

本机检查结果如下。`docker images` 确认镜像 `zgp-vllm-pr38479-snap:latest` 存在，镜像大小 `25.1GB`，创建于两周前。`docker image inspect` 显示它以 `vllm/vllm-openai:v0.17.0` 为基底，默认工作目录是旧 PR 工作树 `/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant`。本机空卡里最干净的是 `GPU0/GPU1/GPU7`，显存都只有 `17 MiB` 且利用率 `0%`，因此本轮本机验证优先选这三张卡里的任一张，后续真正起服务时优先使用 `GPU0`。

对快照镜像做最小工具链自检后，得到几个关键结论。镜像内已有 `uv 0.10.9`、`git`、`cmake 3.31.10`；`python3 -m pip show vllm` 显示当前安装的是一个已经编译过的 editable `vllm`，版本为 `0.18.1rc1.dev258+gdaaf633e4`，其 editable project location 指向旧 PR 工作树；`ldconfig -p` 能看到 `libnvrtc.so.12`、`libcublas.so.12`、`libcusparse.so.12` 等 CUDA 开发库。也就是说，这个快照镜像和远端 0.19 runtime 镜像的本质差别在于：它已经具备直接源码 build 最新 PR 的基本工具链和 CUDA dev 运行条件。

第一次在快照镜像里对最新版 PR `ac46a983e` 做真实重装时，使用的是直接对共享工作树执行 `uv pip install --system -e ... --no-build-isolation --no-deps`。这次构建很快失败，但失败点已经从“镜像缺工具链”缩小到“源码树残留构建缓存污染”：CMake 报错当前 `CMakeCache.txt` 位于 `.../pr-38479-upstream-latest/.deps/cutlass-subbuild/`，而缓存最初生成于另一条路径 `/workspace/pr/.deps/cutlass-subbuild`。这说明问题不在快照镜像本身，而在最新版 worktree 自带的旧 `.deps`/`vllm.egg-info` 残留。

据此做了最小修正，不改源码提交，只在容器内复制一份临时副本 `/tmp/prlatest`，并清理其中的 `.deps`、`build` 与 `*.egg-info` 后重试。第二次重试进一步证明这条路可行：`uv` 成功进入 `Building vllm @ file:///tmp/prlatest`，不再报前述 cache 路径错配。为了避免在 A100 上浪费时间编译 Hopper/Blackwell 目标，又按旧 PR 复现记录把构建参数收紧为 `TORCH_CUDA_ARCH_LIST=8.0`、`MAX_JOBS=16`、`NVCC_THREADS=8`。

按 A100 定向配置后的第三次重试已经进入真实 `nvcc` 编译阶段。容器内 `ps` 看到的关键命令是 `cmake --build . -j=2 --target ... _C ...`，后续子进程明确在执行 `nvcc ... -gencode arch=compute_80,code=sm_80 ...`，例如正在编译 `selective_scan_fwd.cu`、`cache_kernels.cu` 等对象文件。和上一轮未限制架构时不同，这次已经确认只针对 `sm_80` 编译，说明快照镜像这条路线至少已经跨过“能否直接 build 最新 PR”的主要环境门槛，当前剩余问题不再是基础工具链缺失，而只是等待完整编译是否最终成功。

随后继续轮询本机构建容器，确认这轮 build 没有静默失败。容器 `pedantic_pascal` 持续存活，主进程仍是 `uv pip install --system -e /tmp/prlatest --torch-backend=auto --no-build-isolation --no-deps`，其下游仍保持 `cmake --build` 与 `ninja` 活跃。最新一次检查时，`nvcc` 还在继续编译 `awq_marlin_repack.cu` 与 `allspark_repack.cu`，同样只带 `sm_80` 目标。因此截至本次状态检查，结论不是“已经编译完成”，而是“仍在稳定编译最新版 PR，没有出现新的构建错误，也没有退回到早期的环境缺失问题”。

为了满足“编译完成后立即测试”的要求，后续又把这次随机名的临时构建容器替换成一个持久的 build+smoke 容器 `zgp-vllm-pr38479-snap-buildsmoke`。这样做的原因很直接：之前那条 `docker run --rm ... uv pip install ...` 线路即使编译成功也会在安装完成后立刻退出并销毁环境，无法做到“同一环境里编译成功后立即起服务并 smoke”。新的 build+smoke 脚本固定了 `GPU0`、本机 `8066` 端口和 `Qwen3-32B-FP8` 模型；执行顺序是：复制最新版 PR 到 `/tmp/prlatest`、清理构建残留、重新 editable install、安装成功后直接执行 `vllm serve ... --attention-backend TURBOQUANT --kv-cache-dtype turboquant_4bit_nc`，随后用 `/v1/models` 和一个最小 `/v1/completions` 请求做 smoke。

`zgp-vllm-pr38479-snap-buildsmoke` 启动后，结果目录 `results/pr_latest_vs_plugin_qwen3_20260415/local_snap_buildsmoke/` 已开始持续写入 `build.log` 和 `env.txt`。当前日志显示它已经完成 `build+smoke start`、`start editable install`，并进入 `Using Python 3.12.13 environment at: /usr`、`Building vllm @ file:///tmp/prlatest`。进一步查看容器内进程，可以确认这一条持久化流程已经进入 `cmake /tmp/prlatest ...`、`cmake --build . --config Debug` 和 `ninja`，但在本次记录点还没有切到最终的 `vllm serve` 阶段。因此截至目前，自动化链路已经搭好且在运行中，但 smoke 结果仍需等待这条新的持久化构建容器编译完成后再给出。

后续实际跑完一轮 build+smoke 后，第一次 smoke 失败的根因也已经收敛。`build.log` 明确显示最新版 PR 已经成功 build 并安装为 editable `vllm==0.19.0`，失败发生在随后 `vllm serve` 的 GPU 初始化阶段，而不是构建阶段。`serve.log` 里的关键错误是：`torch.cuda._lazy_init()` 在 `cudaGetDeviceCount()` 上返回 `Error 803: system has unsupported display driver / cuda driver combination`。为确认这是否是最新版 PR 引入的问题，又在不重新安装 PR 的情况下直接对快照镜像做了最小容器测试：在默认镜像环境中，`nvidia-smi` 可以正常看到 A100 和驱动 `575.51.03`，但 `torch 2.10.0+cu129` 的 `torch.cuda.is_available()` 返回 `False`，随后在真正访问 CUDA tensor 时同样报 `Error 803`。这说明问题不是 TurboQuant 逻辑或最新版 PR 的 Python/CUDA 代码，而是快照镜像默认运行时环境与当前主机上 `torch 2.10.0+cu129` 的 CUDA 初始化组合存在兼容性问题。

进一步对照旧的 PR benchmark runner，发现它一直显式覆盖 `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu`，而不是沿用镜像默认的 `/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64`。复现实验验证了这个差异就是最小修复：在相同 `zgp-vllm-pr38479-snap:latest` 镜像、相同 `--gpus device=0` 条件下，只改 `LD_LIBRARY_PATH` 为旧 runner 的值后，`torch.cuda.is_available()` 立即变为 `True`，`torch.cuda.device_count()` 返回 `1`，并且能够成功创建 `tensor_device cuda:0`。因此，第二轮 build+smoke 的核心修正不是重新改 PR 代码，而是把旧 runner 里已经验证过的 `LD_LIBRARY_PATH` 修复显式收敛进自动化脚本。

在修复 `LD_LIBRARY_PATH` 之后，又顺手优化了 build+smoke 的工程细节。最初脚本会把最新版 worktree 连同巨大的 `.deps` 目录整个复制到宿主持久 build 根目录，再删除 `.deps`，导致在宿主文件系统上白白消耗大量 I/O。现在脚本已改为通过 `tar --exclude=.deps --exclude=build --exclude=*.egg-info` 只复制干净源码，同时每次使用唯一的 `BUILD_ROOT` 后缀目录，避免为了“清理上一次 build 根目录”而在启动前卡住很久。这一轮新的自动化容器固定使用 `GPU2`、本机 `8066` 端口，并持续把状态写入 `status.json`；截至当前记录点，`status` 仍为 `building`，表明新的带修复版本构建流程已经重新开始，正在等待进入下一阶段的 `cmake --build` / `nvcc` 实编与最终 smoke。

随后在本机继续往下收敛，最终把“最新版 PR 能否在 A100 + Qwen3-32B-FP8 上真正跑起来”这个问题拆成了三个独立层次。第一层是构建层，结论已经明确：最新版 PR 可以在 `zgp-vllm-pr38479-snap` 快照镜像上完成完整源码编译。第二层是源码模式运行层，最初会因为缺 package metadata 而让 `platforms.current_platform` 退回 `UnspecifiedPlatform`，但这一层也已经被最小 workaround 验证可绕过：只要为源码模式补一个最小 `vllm-*.dist-info`，并沿用旧 runner 的 `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu`，最新版 PR 就能继续往下走，不再卡在设备推断或 `cudaGetDeviceCount Error 803`。

第三层也是最终真正的硬阻塞层：`Qwen3-32B-FP8` 在最新版 PR 上会走 block-FP8 线性层 kernel 选择，而这条路径在 A100 上没有可用后备。用源码模式 one-off `LLM.generate()` 做最终验证时，运行链路已经成功进入以下阶段：模型配置解析、TurboQuant boundary skip layer 注入、边界层自动使用 `FLASH_ATTN`、中间层自动使用 `TURBOQUANT`、完整权重加载、`torch.compile`。真正失败发生在首个 profile run 的 block-FP8 GEMM 编译：默认情况下它会选中 `TritonFp8BlockScaledMMKernel`，然后在 Triton 编译时抛出 `ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")`。

为了判断这是不是单纯“选错 kernel”而不是“这个模型在 A100 上整体不支持”，又补做了一轮离线 kernel 可用性检查和一轮运行时禁用 Triton 的试验。离线检查表明，这个模型对应的 block-FP8 kernel 候选是 `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`、`DeepGemmFp8BlockScaledMMKernel`、`CutlassFp8BlockScaledMMKernel` 和 `TritonFp8BlockScaledMMKernel`。对 `sm80` 而言，前三者分别因为“FlashInfer block-scale FP8 GEMM 不可用”、“DeepGEMM 只支持 Hopper/Blackwell”和“Cutlass block-FP8 不支持 compute capability 80”而不可用；运行时再通过 `VLLM_DISABLED_KERNELS=TritonFp8BlockScaledMMKernel` 禁掉 Triton 后，报错也精确印证了这一点：A100 上没有任何剩余 kernel 能实现这条 block-FP8 线性层路径。

因此，截至本轮调试，关于最新版 PR 的最精确结论是：它已经不再被“官方 runtime 镜像工具链不够”或“源码模式平台探测失败”这些外围问题阻塞；真正的主阻塞是 `Qwen3-32B-FP8` 这类 block-FP8 权重模型在 A100 上缺少可用的 block-FP8 GEMM backend。换句话说，最新版 PR 在本机 A100 上对 `Qwen3-32B-FP8` 的 TurboQuant smoke 失败，根因不是 TurboQuant attention backend 本身，而是上游 vLLM 当前的 block-FP8 线性层 kernel 只在 Triton 路径上能走到最远，而该 Triton 路径又会生成 A100 不支持的 `fp8e4nv` 类型；禁掉 Triton 之后则根本没有任何可用 fallback。这也意味着，如果要继续做“最新版 PR vs 插件方案”的公平对比，要么换一个不依赖 block-FP8 权重 kernel 的模型，要么换到 Hopper/Blackwell 这类能跑这些 backend 的 GPU。

为了把“TurboQuant runtime 本身是否可用”与“Qwen3-32B-FP8 的 block-FP8 权重 kernel 缺口”分离开来，又进一步做了一个最小替代验证：使用共享存储里的 `/share/models/official/Qwen3-8B`。这个模型不带 `FP8` 权重，因此不会走 `Qwen3-32B-FP8` 那条 block-FP8 GEMM 路径。验证方式不是 API server，而是更直接的源码模式 one-off `LLM.generate()`：先在快照镜像里卸掉旧的 editable `vllm`，为最新版 PR 的 build 根目录补一个最小 `vllm-*.dist-info`，让 `importlib.metadata.version("vllm")` 与平台自动探测链路恢复，再用 `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu` 和最新版源码路径直接执行 `LLM(model='/share/models/official/Qwen3-8B', kv_cache_dtype='turboquant_4bit_nc', ...)`。

这次 `Qwen3-8B` 验证是成功的，而且成功位置足够靠后，足以说明最新版 PR 的 TurboQuant 运行链路在 A100 上并非整体不可用。运行日志显示：边界保护层 `0/1/34/35` 会自动回退到 `FLASH_ATTN`，中间层则自动切到 `TURBOQUANT` backend；模型权重完整加载成功；`torch.compile` 和 mixed prefill/decode CUDA graph capture 都顺利完成；随后 one-off generation 成功返回输出。运行中测得 `GPU KV cache size: 437,664 tokens`，在 `max_model_len=4096` 下 `Maximum concurrency` 约为 `106.85x`。最终 smoke prompt `"Reply with OK only."` 返回了一个有效 completion，虽然输出文本本身并不严格遵守指令，但这已经证明最新版 PR 的 TurboQuant runtime、KV cache 初始化、CUDA graph capture 和一次完整生成在 A100 + 非 block-FP8 Qwen3 模型上是可以跑通的。

因此，本轮调查可以把结论拆成两部分。第一部分是关于最新版 PR 的整体可运行性：在 A100 上它不是“完全跑不起来”，只要模型本身不触发当前缺失的 block-FP8 权重 kernel，就可以完成 TurboQuant runtime 初始化和真实生成。第二部分是关于本轮指定比较对象 `Qwen3-32B-FP8`：它在最新版 PR 上的失败是一个更窄、更明确的上游 kernel 覆盖问题，而不是 TurboQuant attention backend 逻辑错误。后续如果继续坚持用 `Qwen3-32B-FP8` 做最新版 PR vs 插件方案对比，合理路径只剩两条：一是换到 Hopper/Blackwell；二是等待上游为 A100 提供可用的 block-FP8 GEMM fallback，或者专门为这条路径做权重/kernel 侧适配。
