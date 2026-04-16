# TurboQuant Fused Kernel Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `.4` 上跑通 `turboquant-vllm` 的融合 decode kernel，并完成 `baseline-bf16`、`baseline-fp8`、`tq-unfused`、`tq-fused` 四组对比，输出中文性能与 KV cache 报告。

**Architecture:** 服务端统一使用 `vllm/vllm-openai:v0.19.0-x86_64-cu130` 官方镜像，TurboQuant 组在容器内安装 `turboquant-vllm`。基准压测由本地客户端发起，统一记录服务日志、benchmark JSON 和中文实验文档。优先验证非 `eager` 路径，再根据实际稳定性决定是否保留回退配置。

**Tech Stack:** Docker, vLLM 0.19.0, turboquant-vllm 1.5.x, Triton kernel, Qwen3.5-35B-A3B, A100 80GB, SSH remote execution

---

## Chunk 1: 环境与代码路径确认

### Task 1: 固化代码与环境事实

**Files:**
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/01-worklog.md`
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/12-fused-kernel-benchmark-design-2026-04-10.md`

- [ ] **Step 1: 记录当前分支、worktree 与目标模型路径**

Run: `git -C /home/E01442/.config/superpowers/worktrees/turboquant/tq-fused-bench status --short`
Expected: clean worktree

- [ ] **Step 2: 确认 fused gate、CUDA graph gate、fallback 日志位置**

Run: `rg -n "TQ4_USE_FUSED_PAGED|UNIFORM_SINGLE_TOKEN_DECODE|dynamic fallback|_fused_decode_path" turboquant-vllm/src/turboquant_vllm/vllm/tq4_backend.py`
Expected: 命中 gated path 与 fallback 分支

- [ ] **Step 3: 把这些事实补进 worklog**

Expected: worklog 中能回溯本轮实验采用的代码路径与预期风险

## Chunk 2: 服务拉起与非 eager 验证

### Task 2: 在 `.4` 上并行启动四组服务

**Files:**
- Create: `doc/turboquant-pr38479-qwen35-benchmark/results/fused_kernel_v019/`
- Create: `doc/turboquant-pr38479-qwen35-benchmark/14-fused-kernel-worklog-2026-04-10.md`
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/01-worklog.md`

- [ ] **Step 1: 准备四组容器命名、GPU 绑定、端口与环境变量**

Expected: 每个容器都使用 `zgp-` 前缀，且端口/显卡唯一

- [ ] **Step 2: 启动 `baseline-bf16` 与 `baseline-fp8`**

Run: `docker run ... vllm/vllm-openai:v0.19.0-x86_64-cu130 ...`
Expected: 服务健康，日志中出现模型加载完成和 `/v1/models` 可访问

- [ ] **Step 3: 启动 `tq-unfused` 与 `tq-fused`**

Run: `docker run ... pip install turboquant-vllm ...`
Expected: plugin 安装成功，`CUSTOM` backend 可启动

- [ ] **Step 4: 验证不加 `--enforce-eager` 时的真实行为**

Expected: 从日志判定是否命中 CUDA graph / compile 路径，或是否在 warmup/capture 失败

## Chunk 3: Smoke benchmark 与排障

### Task 3: 先做小规模对比，确认正式实验参数

**Files:**
- Create: `doc/turboquant-pr38479-qwen35-benchmark/results/fused_kernel_v019/smoke/`
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/14-fused-kernel-worklog-2026-04-10.md`

- [ ] **Step 1: 对四组服务做单次健康请求**

Run: `curl http://<host>:<port>/v1/models`
Expected: 返回模型列表

- [ ] **Step 2: 做短输入、长输出的 smoke benchmark**

Expected: 拿到 TTFT/TPOT/throughput 初值，并确认 `tq-fused` 没有立即退化或崩溃

- [ ] **Step 3: 如有失败，优先排查 JIT、capture、paged fallback、显存参数**

Expected: 给出可复现错误和修复动作，不中断全局实验

## Chunk 4: 正式 benchmark

### Task 4: 完成性能与 KV cache 对比

**Files:**
- Create: `doc/turboquant-pr38479-qwen35-benchmark/results/fused_kernel_v019/full/`
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/14-fused-kernel-worklog-2026-04-10.md`
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/04-experiments.md`

- [ ] **Step 1: 固定正式 benchmark 参数**

Expected: 至少覆盖一个中等长度档和一个长上下文档，并把并发提高到能压到 decode

- [ ] **Step 2: 并行跑四组服务 benchmark**

Expected: 产出结构化 JSON、日志与显存记录

- [ ] **Step 3: 统计 `num_gpu_blocks`、静态显存、吞吐、TTFT、TPOT**

Expected: 能直接比较 `tq-unfused` 与 `tq-fused` 的净差值

## Chunk 5: 报告与清理

### Task 5: 写中文结论报告并清理残留服务

**Files:**
- Create: `doc/turboquant-pr38479-qwen35-benchmark/15-fused-kernel-report-2026-04-10.md`
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/05-summary.md`
- Modify: `doc/turboquant-pr38479-qwen35-benchmark/01-worklog.md`

- [ ] **Step 1: 汇总四组结果，写清楚性能收益和 KV cache 收益**

Expected: 报告里明确说明何时 `tq-fused` 胜出，何时不胜出

- [ ] **Step 2: 解释融合算子为何快/慢，是否真的绕过了解压后 attention**

Expected: 结论和日志、代码路径、指标对应起来

- [ ] **Step 3: 清理 `.4` 上残留容器并回写文档**

Run: `ssh guipeng@10.90.24.4 'docker rm -f ...'`
Expected: 无残留 `zgp-` benchmark 容器
