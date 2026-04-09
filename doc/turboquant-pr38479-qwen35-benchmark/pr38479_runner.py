from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from benchmark_helpers import completed_to_dict, ensure_dir, run_command



DEFAULT_IMAGE = "zgp-vllm-pr38479-snap:latest"
DEFAULT_MODEL = "/share/models/official/Qwen3.5-35B-A3B"
DEFAULT_LD_LIBRARY_PATH = "/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu"
MAX_CLEAN_GPU_MEMORY_MB = 1024
CASE_RUNNER_SCRIPT = (
    "/ceph/User/E01442/turboquant/doc/"
    "turboquant-pr38479-qwen35-benchmark/container_case_runner.py"
)
RETRIEVAL_RUNNER_SCRIPT = (
    "/ceph/User/E01442/turboquant/doc/"
    "turboquant-pr38479-qwen35-benchmark/container_retrieval_runner.py"
)


def query_host_gpu_memory() -> dict[int, int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    usage: dict[int, int] = {}
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        idx, mem = [x.strip() for x in line.split(",")]
        usage[int(idx)] = int(mem)
    return usage


def run_case(
    *,
    mode: str,
    prompt: str,
    output_dir: str | Path,
    image: str = DEFAULT_IMAGE,
    model: str = DEFAULT_MODEL,
    gpu_id: int = 1,
    max_model_len: int = 128,
    gpu_mem: float = 0.94,
    max_tokens: int = 16,
    warmup_runs: int = 1,
    warmup_max_tokens: int = 0,
    measure_trials: int = 1,
    result_suffix: str | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    output_stem = f"{mode}-ctx{max_model_len}-gpu{gpu_id}"
    if result_suffix:
        output_stem = f"{output_stem}-{result_suffix}"
    output_json = out_dir / f"{output_stem}-result.json"
    used_mb = query_host_gpu_memory().get(gpu_id, 0)
    if used_mb > MAX_CLEAN_GPU_MEMORY_MB:
        payload = {
            "mode": mode,
            "gpu_id": gpu_id,
            "max_model_len": max_model_len,
            "output_json": str(output_json),
            "result": {
                "status": "skipped_dirty_gpu",
                "used_mb": used_mb,
                "reason": (
                    f"GPU {gpu_id} already has {used_mb} MB allocated, "
                    f"exceeding clean threshold {MAX_CLEAN_GPU_MEMORY_MB} MB."
                ),
            },
        }
        output_json.write_text(json.dumps(payload["result"], ensure_ascii=False, indent=2) + "\n")
        return payload
    container_name = (
        f"zgp-bench-{mode}-ctx{max_model_len}-gpu{gpu_id}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "bash",
        "--gpus",
        "all",
        "--ipc=host",
        "--shm-size=16g",
        "--name",
        container_name,
        "-v",
        "/ceph/User/E01442/turboquant:/ceph/User/E01442/turboquant",
        "-v",
        "/share/models/official:/share/models/official",
        "-v",
        "/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/cache/torch_extensions:/root/.cache/torch_extensions",
        "-w",
        "/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant",
        image,
        "-lc",
        " ".join(
            [
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                f"LD_LIBRARY_PATH={shlex.quote(DEFAULT_LD_LIBRARY_PATH)}",
                "python3",
                shlex.quote(CASE_RUNNER_SCRIPT),
                "--mode",
                shlex.quote(mode),
                "--model",
                shlex.quote(model),
                "--prompt",
                shlex.quote(prompt),
                "--max-model-len",
                str(max_model_len),
                "--gpu-mem",
                str(gpu_mem),
                "--max-tokens",
                str(max_tokens),
                "--warmup-runs",
                str(warmup_runs),
                "--warmup-max-tokens",
                str(warmup_max_tokens),
                "--measure-trials",
                str(measure_trials),
                "--output-json",
                shlex.quote(str(output_json)),
            ]
        ),
    ]
    proc = run_command(cmd, timeout=3600)
    payload: dict[str, Any] = {
        "mode": mode,
        "gpu_id": gpu_id,
        "max_model_len": max_model_len,
        "container_name": container_name,
        "command": cmd,
        "process": completed_to_dict(proc),
        "output_json": str(output_json),
    }
    if output_json.exists():
        payload["result"] = json.loads(output_json.read_text())
    return payload


def run_retrieval_batch_case(
    *,
    mode: str,
    dataset_path: str | Path,
    output_dir: str | Path,
    target_tokens: int,
    image: str = DEFAULT_IMAGE,
    model: str = DEFAULT_MODEL,
    gpu_id: int = 1,
    gpu_mem: float = 0.94,
    max_tokens: int = 24,
    warmup_runs: int = 1,
    warmup_max_tokens: int = 0,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    output_json = out_dir / f"{mode}-ctx{target_tokens}-gpu{gpu_id}-retrieval-result.json"
    used_mb = query_host_gpu_memory().get(gpu_id, 0)
    if used_mb > MAX_CLEAN_GPU_MEMORY_MB:
        payload = {
            "mode": mode,
            "gpu_id": gpu_id,
            "target_tokens": target_tokens,
            "output_json": str(output_json),
            "result": {
                "status": "skipped_dirty_gpu",
                "used_mb": used_mb,
                "reason": (
                    f"GPU {gpu_id} already has {used_mb} MB allocated, "
                    f"exceeding clean threshold {MAX_CLEAN_GPU_MEMORY_MB} MB."
                ),
            },
        }
        output_json.write_text(json.dumps(payload["result"], ensure_ascii=False, indent=2) + "\n")
        return payload

    container_name = (
        f"zgp-retrieval-{mode}-ctx{target_tokens}-gpu{gpu_id}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "bash",
        "--gpus",
        "all",
        "--ipc=host",
        "--shm-size=16g",
        "--name",
        container_name,
        "-v",
        "/ceph/User/E01442/turboquant:/ceph/User/E01442/turboquant",
        "-v",
        "/share/models/official:/share/models/official",
        "-v",
        "/ceph/User/E01442/turboquant/doc/turboquant-pr38479-qwen35-benchmark/cache/torch_extensions:/root/.cache/torch_extensions",
        "-w",
        "/ceph/User/E01442/turboquant/vllm/.worktrees/pr-38479-turboquant",
        image,
        "-lc",
        " ".join(
            [
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                f"LD_LIBRARY_PATH={shlex.quote(DEFAULT_LD_LIBRARY_PATH)}",
                "python3",
                shlex.quote(RETRIEVAL_RUNNER_SCRIPT),
                "--mode",
                shlex.quote(mode),
                "--model",
                shlex.quote(model),
                "--dataset-path",
                shlex.quote(str(dataset_path)),
                "--target-tokens",
                str(target_tokens),
                "--gpu-mem",
                str(gpu_mem),
                "--max-tokens",
                str(max_tokens),
                "--warmup-runs",
                str(warmup_runs),
                "--warmup-max-tokens",
                str(warmup_max_tokens),
                "--output-json",
                shlex.quote(str(output_json)),
            ]
        ),
    ]
    proc = run_command(cmd, timeout=3600)
    payload: dict[str, Any] = {
        "mode": mode,
        "gpu_id": gpu_id,
        "target_tokens": target_tokens,
        "container_name": container_name,
        "command": cmd,
        "process": completed_to_dict(proc),
        "output_json": str(output_json),
    }
    if output_json.exists():
        payload["result"] = json.loads(output_json.read_text())
    return payload
