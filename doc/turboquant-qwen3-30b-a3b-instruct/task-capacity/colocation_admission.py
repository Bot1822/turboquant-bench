#!/usr/bin/env python3
"""Two-process colocation test for post-free memory reuse."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("PYTHON", str(Path(__file__).resolve().parents[3] / "vllm/.venv/bin/python"))


def gpu_mem():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        int(line.split(",")[1].strip())
        for line in result.stdout.strip().splitlines()
        if line.strip()
    ]


def launch_hold(mode: str):
    env = os.environ.copy()
    env["MODE"] = mode
    env.setdefault("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
    env.setdefault("TP", "2")
    env.setdefault("MAX_MODEL_LEN", "8192")
    env.setdefault("GPU_MEM", "0.85")
    env.setdefault("INPUT_TOKENS", "4096")
    env.setdefault("CONCURRENCY", "1")
    env.setdefault("MAX_TOKENS", "64")
    env.setdefault("SLEEP_SECS", "180")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    proc = subprocess.Popen(
        [PYTHON, str(ROOT / "hold_engine.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc


def wait_ready(proc, timeout_s: int = 240):
    deadline = time.time() + timeout_s
    captured = []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        captured.append(line.rstrip())
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("ready"):
            return payload, captured
    return None, captured


def run_second_engine():
    env = os.environ.copy()
    env["MODEL"] = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
    env["TP"] = os.environ.get("TP", "2")
    env["MAX_MODEL_LEN"] = os.environ.get("SECOND_MAX_MODEL_LEN", "2048")
    env["GPU_MEM"] = os.environ.get("SECOND_GPU_MEM", "0.70")
    env["INPUT_TOKENS"] = os.environ.get("SECOND_INPUT_TOKENS", "1024")
    env["CONCURRENCY"] = os.environ.get("SECOND_CONCURRENCY", "1")
    env["MAX_TOKENS"] = os.environ.get("SECOND_MAX_TOKENS", "32")
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = os.environ.get(
        "VLLM_ENABLE_V1_MULTIPROCESSING", "0"
    )

    result = subprocess.run(
        [PYTHON, str(ROOT / "baseline_generate.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=int(os.environ.get("SECOND_TIMEOUT", "300")),
    )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip().splitlines()[-20:],
        "stderr_tail": result.stderr.strip().splitlines()[-20:],
    }


def main():
    scenario = os.environ.get("FIRST_MODE", "baseline")
    proc = launch_hold(scenario)
    try:
        ready_payload, logs = wait_ready(proc)
        if ready_payload is None:
            print(json.dumps({
                "scenario": scenario,
                "ready": False,
                "mem_before_second": gpu_mem(),
                "hold_logs_tail": logs[-40:],
            }, ensure_ascii=False))
            return

        second = run_second_engine()
        print(json.dumps({
            "scenario": scenario,
            "ready_payload": ready_payload,
            "mem_before_second": gpu_mem(),
            "second_engine": second,
        }, ensure_ascii=False))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
