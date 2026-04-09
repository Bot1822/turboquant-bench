#!/usr/bin/env python3
"""Long-decode streaming benchmark for baseline vs TurboQuant server modes."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from turboquant.stream_bench import (
    build_prompt_from_corpus_tokens,
    summarize_stream_metrics,
)


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
URL = os.environ.get("URL", "http://127.0.0.1:8010")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen30-base")
FREE_URL = os.environ.get("FREE_URL", "")
FREE_AFTER_FIRST_CHUNK = os.environ.get("FREE_AFTER_FIRST_CHUNK", "0") == "1"
INPUT_TOKENS = int(os.environ.get("INPUT_TOKENS", "1400"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "600"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))


def build_prompt(tokenizer, target_tokens: int) -> str:
    corpus_paths = [
        Path("/ceph/User/E01442/turboquant/vllm/benchmarks/sonnet.txt"),
        Path("/share/models/official/Qwen3-30B-A3B-Instruct-2507/README.md"),
    ]
    corpus = "\n".join(path.read_text() for path in corpus_paths)
    prompt, _ = build_prompt_from_corpus_tokens(
        tokenizer=tokenizer,
        corpus_text=corpus,
        target_tokens=target_tokens,
    )
    return prompt


async def maybe_free(session: aiohttp.ClientSession):
    if FREE_AFTER_FIRST_CHUNK and FREE_URL:
        async with session.post(FREE_URL, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            return {
                "status": resp.status,
                "body": await resp.text(),
            }
    return None


async def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = build_prompt(tokenizer, INPUT_TOKENS)
    input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream_options": {"include_usage": False},
    }

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    headers = {"Content-Type": "application/json"}

    ttft_ms = None
    chunk_count = 0
    text_chunks = []
    free_result = None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter_ns()
        async with session.post(
            f"{URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for chunk in response.content:
                if not chunk:
                    continue
                line = chunk.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                delta = event["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                now = time.perf_counter_ns()
                if ttft_ms is None:
                    ttft_ms = (now - start) / 1_000_000.0
                    free_result = await maybe_free(session)
                text_chunks.append(delta)
                chunk_count += 1

        end = time.perf_counter_ns()

    full_text = "".join(text_chunks)
    output_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
    summary = summarize_stream_metrics(
        ttft_ms=ttft_ms or 0.0,
        latency_ms=(end - start) / 1_000_000.0,
        output_tokens=output_tokens,
        chunk_count=chunk_count,
    )
    result = {
        "url": URL,
        "served_model_name": SERVED_MODEL_NAME,
        "free_after_first_chunk": FREE_AFTER_FIRST_CHUNK,
        "input_tokens": input_tokens,
        "max_tokens": MAX_TOKENS,
        "free_result": free_result,
        "sample_text": full_text[:300],
        **summary,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
