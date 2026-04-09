from vllm import LLM, SamplingParams


def main() -> None:
    model = "/share/models/official/Qwen3.5-35B-A3B"
    llm = LLM(
        model=model,
        dtype="bfloat16",
        kv_cache_dtype="tq3",
        max_model_len=128,
        gpu_memory_utilization=0.94,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=1,
        trust_remote_code=True,
    )
    print("loaded_model", model)
    print("cache_dtype", llm.llm_engine.vllm_config.cache_config.cache_dtype)
    out = llm.generate(
        ["Write one short sentence about TurboQuant."],
        SamplingParams(temperature=0.0, max_tokens=12),
    )
    print("generated_text", repr(out[0].outputs[0].text))


if __name__ == "__main__":
    main()
