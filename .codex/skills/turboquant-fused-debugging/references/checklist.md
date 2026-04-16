# TurboQuant Fused Debug Checklist

## KV Cache Chain

Trace in this order:

1. slot bytes per KV head
2. attention page bytes for 1 token
3. hybrid block size chosen by vLLM
4. final `GPU KV cache size`

## Fused Startup Checklist

- launch a debug container without `--rm`
- keep `VLLM_LOGGING_LEVEL=INFO`
- grep for:
  - `operation not permitted when stream is capturing`
  - `cudaErrorStreamCaptureInvalidated`
  - `Engine core initialization failed`

## Comparison Matrix

At minimum, compare:

- `fp8`
- `tq-unfused`
- `tq-fused`

under the same:

- model
- image
- endpoint
- prompt/output lengths
- concurrency

## Project-Learned Signals

- `Qwen3.5` hybrid model made page-size bugs much worse
- `Qwen3` dense model still reproduced fused+cudagraph capture failure
- fixing cache sizing alone did not make TurboQuant beat `fp8`
