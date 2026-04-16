# TurboQuant Experiment Command Patterns

## Remote service checks

```bash
ssh guipeng@10.90.24.4 'docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep zgp- || true'
ssh guipeng@10.90.24.4 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
ssh guipeng@10.90.24.4 'curl -sf http://127.0.0.1:<port>/v1/models || true'
```

## Local service checks

```bash
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep zgp- || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
curl -sf http://127.0.0.1:<port>/v1/models || true
```

## Result extraction

```bash
jq '{request_throughput, output_throughput, total_token_throughput, median_ttft_ms, median_tpot_ms, median_itl_ms}' <result.json>
```

## Cleanup

```bash
ssh guipeng@10.90.24.4 'docker rm -f <container...> >/dev/null 2>&1 || true'
ssh guipeng@10.90.24.4 'docker ps -a --format "{{.Names}}" | grep zgp- || true'
```
