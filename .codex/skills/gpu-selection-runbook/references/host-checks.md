# Host Check Commands

## Local

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## Remote `.4`

```bash
ssh guipeng@10.90.24.4 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
ssh guipeng@10.90.24.4 'docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
```

## Decision Example

- If local `gpu1` shows `17 MiB, 0%` and `.4 gpu6` shows `17 MiB, 0%`, prefer the host with fewer leftover experiment containers.
- If local `gpu1` shows `500 MiB, 0%` but `.4 gpu6` shows `17 MiB, 20%`, prefer the idle local GPU.
