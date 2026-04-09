# TurboQuant 深度集成 vLLM 实施计划

## Goal

完成一条单实例在线服务可运行的 TurboQuant 集成路径，并具备 baseline /
TurboQuant 的多轮 benchmark A/B 能力。

## Plan

1. 读 vLLM server 启动链，找到安全注入 TQ hooks 的位置
2. 实现 TQ 版 server launcher
3. 生成长上下文 multi-turn workload 配置
4. 跑 baseline / TQ 的单实例 serving benchmark
5. 根据结果继续迭代 toward deeper integration
