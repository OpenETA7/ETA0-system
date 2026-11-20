# LIBERO 40 任务配对评测

Spatial、Object、Goal、LIBERO-10 各取全部 10 个任务。每个任务固定初态 0，环境和模型起始 seed 为 1234，原 harness 与新系统各跑一次，共 80 次。每个任务只有一个初态，本轮成功率描述这 40 个场景。

| 条件 | 原 harness | 新系统 |
| --- | --- | --- |
| 执行循环 | 从上游提交 `cd6fcd91d19472ce24ea4dbd226930bafbf907c5` 打包原版 `harness.py` | 本仓库的 planner、监督者与 one-shot teacher |
| 底层 | 同任务的两个 episode 固定到同一份 fullmix-2500 + DFlash2 TP2 | 同左 |
| 观测、动作、采样 | 冻结的 LIBERO adapter，双视角，原生成功判定，开启截断恢复 | 同左 |
| 预算 | 每次 120 个动作、360 次动作生成、45 分钟墙钟 | 同左，上层另有 60 次逻辑调用 |

整机模式在单个 H200 八卡作业内启动 4 个 TP2，每份使用独立端口和两张独占的卡。四份都通过图像检查后，启动 8 对独立仿真，最多同时有 16 个 episode、8 个上层请求、8 条正在解码的底层序列。同一对并行开始，交替提交顺序。完成一对就保存结果并开始该通道的下一对。任务在四套 suite 之间交错分配。上层 API 采用已有 Boyue 配置，网络重试时间计入系统耗时。

在 Mac 上使用以下命令预览。追加 `--submit` 提交整机作业。若现有 TP2 的八卡主机空余完整六卡，可传 `--tp2-node HOST --executor-client PATH` 复用该副本。可用 `--replicas 3` 在同一节点以 3 份 TP2、12 个仿真槽位和最多 6 个上层并发请求运行，实际配置写入 manifest。

```bash
python3 deploy/boyue_episode.py --libero40 --initial-index 0
```

GPFS 运行目录的 `batch/manifest.json` 固定任务、suite、seed 和资源配置。`batch/<suite>/task-NN/{base,teacher}.result.json` 记录逐次结果、首帧和初态哈希、动作数、调用数、退出状态和耗时，旁边保留两组仿真输出。作业日志中的 `LIBERO40_TRIAL` 逐次发布结果，`LIBERO40_REPORT` 发布汇总。

报告分别列出原生成功数、取得原生评分的次数与实际尝试数。无判定的中断保留原因和耗时。配对统计核验两组任务定义、初态及首帧的哈希，列出两者皆成、皆败、仅原版成功、仅新版成功，以及新版减原版的成功率差（百分点）。每次端到端墙钟从启动仿真子进程计至退出，包含环境载入、推理排队、API 重试及清理。报告两组均值、中位数、P90，以及有效对的耗时差。模型加载和云端资源排队单列在 episode 计时以外。
