# LIBERO 40 任务配对评测

本轮仅用 27B 的 40 × 5 进度查询开关实验使用[独立配对方案与运行入口](progress-query-ablation.md)。

新版默认比较相同 revised 指令与完成度工具下的 27B 单独执行和按需指导。两组分开核算执行器与上层墙钟。原始 80 条实验配置记录如下。[新版配置和轨迹验证](executor-led.md)

Spatial、Object、Goal、LIBERO-10 各取全部 10 个任务。每个任务固定初态 0，环境和模型起始 seed 为 1234，原 harness 与新系统各跑一次，共 80 次。每个任务只有一个初态，本轮成功率描述这 40 个场景。

| 条件 | 原 harness | 新系统 |
| --- | --- | --- |
| 执行循环 | 从上游提交 `cd6fcd91d19472ce24ea4dbd226930bafbf907c5` 打包原版 `harness.py` | 本仓库的 planner、监督者与 one-shot teacher |
| 底层 | 同任务的两个 episode 固定到同一份 fullmix-2500 + DFlash2 TP2 | 同左 |
| 观测、动作、采样 | 冻结的 LIBERO adapter，双视角，原生成功判定，开启截断恢复 | 同左 |
| 预算 | 每次 120 个动作、360 次动作生成、45 分钟墙钟 | 同左，上层另有 60 次逻辑调用 |

整机模式可在单台 8 卡节点上启动 4 个 TP2 副本（通过 `deploy/serve_vllm.py` 为每个副本分配独立端口与两张 GPU），或直接通过 `--remote-client` 复用已运行的 27B 副本。所有副本通过图像验收后，并行运行配对仿真，同一对任务共享同一份底层 27B 副本与随机种子，并交替提交顺序。任务在四套 suite 之间交错分配；上层 API 网络重试时间计入系统耗时。

```bash
python3 deploy/libero40.py \
  --boyue-config /path/to/private/upper.json \
  --remote-client /path/to/replica-1/client.json \
  --remote-client /path/to/replica-2/client.json \
  --output /path/to/batch
```

运行目录的 `batch/manifest.json` 固定任务、suite、seed 和资源配置。`batch/<suite>/task-NN/{base,teacher}.result.json` 记录逐次结果、首帧和初态哈希、动作数、调用数、退出状态和耗时，旁边保留两组仿真输出。日志中的 `LIBERO40_TRIAL` 逐次发布结果，`LIBERO40_REPORT` 发布汇总。

报告分别列出原生成功数、取得原生评分的次数与实际尝试数。无判定的中断保留原因和耗时。配对统计核验两组任务定义、初态及首帧的哈希，列出两者皆成、皆败、仅原版成功、仅新版成功，以及新版减原版的成功率差（百分点）。每次端到端墙钟从启动仿真子进程计至退出，包含环境载入、推理排队、API 重试及清理。报告两组均值、中位数、P90，以及有效对的耗时差。模型加载单列在 episode 计时以外。

## 多个上层模型与五个初态

可在同一套分层系统中通过 `deploy/compare_upper.py` 分别运行多个上层模型与原 harness 基线。每个任务测试初态 0–4，仿真和执行器 seed 固定为 1234。每个任务和初态的配对块在同一 27B 副本上并发执行，打乱提交顺序，保持预算一致。

```bash
python3 deploy/compare_upper.py \
  --boyue-config /path/to/private/upper.json \
  --models "$MODEL_A" "$MODEL_B" "$MODEL_C" \
  --initial-index 0 --initial-count 5 \
  --remote-client /path/to/replica-1/client.json \
  --remote-client /path/to/replica-2/client.json \
  --output /path/to/compare-batch
```

启动前会先用真实 LIBERO 双图请求验证各上层端点返回的模型 ID 与格式结构。通过后共运行 200 个配对块、800 个 episode。每个配对块完成后更新 `episodes.json` 和 `report.json`，其中两两对比均核对任务、初态和首帧哈希，并逐条记录底层动作数、上层实际模型及规划、监督、示范各自的 API 耗时。
