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

整机模式在单个 H200 八卡作业内启动 4 个 TP2，每份使用独立端口和两张独占的卡。四份都通过图像检查后，启动 8 对独立仿真，最多同时有 16 个 episode、8 个上层请求、8 条正在解码的底层序列。同一对并行开始，交替提交顺序。完成一对就保存结果并开始该通道的下一对。任务在四套 suite 之间交错分配。上层 API 采用已有 Boyue 配置，网络重试时间计入系统耗时。

在 Mac 上使用以下命令预览。追加 `--submit` 提交整机作业。若现有 TP2 的八卡主机空余完整六卡，可传 `--tp2-node HOST --executor-client PATH` 复用该副本。可用 `--replicas 3` 在同一节点以 3 份 TP2、12 个仿真槽位和最多 6 个上层并发请求运行，实际配置写入 manifest。

```bash
python3 deploy/boyue_episode.py --libero40 --initial-index 0
```

GPFS 运行目录的 `batch/manifest.json` 固定任务、suite、seed 和资源配置。`batch/<suite>/task-NN/{base,teacher}.result.json` 记录逐次结果、首帧和初态哈希、动作数、调用数、退出状态和耗时，旁边保留两组仿真输出。作业日志中的 `LIBERO40_TRIAL` 逐次发布结果，`LIBERO40_REPORT` 发布汇总。

报告分别列出原生成功数、取得原生评分的次数与实际尝试数。无判定的中断保留原因和耗时。配对统计核验两组任务定义、初态及首帧的哈希，列出两者皆成、皆败、仅原版成功、仅新版成功，以及新版减原版的成功率差（百分点）。每次端到端墙钟从启动仿真子进程计至退出，包含环境载入、推理排队、API 重试及清理。报告两组均值、中位数、P90，以及有效对的耗时差。模型加载和云端资源排队单列在 episode 计时以外。

## CPU 区运行仿真

八卡开发机 `ws-8359c2f5cfd7da75` 使用 4 个 TP2，执行器为 fullmix step2500。本轮按 `ETA0_DFLASH_DRAFT=/mnt/model/Qwen3.8-27B-DFlash2` 选择 Qwen 原版 draft。服务启动命令如下，在开发机上运行，`CLIENT_ROOT` 取本次共享目录的新路径。每个副本通过图像验收后会写入包含实际 target 和 draft 路径的私有连接文件。

```bash
export ETA0_DFLASH_DRAFT=/mnt/model/Qwen3.8-27B-DFlash2
export PATH="/usr/local/nvidia/bin:/usr/local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
python3 deploy/serve_pool.py --persistent --total-replicas 4 --root "$CLIENT_ROOT"
```

每个 `--remote-client` 指向一份 TP2 服务在 GPFS 上发布的私有连接文件。此模式只为模拟器申请 CPU，在同一对任务中复用同一份底层服务。每份服务对应 4 个并行仿真槽位，其中 2 个属于 teacher 组。仿真进程先用真实外部相机和腕部图调用商用 teacher，检查模型名和返回的示范格式，再开始正试。该检查的耗时从 episode 计时中剔除。

```bash
python3 deploy/boyue_episode.py --libero40 --config "$HOME/.config/robodojo/deepseek-official.json" \
  --remote-client "/root/tests/inference/eta0-system/cpu-remote-cc57436d67/replica-1/client.json" \
   --remote-client "/root/tests/inference/eta0-system/cpu-remote-cc57436d67/replica-2/client.json" \
   --remote-client "/root/tests/inference/eta0-system/cpu-remote-cc57436d67/replica-3/client.json" \
   --remote-client "/root/tests/inference/eta0-system/cpu-remote-cc57436d67/replica-4/client.json"
```

2026-11-21 的 CPU 验收作业 `j-1795241773-238664` 通过内部 Pod 网络向 `lg-cmc-gpu-1164` 上的常驻 27B 服务请求模型列表，并完成蓝色图像识别。正式 CPU 评测作业 `j-1795241877-420928` 通过官方 DeepSeek Flash 双图预检。推理服务使用 vLLM、DFlash2 和 YICloud GPU 作业。该次托管 InferSet 创建被平台拒绝，`fdj_gpu_taskpool` 的最高优先级为 3，托管部署要求至少 4。

CPU 区的 Boyue 验证作业 `j-1795242672-624358` 使用相同的 LIBERO 双视角请求 `kimi-k3.1`，返回 HTTP 403。该 key 的模型列表仅含 `deepseek-v4.1-flash-1017`。当前实验保留原 harness 与官方 DeepSeek Flash 两组。Boyue 更新模型权限后，可用以下命令重测。

```bash
python3 deploy/boyue_episode.py --check-model kimi-k3.1 --submit
```

## 三个上层模型与五个初态

在同一套分层系统中分别运行三个上层模型，第四组共用原 harness 基线。每个任务测试初态 0–4，仿真和执行器 seed 固定为 1234。每个任务和初态的四组在同一 TP2 上并发执行，打乱提交顺序，每次仅运行该配对块。四个副本合计最多 16 个 episode。所有条件采用原有 45 分钟预算，正式计时含 API 等待和重试。

下面的命令默认预览，追加 `--submit` 才提交。模型 ID 取新 Boyue key 实际支持的三个视觉模型。配置可沿用 Boyue 文件格式，未指定默认模型时，比较入口使用命令行模型 ID。

```bash
root=/root/tests/inference/eta0-system/cpu-remote-cc57436d67
python3 deploy/boyue_episode.py --libero40 \
  --config "$HOME/.config/robodojo/boyue-multimodel.json" \
  --compare-models "$MODEL_A" "$MODEL_B" "$MODEL_C" \
  --initial-index 0 --initial-count 5 \
  --remote-client "$root/replica-1/client.json" --remote-client "$root/replica-2/client.json" \
  --remote-client "$root/replica-3/client.json" --remote-client "$root/replica-4/client.json"
```

三个端点先在 CPU 区接受真实 LIBERO 双图请求，校验返回的模型 ID 和示范结构。通过后共运行 200 个配对块、800 个 episode。同一版本下可分阶段提交，例如先跑初态 0，再跑初态 1–4。阶段间保持模型、harness、推理副本和预算一致。各次运行的 manifest 记录模型与分组映射。每个配对块完成后更新 episodes.json 和 report.json，其中六组两两对比都检查任务、初态和首帧哈希。逐条结果同时记录底层动作数、上层实际模型，以及规划、监督、示范各自的 API 耗时。
