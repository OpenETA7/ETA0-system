# 27B 主动查询环境进度的配对消融

这轮比较的是给 27B 开放进度查询后，任务成功率和执行成本如何变化。40 个任务来自 LIBERO-Spatial、Object、Goal、LIBERO-10 四个套件，各 10 项。每个任务取官方初态索引 0–4，每个初态分别运行查询关闭和查询开启两组，共 200 对、400 个 episode。当前阶段为准备，运行结果表中的空值代表尚未观测。

## 配对与控制变量

| 项目 | 预设 |
|---|---|
| 执行器 | fullmix 27B step2500，Qwen 原版 DFlash2。每对固定到同一推理服务副本。planner、supervisor、teacher 全关。 |
| 实验差异 | 开启组看到工具说明，可以主动调用 `check_task_completion`。关闭组无法调用。开启组自然选择零次查询的 episode 仍纳入开启组。 |
| 两组共同获得的反馈 | revised 0.3.2 指令、图像、实测状态、视觉查询和原生成功信号。两组同样采集进度供审计；27B 提前报完成时只收到相同的原生完成反馈。 |
| 随机性与预算 | 同一初态的环境和模型起始 seed 均为 1234，采样设置与截断恢复相同。每条最多 2,700 秒底层墙钟、360 次执行器决策、120 个物理动作。主动查询占一次决策并消耗实际墙钟，推进 0 个物理 tick。 |
| 初态校验 | 根据固定的 LIBERO checkout，预先计算 200 个初态向量与 40 个 BDDL 的 SHA-256。分析还核对套件、任务、初态索引、seed、首帧哈希及推理副本。 |
| 并行与顺序 | 在同一条执行通道顺序跑完一对，其他通道可并发。100 对先开、100 对先关；每个套件各有 25 对先开。默认每个已有副本 2 条仿真通道，4 个 TP2 共 8 条通道。 |

固定 checkout 为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。40 个任务的原名、原生任务文件哈希和每个索引对应的初态哈希保存在[初态清单](../experiments/libero40-progress-states.json)。[专用评测入口](../deploy/progress_ablation.py)复用当前执行循环和单 episode 启动器。

## 按任务判读

- **成功率**取环境原生成功，分别给出开、关两组的成功数与比例。配对差为「仅开启组成功的对数减去仅关闭组成功的对数」除以有效配对数。只有同一任务 5 对齐全才标注改善、退化或持平。每净多成功一个初态是 +20 个百分点；一进一退会列出两个初态，即使净值为零。
- **耗时**从仿真子进程启动计至退出，包括初始化、排队、27B 推理和清理。报告列出两组平均秒数，并按相同初态计算开减关的平均差、中位差。另列两边都成功时的平均耗时差，便于辨别更快完成和更快失败。
- **实际查询次数**取执行器获准调用工具的次数，并与日志中的 `completion_query` 事件数核对。读取到 pending 也算一次。分别统计总次数、平均每条次数、至少查询过一次的 episode 数。环境被动记账和视觉查询另计。
- **分母**只使用两边均有原生结果、进程正常退出且身份校验通过的配对。正常跑到动作、决策或墙钟预算并判负的 episode 计为失败。基础设施中断、单侧缺失、初态或首帧不一致会在明细中留原因。报告展示有效对数；样本未齐时，整体指标为暂估。
- **结论范围**每个任务只有 5 个配对初态，任务方向描述这 5 个初态上的观测差异。总体按 200 个配对汇总，齐套后每个任务等权。查询次数是 27B 自己选择的，查询频繁和任务困难可能同时发生。

## 运行入口

在仓库根目录生成计划和空报告。这两条命令仅使用本地文件，无需算力或 API 凭据。

```bash
OUT=outputs/libero40-progress-query
python3 deploy/progress_ablation.py plan --output "$OUT"
python3 deploy/progress_ablation.py report --output "$OUT"
```

正式运行时，将这份仓库和初态清单放在已有 LIBERO 仿真运行环境中。以下命令用已有服务；`CLIENT_ROOT` 指同一八卡节点上四份 27B 服务的私有连接文件目录，`LIBERO_ROOT` 指固定源码 checkout。在集群结果目录先执行上面的 `plan`。

```bash
python3 deploy/progress_ablation.py run --output "$OUT" \
  --libero-source-root "$LIBERO_ROOT" --slots 2 \
  --client "$CLIENT_ROOT/replica-1/client.json" \
  --client "$CLIENT_ROOT/replica-2/client.json" \
  --client "$CLIENT_ROOT/replica-3/client.json" \
  --client "$CLIENT_ROOT/replica-4/client.json"
```

`manifest.json` 固定组别设置和 200 行配对顺序，`runtime.json` 记录实际使用的副本、并发和代码摘要。每个 episode 独立保留原始日志。在同一输出目录用同一运行配置续跑，会跳过已有完整结果，归档未完成或未评分的尝试后再执行。运行中可随时用 `report` 生成快照；`report.md` 列出 40 项任务的对照和转胜、转败初态，`report.json` 提供每对校验状态、两臂原始行及详细统计。
