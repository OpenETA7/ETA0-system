# 请求级 harness

一次请求只初始化一次 LIBERO 环境。默认有两轮机会，每轮给 27B 最多 120 个动作、360 次决策、2700 秒。上层另有整个请求共享的 900 秒和 8 次逻辑调用。每轮只在交接完毕后启动 27B 时钟。请求总延迟包含上层推理、用户等待和续做。

## 首次部署与 GUI

设备无关的部署路径与私有配置见 [README](../README.md)。具备 27B 端点后可直接运行 `python3 deploy/first_run.py`，也可指定 `--suite`、`--task`、`--initial-index` 和 `--instruction`；暂未部署 27B 时可先运行 `examples/request_demo.py --gate 8765 --output outputs/gui-demo` 体验完整交接界面。例如选择中间抽屉任务：

```bash
python3 deploy/first_run.py --suite libero_goal --task 0 --initial-index 5 \
  --instruction "请打开柜子的中间抽屉" --gate 8765
```

Gate 只在首轮动作之前等待人工输入，支持双视角画面点击放大检视、澄清对话记录、指令改写状态提示与一键恢复（原话 / 上层建议）、成功轨迹示例预览与开关切换，以及执行期计时和验收回执复制。使用原指令或手动纠正的内容写入交接记录，手动更改的指令在重试时继续生效；移除示例也会应用于后续轮次。环境成功始终依据所选任务的原生条件，建议选择与用户目标一致的任务。相机图像与审批仅经绑定本机的会话端口提供，远程机器可使用 SSH 转发，终端链接含一次性会话凭证。

## 运行

```bash
python deploy/run_libero.py --mode request --suite libero_goal --task 0 --initial-index 5 \
  --executor-client "$ETA0_EXECUTOR_CLIENT" --boyue-config "$BOYUE_KIMI_PROFILE" \
  --libero-source-root "$ETA0_LIBERO_ROOT" --instruction "打开柜子的中间抽屉" \
  --interactive --rewrite --retrieve --demo-library "$ETA0_DEMO_LIBRARY" \
  --output /tmp/eta0-first-request
```

`--instruction` 替换模型看到的任务描述。评测时应与该环境原生目标一致。自定义人机对话需要实有人类回答，若没有输入可返回 `needs_clarification` 和问题。请求使用的上层 profile 要包含 `base_url`、`model`、`api_key`，建议用 `allowed_returned_models` 固定实际返回的模型。

| 开关 | 开启时的行为 | 默认值 |
|---|---|---|
| `--clarify` | 首轮准备时可向用户提问，`--interactive` 从终端读答案，最多两次 | 开 |
| `--rewrite` | 上层返回完整、以可见线索为依据的执行指令，原始目标与用户回答同时保留 | 开 |
| `--retrieve` | 上层在至多 8 条候选中选择 0 或 1 条真实成功示例 | 指定库时开 |
| `--max-attempts` | 动作、调用或执行时间用完后，从当前物理状态启动新一轮 | 2 |

三个功能均接受 `--no-*`。把三个都关闭即得到同预算的 27B-only 基线，运行时无需上层配置。检索以目标词重合度粗排，用上层模型结合实时图像、候选目标和历史动作摘要选择示例。同初态哈希自动排除。库较大时，文本粗排对跨语言目标的召回能力仍需实测。

```bash
python deploy/build_demo_library.py --output /path/to/new-library \
  /path/to/successful-episode-1 /path/to/successful-episode-2
```

每个输入 episode 需有原生成功 RESULT、终态成功记录、控制器轨迹及动作前图像。导出器采样首、中、末三个动作，保存动作类型、实测相对位移、夹爪指令、实际执行 tick、终止状态和对应外部图像。运行时校验图像摘要，并给 27B 标明历史来源。示例在本轮的每次决策中可见。

## 审查一次请求

`run.json` 记录模型、开关和预算。`rollout/events.jsonl` 记录澄清及每轮冻结的指令和示例 ID。`rollout/attempt-NN/` 保留本轮的相机帧、动作及原生结果。`rollout/RESULT.json` 包含首轮和最终结果、失败原因、调用、进度查询、各阶段用时。上层每次返回时，`preparation_response` 记录结束原因、服务端提供的输出 token 数和正文字符数，便于判断截断。

上层返回空正文、截断、无效 JSON 或不符合交接格式的内容时，每条待答消息最多额外重试 4 次。首发输出上限为 1500 tokens，发生截断后本条消息的重试上限调为 4096。所有尝试共享本请求的上层调用次数和墙钟预算。4 次重试仍未得到有效交接，或上层预算已耗尽时，27B 使用原指令和用户已经确认的答复，从当前场景继续本轮，不携带 demo。有效澄清问题仍等待用户回答。

每条无效回复的原因记录在 `preparation_invalid`，每次兜底记录为 `preparation_fallback`，最终回执的 `preparation_fallbacks` 是实际兜底轮数。评测按环境原生完成情况给兜底请求计分。模型标识校验等其他准备错误仍记为 `preparation_error`。

## 云端两组配对比较

四个套件各取全部 10 项任务、初态索引 5，每组 40 个请求。从旧实验 27B-only 关闭查询组的初态 0–4 中，按初态索引为每个任务取首条原生成功轨迹。每项测试显式排除同任务示例。两组共享每一对的 27B 副本、种子、revised 指令、进度查询和两轮预算。双系统固定 Boyue 的 `kimi-k3.1-pro`，基线关闭三个上层功能。可以先预览，再加 `--submit` 提交 0 GPU 的 CPU 仿真作业。

```bash
root=/root/tests/inference/eta0-system/cpu-remote-cc57436d67
python deploy/boyue_episode.py --request-comparison \
  --config ~/.config/robodojo/boyue-multimodel.json \
  --remote-client "$root/replica-1/client.json" --remote-client "$root/replica-2/client.json" \
  --remote-client "$root/replica-3/client.json" --remote-client "$root/replica-4/client.json"
```

若开发机可通过 SSH 访问该作业的共享卷，可追加 `--shared-ssh /path/to/ssh-wrapper`。作业把临时公钥写入本次作业目录，提交端通过 SSH 交付密文，私钥仅在 CPU 容器中使用。

使用独立计费的 Moonshot 开放平台时，将上述命令的配置参数替换为 `--config ~/.config/robodojo/official-api.json --config-profile kimi`。提交器只读取所选账户，按该配置的模型 ID 校验实际响应。实验清单记录 API 地址和模型 ID，恢复同一批请求时会核对这两项。

若批量评测网关仅在 Mac 上可达，可在命名配置下追加 `--mac-preparation --shared-ssh /path/to/ssh-wrapper`。Mac 进程须在实验期间保持运行。CPU 作业通过本次作业的私有共享目录交换双图准备请求，Mac 只向所选网关和模型发送这些请求，直连时跳过系统代理。网关密钥留在 Mac。启动时会从仿真区用一组归档图像验证交接，预检结果位于 `batch/upper-preflight.json`。正式请求的上层用时包含两端的传输和等待。

作业从旧轨迹重新构建冻结库，保存版本摘要。报告先核对任务定义、初态、首帧和语义包哈希，再统计首轮及最终成功率、逐初态转胜或转败、调用、示例采用和耗时。无人回答的澄清列为未完成请求；服务故障列为未评分。可在已有结果目录执行 `python deploy/request_comparison.py --report --output /path/to/batch`。每项任务仅测试一个留出初态，逐任务转胜或转败是单个案例。

若 CPU 作业被抢占，先通过调度器确认其已终止。沿用上述命令的副本顺序和配置，追加 `--resume-job <已终止的作业 ID> --resume-from <首个执行作业的 batch 绝对路径> --submit`。恢复作业校验冻结的执行代码和示例库，保留已经评分的成功与失败请求。尚未完成的请求从同一初态重跑，上次轨迹归档到本次作业的 `batch/interrupted/`。汇总仍在最初的 `batch/` 目录；中断前的资源消耗单列。

报告同时给出两种分母。原生评分对照要求双方都取得评分；用户请求完成率计入已明确返回的准备阶段失败。云端抢占造成的半条轨迹归档后重跑，同一请求的终态只计一次。
