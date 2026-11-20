# ETA0 分层执行设计

## 用哪类失败检验系统

11 月 18 日的测评表记录了 LIBERO 239/400、LIBERO-Pro 559/1000、RoboCasa Cosmos 61/240。VLABench 的模型 rollout 为 0/960。LIBERO 内部，object 为 85/100，10 为 41/100。长任务也有 10/10 的“开炉并放上摩卡壶”，而“黑碗放进最下层抽屉再关抽屉”为 1/10，“黄白杯放进微波炉再关门”为 0/10。首轮应检查子目标交接、相似对象指代和最后一个状态变化。失败原因需要逐帧标注。[原始测评与 trial 清单](https://sii-czxy.feishu.cn/docx/Lgz5drlp2oPxoBxQ7J0cfoV9nag)

## 一次 episode 如何执行

1. **理解意图。** Planner 同时读取原始请求、初始双视角和实测状态，输出有序的短文本指令与可观察的完成条件。原始目标在整个 episode 中保留。若“把杯子放好”对应多个杯子或多个合理目的地，上层返回一个具体问题。交互入口收集回答并重新规划，无人值守时记录问题、在动作下发前结束。
2. **执行短任务。** 例如“把黑碗放进最下层抽屉并关上”，可拆为打开目标抽屉、放碗并松爪、关上该抽屉。每一步都把对象和目标位置写全。27B 从新帧确定 TCP、坐标、夹爪和轨迹。环境只在 episode 开头 reset，跨子任务的物体位置、已执行历史和动作预算连续。
3. **生成示范。** Teacher 看当前帧，为每个新子任务生成一个 2–6 段的文字示范。每段包含“观察线索、应采取的动作、预期看到的变化”。例如“碗已在抽屉口上方、下降并释放、松爪后碗留在抽屉内”。示范注明为假想轨迹，在每次发送给 27B 的 system 消息中保留。实测位置与当前 observation ID 来自线上观测。子任务更换或重规划时替换示范。此版本实现语义轨迹 ICL，带真实示范帧和数值 waypoint 的 teacher rollout 需要独立演示环境。
4. **监督与打断。** 默认每三个已执行动作、27B 报告 finish、收到动作拒绝或明确的控制器错误时同步复核。上层收到按时间排序的视图和动作反馈，可以继续并提示、接受当前步骤、替换全部未完成步骤或停止。决策记录必须引用本次提供的帧。打断在两个动作段之间生效，一次已派发轨迹由控制器处理跟踪偏差。上层网络变慢时，下一段动作会等待。
5. **判断完成。** advance 只更新子任务进度。最后一个子任务报告完成、但原生目标仍为假时，27B 接收反馈并继续。原生目标一旦为真即结束。supervisor_stop、澄清和预算耗尽各有独立结束原因。物理执行异常保留已开始的动作记录，异常路径禁止自动重放。

## 观测、预算与回放

每次新观察保存两张原始相机图及腕部 overlay。查询图与同一 observation ID 关联。Supervisor 默认看到初始、上一、当前观察（去重），每帧带有上一段动作和实测反馈。日志保存发送的帧标签、状态、角色、提示、决定、耗时和上层返回的 token 用量。LIBERO adapter 另存控制器逐 tick 报告、视频、seed、初始状态与 BDDL 哈希，支持从 action_started 定位到物理动作。

下层沿用 512×512 图像、thinking、16K 输出、200K 阈值的文本 compact 和 262144 上下文。当前子任务及 teacher 示例在 compact 计数之前放入 system 消息，审计保留完整事件。动作预算 120，27B 动作生成预算 360，上层另有 60 次调用限额，共享 45 分钟墙钟预算。调用前和物理 dispatch 前检查期限，已在执行的控制段可能跨过该边界。API 失败或不完整的上层输出保留诊断事件并中止。此类中断从模型能力分母单列。

### 输出截断的恢复实验

已部署的 27B 在一次真实 LIBERO-10 任务 03 验收中，连续用满 16K 输出预算并返回 length，控制器未收到这些动作。开启 `--recover-length` 后，下一次生成沿用同一实测观测，临时关闭 thinking 并要求只回一个动作。该次回复仍须通过完整生成、observation ID 和几何校验。再下一次恢复正常 thinking。默认关闭此选项以便复现实验配置。比较四组系统时应保持此开关一致，并单列截断次数、恢复后合法动作率和额外时延。

## 独立副本与性能实验

| 项 | 部署配置 |
| --- | --- |
| 主模型 | `/mnt/share/yfzhang/training-mixture/exports/fullmix-qwen38-27b-step2500-20261117/hf` |
| Draft | `/root/tests/inference/dflash2-qwen38-27b-fullmix-step2500-20261117/finetuned-dflash2` |
| 服务 | 单 Pod、两张 GPU、BF16、TP2、vLLM 0.33.0、DFlash 的 speculative tokens 为 7，最多两个并发序列 |
| 执行采样 | temperature 0.8、top_p 0.95、top_k 20、repetition_penalty 1.08、presence_penalty 0 |
| 运行时 | 使用冻结的 LIBERO text-compact 评测计划定位 serving runtime 和仿真 runtime，代码与推理缓存按独立副本隔离 |

先固定上述参数做系统消融。随后固定任务和并发，只改变复核间隔（1、3、5 个动作），记录错误子目标推进率、首次偏差至停止的动作数、闭源调用费用和 episode 时延。DFlash 自身的加速需要相同输入、目标模型和并发下的 target-only 配对计时，分别报告 prefill、decode、draft 接受率和每成功任务 GPU 分钟。[原模型适配记录](https://github.com/OpenETA7/ETA0-training/blob/658ec3ec975be54411c604c011dd464bd521d565/docs/dflash2-qwen27b-fullmix2500-20261117.md)

## 2026-11-20 云端验收

独立 GPU 作业 `j-1795160878-602322` 通过带鉴权的 512×512 图像请求。另一台 CPU 节点也通过 Pod 网络读到了同一模型。两次原生冒烟都使用 libero_10 任务 03、初始状态 0、环境 seed 1234，限 1 个动作、6 次调用。第一次六次输出均被截断，0 次派发。第二次启用截断恢复，按顺序完成两次视觉查询、拒绝一次截断输出，然后派发 move_to，控制器实测执行 30 ticks，以 action_budget 结束。两次原生目标均未达成。两轮生成采样种子设置存在差异，此处验证恢复路径及动作接线，能力增益需另做配对评测。

原始输出位于 `/root/tests/inference/eta0-system/` 下的 `native-smoke-1795161712457092226`（原始）与 `native-smoke-1795161998400138055`（恢复）。各自的 `episode/rollout/events.jsonl` 和 `RESULT.json` 记录所有模型回复与动作；控制器报告位于 `episode/simulation/controller.jsonl`。上层三个角色已通过脚本化集成测试，本轮云端试验使用 27B executor。

## 怎样判断系统有提升

- 四个条件依次为 27B 原始基座、加规划、加视觉监督、加 teacher。每组保留相同的动作协议、采样配置、seed、初始物理状态、原始任务和总时间。历史 41/100 用作参照，同期运行 A 组校正 harness 版本差异。
- 先用 LIBERO-10 的 03、06、09 做故障定位，02 做回归对照。调参后冻结提示和预算，再运行完整 10 个任务 × 10 个初始状态。单列每组原生成功数、有效数、基础设施中断与预算耗尽数，报告配对胜负及其不确定性。单次集成验收无法支持成功率结论。
- 给同一初始场景构造明确指令、同义改写和有实质歧义的指令，人工预标允许的对象、目的地和必要澄清。记录首次动作前是否问对问题、回答后是否满足原始意图、已掌握任务是否退化。用独立标注集核对 supervisor 的误放行、误打断和漏打断。
- Teacher 每次只给一个示范，锁定相同模型和上下文预算，和仅给完成条件的组比较。示范仅使用执行时可见信息。保留因示范错误而偏离、以及跳过闭源调用即可成功的案例，以评估额外 token 和延迟是否值得。

基座固定于 `cd6fcd91d19472ce24ea4dbd226930bafbf907c5`。原生 LIBERO adapter 使用训练仓的 `658ec3ec975be54411c604c011dd464bd521d565` 评测发布。[基座分支](https://github.com/OpenETA7/minimal-robot-harness/tree/eval-text-compact) · [原生 adapter](https://github.com/OpenETA7/ETA0-training/blob/658ec3ec975be54411c604c011dd464bd521d565/scripts/inference/rollout_libero_minimal.py)
