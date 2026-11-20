# ETA0 System

给 fullmix step2500 的 27B 具身模型增加长程任务规划、视觉监督和单样例上下文学习。一次用户请求对应一次环境 reset。上层商用视觉模型维护目标和进度，27B 根据当前子任务、外部相机、腕部相机及实测状态选择动作。整个过程共用 benchmark 的动作校验、控制器和原生成功条件。

[系统设计与评测方案](docs/system-design.md) · [基座使用说明](README.base.md)

## 本地体验

需要 Python 3.10 或以上版本。下列示例在同一个模拟 episode 中执行两个短任务，展示 supervisor 如何切换目标，以及每次切换后 teacher 给 27B 的指导内容。模型回复和控制器均由脚本模拟。

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python examples/hierarchy_demo.py --output outputs/first-demo
.venv/bin/python -m pytest -q
```

每次使用新的输出目录。打开 `events.jsonl`，按 `subgoal → action_started → execution → upper_request → upper_response` 查看决策及引用的观测目录。每个观测目录包含外部 RGB、腕部原图、TCP 叠加图和投影元数据。

## 在 YICloud 运行一个 LIBERO episode

当前部署使用 `fdj-infra` 项目的独立作业 `j-1795160878-602322`。服务名为 `eta0-system-fullmix-2500`，本次就绪地址为 `http://172.19.138.62:18680/v1`。访问范围是具备 Pod 网络和相同 GPFS 挂载的集群客户端。启动器在图像验收通过后输出 `ETA0_READY`，其中的 `client_file` 指向该副本的连接信息和凭据。该文件按运行用户私有权限保存。下列命令在带有本仓库代码、对应挂载和仿真环境的集群容器内执行。

```bash
cd /root/tests/inference/eta0-system/releases/26be2b0b8150
export ETA0_EXECUTOR_CLIENT=/root/tests/inference/eta0-system/rjob-8d131ad6dcf558ed-88c9a4c797eb2ec1-0/client.json
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode base --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-base
```

完整层级使用支持多图输入和 Chat Completions 的商用端点。通过运行环境提供 `ETA0_UPPER_BASE_URL`（以 `/v1` 结尾）、`ETA0_UPPER_MODEL` 和 `ETA0_UPPER_API_KEY`，客户端须能访问该端点。上层将收到所选相机帧、用户目标和实测状态。

```bash
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode teacher --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-teacher
```

`--mode` 支持四种累加设置。用 `base` 单独调用 27B；`plan` 加入规划，由执行器报告子任务结束；`judge` 加入视觉监督；`teacher` 再为每个新子任务生成一段示范。对用户意图测试可传 `--instruction` 改写用户措辞，原生任务和判定仍来自 adapter。如果 27B 因超长思考返回 length，可启用 `--recover-length`，在同一观测上用一次关闭 thinking 的生成来恢复。交互场景使用 `--interactive` 回答澄清问题，批量模式会将问题写入结果。

`deploy/run_libero.py` 从冻结评测计划读取解释器、PhysLM 路径和仿真环境变量。其他 benchmark 可以通过 `harness.run(benchmark, policy, hierarchy=...)` 接入。

## 从 Mac 用 Boyue 启动一次 teacher 联调

在本机运行以下命令会预览一个 0 GPU 的 LIBERO 客户端作业，执行时追加 `--submit`。它复用现有 TP2 服务，默认读取 `~/.config/robodojo/boyue-api.json` 的 boyue 配置。用 `--cli` 可以指定本机的 yicloud 可执行文件或包装脚本。

```bash
python3 deploy/boyue_episode.py --task 3 --initial-index 0
```

提交后，终端会给出作业 ID 与 GPFS 上的 episode 路径。Mac 用该次作业的临时公钥加密凭据，交付作业将密文传入集群。解密后的配置只在 CPU 作业的本机临时目录内存在。每次运行从 Mac 重新交付，凭据轮换后沿用同一入口。客户端用配置里的 `allowed_returned_models` 检查上层实际返回的模型名。

如果运行环境已有受保护的 Boyue 配置文件，也可以把 `--boyue-config /path/to/boyue-api.json` 直接传给 `deploy/run_libero.py`。每个 episode 的 `rollout/events.jsonl` 记录 planner、teacher、supervisor 的回复和所用帧，`rollout/RESULT.json` 给出结束原因及原生成功标记。

## 部署与复核

`deploy/submit.py` 默认预览独立单副本作业请求，传 `--submit` 会新建作业。已部署的副本可以直接复用。启动命令及加载检查见 [serve.py](deploy/serve.py)。配额组的最高优先级为 3，作业可被平台抢占，客户端凭据和 Pod 地址随新副本更新。

```bash
python3 deploy/submit.py
yicloud job replicas j-1795160878-602322 --project fdj-infra
yicloud job logs j-1795160878-602322 --project fdj-infra --keyword ETA0_READY
```
