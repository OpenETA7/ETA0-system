# ETA0 System

27B fullmix step2500 接收用户的完整目标，在一轮内自主观察、规划、查询进度和执行。默认的 `request` 模式让上层在开跑前看双相机画面，按需询问用户、改写指令，并可从成功轨迹库选择一个示例。整轮用完预算后，上层可以根据末态更新交接内容。下一轮 27B 从同一个环境继续，原生成功即结束请求。上层返回空内容、截断或无效交接时，在自己的预算内最多额外重试 4 次。仍失败时，27B 按原指令及已确认的用户答复继续，本轮使用零个示例。

[请求级运行与双系统对照](docs/request-harness.md) · [旧版轮内监督策略](docs/executor-led.md) · [基座使用说明](README.base.md)

27B 可以自主调用 `check_task_completion`，读取环境最近一次原生评估中各项目标的满足情况，再决定下一步。所有任务使用同一个工具协议。仅测试这项能力时，可用 `--mode base --completion-query` 和 `--mode base --no-completion-query` 做配对，两个配置都只使用 27B。正式的 40 任务 × 5 初态比较使用[进度查询配对消融方案](docs/progress-query-ablation.md)，它统一了两组的回执采集和提前 finish 反馈。

## 给 coding agent 的部署提示词

可以把以下指令发给 Codex 或其他具备终端访问能力的 coding agent：

> 请部署 OpenETA7/ETA0-system 的请求级 harness。优先使用现有 checkout，需要时克隆仓库。用 Python 3.11 创建虚拟环境并安装 dev extra，运行 `eta0-run --help` 及 `python examples/request_demo.py --output <新目录>`，读取 RESULT.json 检查两轮续做。然后按 docs/request-harness.md 检查 LIBERO 仿真解释器、固定版本 checkout、27B client.json、商用模型配置和成功示例库。实机配置齐备时，在仿真解释器安装 libero extra，用 `--mode request --rewrite --retrieve --demo-library <路径>` 跑一个 LIBERO 初态；显式传入私有客户端配置及 LIBERO 根目录。如果仅有 27B 服务，使用 `--mode request --no-clarify --no-rewrite --no-retrieve` 验证基线。如果只有示例库缺失，关闭检索，验证指令改写。报告实际启用的功能、首轮与最终成功标记、上下层调用次数、耗时、输出目录及缺少的配置。凭据从本机私有配置读取。

历史实验依据与产品设计见[11 月 24 日设计提案](docs/attempt-design-2026-11-24.md)。

## 本地体验

需要 Python 3.10 或以上版本。下面的离线示例使用脚本模型与两步模拟机器人。第一轮只够执行一个动作，第二轮从上次位置续做。打开输出目录的 `RESULT.json` 和 `events.jsonl` 可查看澄清、轮次、终止原因与调用数。

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python examples/request_demo.py --output outputs/first-request
.venv/bin/python -m pytest -q
```

## 在 YICloud 运行一个 LIBERO episode

服务名为 `eta0-system-fullmix-2500`。启动器完成图像验收后输出 `ETA0_READY`，其中的 `client_file` 指向该副本的私有连接文件。开发机和 CPU 仿真作业均需挂载同一份 GPFS。以下示例使用本轮配置的第一个副本，须先等它通过验收。

```bash
export ETA0_EXECUTOR_CLIENT=/root/tests/inference/eta0-system/cpu-remote-cc57436d67/replica-1/client.json
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode base --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-base
```

历史 teacher 模式使用支持多图输入和 Chat Completions 的商用端点。通过运行环境提供 `ETA0_UPPER_BASE_URL`（以 `/v1` 结尾）、`ETA0_UPPER_MODEL` 和 `ETA0_UPPER_API_KEY`，客户端须能访问该端点。上层将收到所选相机帧、用户目标和实测状态。

```bash
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode teacher --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-teacher
```

显式选择旧版 `--mode judge` 会开启长任务规划和停滞监督。使用 `--mode base --teacher` 可单独评估停滞后的示范；`--planner`、`--supervisor`、`--teacher` 及各自的 `--no-*` 可自由组合。`--instruction` 可覆盖用户措辞，`--recover-length` 可在思考被截断后尝试一次直接生成。新版默认加载 0.3.2 revised 指令并启用原生完成度查询。仿真解释器依赖与 `--libero-source-root` 的设置见[接入说明](docs/executor-led.md)。

`deploy/run_libero.py` 从冻结评测计划读取解释器、PhysLM 路径和仿真环境变量。其他 benchmark 可以通过 `harness.run(benchmark, policy, hierarchy=...)` 接入。

## 从 Mac 用 Boyue 启动一次 teacher 联调

在本机运行以下命令会预览一个 0 GPU 的 LIBERO 客户端作业，执行时追加 `--submit`。它复用现有 TP2 服务，默认读取 `~/.config/robodojo/boyue-api.json` 的 boyue 配置。用 `--cli` 可以指定本机的 yicloud 可执行文件或包装脚本。

```bash
python3 deploy/boyue_episode.py --task 3 --initial-index 0
```

提交后，终端会给出作业 ID 与 GPFS 上的 episode 路径。Mac 用该次作业的临时公钥加密凭据，交付作业将密文传入集群。解密后的配置只在 CPU 作业的本机临时目录内存在。每次运行从 Mac 重新交付，凭据轮换后沿用同一入口。客户端用配置里的 `allowed_returned_models` 检查上层实际返回的模型名。

如果运行环境已有受保护的 Boyue 配置文件，也可以把 `--boyue-config /path/to/boyue-api.json` 直接传给 `deploy/run_libero.py`。每个 episode 的 `rollout/events.jsonl` 记录 planner、teacher、supervisor 的回复和所用帧，`rollout/RESULT.json` 给出结束原因及原生成功标记。

上层 API 遇到超时、临时断连、限流或网关故障时会在同一组帧上最多重试 4 次（含首次共 5 次）。等待期间机器人保持静止。作业日志可查看触发重试的错误和等待秒数。

## LIBERO 40 任务配对评测

在单台八卡机上用 4 个 TP2 副本比较原 harness 与新系统，每个任务两组各运行一次，共 80 个 episode。提交入口、并行设定和统计口径见[配对评测说明](docs/libero40.md)。

## 部署与复核

`deploy/submit.py` 默认预览独立单副本作业请求，传 `--submit` 会新建作业。已部署的副本可以直接复用。启动命令及加载检查见 [serve.py](deploy/serve.py)。配额组的最高优先级为 3，作业可被平台抢占，客户端凭据和 Pod 地址随新副本更新。

```bash
python3 deploy/submit.py
yicloud workspace get ws-8359c2f5cfd7da75 --project fdj-infra --zone-id cn-shanghai-a
```
