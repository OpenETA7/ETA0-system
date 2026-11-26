# ETA0 System

27B fullmix step2500 接收用户的完整目标，在一轮内自主观察、规划、查询进度和执行。默认的 `request` 模式让上层在开跑前看双相机画面，按需询问用户、改写指令，并可从成功轨迹库选择一个示例。整轮用完预算后，上层可以根据末态更新交接内容。下一轮 27B 从同一个环境继续，原生成功即结束请求。上层返回空内容、截断或无效交接时，在自己的预算内最多额外重试 4 次。仍失败时，27B 按原指令及已确认的用户答复继续，本轮使用零个示例。

[27B 模型部署（vLLM / DFlash2）](docs/serving.md) · [请求级运行与双系统对照](docs/request-harness.md) · [旧版轮内监督策略](docs/executor-led.md) · [基座使用说明](README.base.md)

27B 可以自主调用 `check_task_completion`，读取环境最近一次原生评估中各项目标的满足情况，再决定下一步。所有任务使用同一个工具协议。仅测试这项能力时，可用 `--mode base --completion-query` 和 `--mode base --no-completion-query` 做配对，两个配置都只使用 27B。正式的 40 任务 × 5 初态比较使用[进度查询配对消融方案](docs/progress-query-ablation.md)，它统一了两组的回执采集和提前 finish 反馈。

## 模型权重与推理部署（HuggingFace）

| 权重 | 用途 | HuggingFace 链接（发布前替换占位符） |
|---|---|---|
| **ETA0-27B** | 主执行模型（支持标准 vLLM 与 DFlash2） | `https://huggingface.co/OpenETA7/ETA0-27B-PLACEHOLDER` |
| **ETA0-27B-DFlash2** | 配套 DFlash2 投机解码草稿权重（支持**单卡 RTX 5090 本地高速开箱部署**） | `https://huggingface.co/OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER` |

本仓库提供无内网路径硬编码的通用启动脚本 [deploy/serve_vllm.py](deploy/serve_vllm.py)，自动按本地 GPU 数量与显存配置 TP 和上下文长度、完成 512×512 视觉验收并生成 `client.json`：

```bash
# 方案 1：标准 vLLM（自动按卡数配置 TP 与上下文）
python3 deploy/serve_vllm.py --model OpenETA7/ETA0-27B-PLACEHOLDER

# 方案 2：vLLM + DFlash2 投机解码（如单卡 RTX 5090 本地高速部署）
python3 deploy/serve_vllm.py \
  --model OpenETA7/ETA0-27B-PLACEHOLDER \
  --draft OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER
```

更多单卡 RTX 5090 / 多卡参数、量化选项、`--dry-run` 预览与跨机访问说明见[27B 模型部署指南](docs/serving.md)。

## 给 coding agent 的部署提示词

把下面这句话直接发给任意有终端权限的 coding agent（如 Claude Code、Codex、Cursor、Antigravity 等）：

> 请克隆 OpenETA7/ETA0-system 并按 README 帮我部署和跑通 ETA0 System：先检测我当前设备的系统、GPU/显存、Python 与渲染环境，不要假定我使用哪类设备或推理框架；再根据我实际的 27B 模型条件（复用已有 OpenAI 兼容服务、使用 deploy/serve_vllm.py 在本地或远程 GPU 上以标准 vLLM 或 DFlash2 启动 27B，或暂未部署 27B 时先用离线演示验证）只向我询问缺失的必要配置，完成安装后跑通一次完整请求（优先开启 `--gate` 浏览器交接界面），最后向我汇报检测到的环境、部署路径、实际启用的功能、环境验证目标、成功标记和回执路径。

### Coding Agent 部署路径决策（不预设硬件）

Agent 读取本节后，应先执行环境探测（OS、`nvidia-smi` / Apple Silicon、Python / `uv`、网络连通性及已有端点），再按用户实际情况选择对应路径：

| 场景 | 适用设备与前提 | 推荐执行路径 |
|---|---|---|
| **A. 已有可访问的 27B 服务** | 任意 Linux / macOS 机器，能访问本机、局域网或云端 27B 端点 | 将端点写入私有 `client.json`，运行 `python3 deploy/first_run.py --gate 8765` |
| **B. 在自有 GPU 上部署 27B（标准 vLLM / DFlash2）** | 单卡 RTX 5090、单卡 80 GB 或多卡 NVIDIA GPU 工作站/云主机 | 运行 [deploy/serve_vllm.py](deploy/serve_vllm.py)（见[部署指南](docs/serving.md)），自动探测 GPU 并输出 `client.json`；执行器自动适配上下文长度（无 `/tokenize` 时回退本地估算，科研复现可加 `--strict-serving-profile`），再按路径 A 运行 |
| **C. 暂无 27B 服务 / 笔记本尝鲜** | macOS / Linux / 纯 CPU 环境，或先验证安装与 GUI 交互 | 运行下方「本地体验」中的离线工作流 `examples/request_demo.py --gate 8765` |

### 首个真实 LIBERO 请求

需要可访问的 27B OpenAI Chat Completions 服务（有 vLLM `/tokenize` 时精确计费压缩，无该接口时自动启用本地保守估算）、Git 和 `uv`。渲染依赖按平台区分：
- **Linux 无头仿真机**：默认使用 Mesa 离屏渲染（`MUJOCO_GL=osmesa`，Ubuntu 需安装 `libosmesa6`、`libgl1`）；若有可用 GPU 也可显式设置 `MUJOCO_GL=egl`。
- **macOS**：使用系统原生图形框架，无需安装 Mesa。

从仓库根目录运行（可先用 `--list-tasks` 浏览 40 个内置 LIBERO 任务，或用 `--dry-run` 预览完整命令而不启动仿真；配置格式参考 [examples/client.example.json](examples/client.example.json) 与 [examples/upper.example.json](examples/upper.example.json)，私有文件建议 `chmod 600`）：

```bash
# 查看任务列表或通过声明式 Profile 预览
python3 deploy/first_run.py --list-tasks --suite libero_goal
python3 deploy/first_run.py --profile configs/profiles/libero_goal_drawer.json --dry-run

# 执行真实 LIBERO 请求
export ETA0_EXECUTOR_CLIENT=/path/to/private/client.json
python3 deploy/first_run.py --profile configs/profiles/libero_object_default.json --gate 8765
```

启动器在用户缓存中创建 Python 3.11 环境、按依赖指纹缓存安装仿真依赖（后续运行无需重复联网，加 `--reinstall` 可强制刷新）并拉取固定版本的 LIBERO 资产。首次安装需要访问 GitHub 与 Python 包源（Linux 下自动使用 PyTorch CPU 源）。设置 `ETA0_SIM_PYTHON`、`ETA0_LIBERO_ROOT` 可复用已有的解释器和固定版本资产；缓存目录可用 `ETA0_CACHE` 指定。默认使用 LIBERO-Object 的第 0 个任务、初态 5，每次写入新的 `outputs/first-request-*`，任务未完成也会报告原因。

| 私有配置 | 启用的功能 |
|---|---|
| `ETA0_EXECUTOR_CLIENT` | 27B 自主执行；JSON 含 `base_url`、`model`、`api_key`，可选 `max_model_len`、`api_style`（`"vllm"` 或 `"openai"`） |
| 再提供 `ETA0_UPPER_CONFIG`（或 `ETA0_UPPER_BASE_URL`/`MODEL`/`API_KEY`） | 视觉上层澄清和改写；JSON 可指定 `allowed_returned_models` |
| 再提供 `ETA0_DEMO_LIBRARY` | 从成功轨迹库选择示例；值为 `library.json` 路径 |
| 可选 `--profile` / `--skills-dir` / `--skill` | 加载声明式任务配置与 SHA-256 校验的只读 Markdown Skill 操作卡（如 [skills/](skills/)，遵循 Tool/Skill 严格分离原则） |

**打开可视化 GUI Gate**：运行 `python3 deploy/first_run.py --gate 8765`，在浏览器打开终端打印的完整会话链接（含 `#` 后的会话令牌）。若在远程机器运行，先在本地电脑执行 `ssh -N -L 8765:127.0.0.1:8765 <远程机器 SSH 别名>`。界面提供 5 阶段流程进度条、等高双相机放大检视、场景遥测信息、多轮澄清对话、指令改写对比与一键恢复、轨迹示例开关、**运行期实时双相机画面与安全停止按钮**以及可一键复制的验收回执。省略 `--gate` 即可直接执行；带上层且无交互终端时，待答澄清会随回执返回。

**多机型与真机生态**：本仓库聚焦 ETA0-27B 的请求级工作台与 LIBERO 评测，核心循环 [harness.py](harness.py) 支持可插拔的 `observation_writer`、`query_backend` 与 `try_render_tcp_overlay`。如需双臂 PiPER、Galaxea R1 Pro 或 Franka/DROID 真机适配器（`readback → preview → execute` 三段安全接入），请参阅同源兄弟仓库 [OpenETA7/ETA-Harness](https://github.com/OpenETA7/ETA-Harness)。历史实验依据与产品设计见[11 月 24 日设计提案](docs/attempt-design-2026-11-24.md)。

## 本地体验

需要 Python 3.10 或以上版本。下面的离线示例使用脚本模型与两步模拟机器人。第一轮只够执行一个动作，第二轮从上次位置续做。打开输出目录的 `RESULT.json` 和 `events.jsonl` 可查看澄清、轮次、终止原因与调用数。

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python examples/request_demo.py --output outputs/first-request
# 可选，在浏览器中回答脚本问题并审阅这次交接
.venv/bin/python examples/request_demo.py --gate 8765 --output outputs/gui-demo
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

`deploy/run_libero.py` 支持 `ETA0_RUNTIME_PLAN` 并兼容旧集群快照；新部署直接使用仓库内的仿真适配器。其他 benchmark 可以通过 `harness.run(benchmark, policy, hierarchy=...)` 接入。

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
