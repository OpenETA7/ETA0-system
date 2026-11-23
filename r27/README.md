# R27：Slow / Fast 异步具身 Harness

这份独立 Python 包保存了 R27 的 LIBERO 参考实现。Fast 自主选择工具并执行，Slow 负责 milestone、约束和后台监督。配置只有这两个模型角色，不启用独立目标审核模型或强模型执行接管。

核心源码来自冻结提交 `f0c490f1e914027205044003b4e0c50eb444bed4`，34 个源码文件保持原样。它与仓库根目录的 27B 执行系统分别安装、分别运行。

- [架构与上下文](docs/ARCHITECTURE.md)
- [已有 LIBERO 结果](docs/RESULTS.md)
- [固定 30 题清单](configs/fixed30-panel.json)
- [模式开关与预算](configs/reference.json)
- [源码来源与发布改动](docs/PROVENANCE.md)

## 安装和离线测试

以下命令均在 `r27/` 目录运行。推荐 Python 3.11，仿真评测使用 Linux；离线测试不需要模型服务或 LIBERO。

```bash
cd r27
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python scripts/verify_source.py
.venv/bin/python -m pytest -q
```

原始测试包含已关闭机制的回归覆盖；测试文件存在不表示参考配置启用了这些机制。两个 ETA worker 生命周期集成测试在缺少外部 ETA 环境时跳过。

## 安装 LIBERO

直接适配器使用上游 LIBERO 的任务、初态和资源，不需要启动 ETA MCP 服务。下面固定到实验机器已核对的 LIBERO 提交；第三方代码与资产由上游提供，不打包到本仓库。

```bash
.venv/bin/python -m pip install -e '.[libero,test]'
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git vendor/LIBERO
git -C vendor/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
.venv/bin/python -m pip install --no-deps -e vendor/LIBERO

export LIBERO_CONFIG_PATH="$PWD/vendor/libero-config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
.venv/bin/python -c 'from libero.libero import get_libero_path; print(get_libero_path("init_states"))'
```

最后一步首次运行会询问数据目录，可选择 `N` 使用默认位置。保留上游 BDDL、资产及官方初态文件；本项目不需要训练轨迹数据集。EGL 需要机器提供可用的 OpenGL 驱动；上述 PyTorch 环境变量用于加载可信的官方初态文件。物理依赖固定了 MuJoCo 3.2.7、robosuite 1.4.1 和 bddl 1.0.1，其余 Python 依赖并非完整锁定环境。

## 配置模型

```bash
cp .model-env.example .model-env
chmod 600 .model-env
```

在本地填写两个支持多图输入、OpenAI Chat Completions 格式的端点和密钥。`.model-env` 不提交到 Git。文件采用 `KEY=value`，不要加 shell 引号或 `export`；已存在的进程环境变量优先。

历史实验使用 Qwen3.6-35B-A3B 作为 Fast，kimi-k3.1-pro 作为 Slow。示例保留了当时的模型选项；替换服务或模型时，应确认对应选项是否受支持，并作为新的实验配置记录。

## 运行一道题

新增入口只负责组装原有命令。默认预览，不读取密钥、不调用模型、不启动仿真：

```bash
.venv/bin/python scripts/run_reference.py \
  --suite libero_object --task 3 --initial-state 0 \
  --mode dual_async --output runs/object3-state0-async
```

确认配置后加 `--execute` 执行。分别选择 `single_fast`、`single_strong`、`dual_async`，即可对同一任务和初态比较三种模式；每次必须指定新的输出目录。

```bash
.venv/bin/python scripts/run_reference.py \
  --suite libero_object --task 3 --initial-state 0 \
  --mode dual_async --output runs/object3-state0-async --execute
```

任务语言从随包的官方任务目录读取。`--initial-state` 指官方初态索引，不是任意随机种子。标准任务预算为 24 次决策、1200 秒；`libero_10` 长任务为 36 次决策、1800 秒。

输出包含 `result.json`、`config.json`、`events.sqlite`、模型请求记录、当前帧及物理初始化记录。任务成功以官方 LIBERO 检查为准，不能用模型的完成声明或技能回执替代。输出可能包含任务内容和模型回复，不应直接作为公开仓库素材。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/embodied_harness/agents` | 模型客户端、Fast 执行、Slow 规划监督及提示 |
| `src/embodied_harness/runtime` | 上下文、milestone、异步调度、消息送达、事件记录和门禁 |
| `src/embodied_harness/adapters` | LIBERO RGB-D 观测、候选测量、抓取/放置等技能；也保留历史 ETA 适配器 |
| `src/embodied_harness/domain` | 状态、动作、回执与接口类型 |
| `memory` | 四张历史外观参考及来源记录 |
| `configs` | 参考配置、30 题清单和旧测试使用的任务目录 |
| `tests` | 原始回归测试及公开运行入口测试 |
| `scripts` | 运行预览、源码核验、直接仿真检查和测试依赖的历史 runner |

`scripts/run_round22_targeted.py` 保留用于原始离线测试；它依赖未发布的历史调度环境，不是本分支推荐的运行入口。请使用 `run_reference.py`。本次发布整理和离线验证不等于重新完成一轮付费模型/仿真实验。
