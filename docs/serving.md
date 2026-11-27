# 27B 模型部署指南（标准 vLLM & DFlash2）

`deploy/serve_vllm.py` 是面向外部与本地设备的通用启动器：无内网路径硬编码，自动探测本地 NVIDIA GPU 数量与显存、选择合适的张量并行度（TP）和上下文长度、执行 512×512 视觉 Canary 验收，并生成供 `deploy/first_run.py` 直接读取的私有 `client.json`。

## HuggingFace 模型权重（发布占位符）

| 组件 | 作用 | HuggingFace 仓库（发布前替换为正式链接） |
|---|---|---|
| **ETA0-27B 主模型** | 自主观察、规划、进度查询与动作执行 | `https://huggingface.co/OpenETA7/ETA0-27B-PLACEHOLDER` |
| **ETA0-27B-DFlash2 草稿模型** | DFlash2 投机解码加速权重（可选） | `https://huggingface.co/OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER` |

## 两种部署方案

| 方案 | 适用场景 | 特点 |
|---|---|---|
| **方案 1：标准 vLLM** | 任意安装官方 vLLM 的 NVIDIA GPU 机器 | 零定制依赖，兼容性最广，支持单卡与多卡 TP |
| **方案 2：vLLM + DFlash2 投机解码** | 需要本地低延迟高速推理（含**单卡 RTX 5090 开箱部署**） | 同步加载 DFlash2 draft 权重（默认 `num_speculative_tokens=7`），大幅提升动作生成吞吐与单步响应速度 |

## 常见硬件配置建议

`deploy/serve_vllm.py` 会按显存自动给出默认值（均可通过命令行覆盖）：

| 硬件配置 | 推荐方案 | 自动 / 建议参数 | 说明 |
|---|---|---|---|
| **单卡 RTX 5090 (32 GB)** | **DFlash2** 或标准 vLLM | `TP=1`，`--max-model-len 65536`，`--max-num-seqs 1`（搭配量化权重或 `--quantization fp8`） | 单卡本地高速开箱部署；`eta0_run.py` 会自动将历史压缩阈值适配到 64K |
| **单卡 A100 / H100 / H800 (80 GB)** | DFlash2 或标准 vLLM | `TP=1`，`--max-model-len 131072`（BF16） | 单卡 BF16 直接运行 |
| **双卡 40–80 GB（如 2×A100 / 2×H100 / 2×RTX 5090）** | DFlash2 或标准 vLLM | `TP=2`，总显存 ≥120 GB 时默认 `262144`，否则 `131072` | 2×80 GB 对应论文/评测冻结的 262K 全上下文配置 |
| **已有远程/云端 OpenAI 兼容端点** | 无需本地起服务 | 手动写 `client.json`（含 `base_url`、`model`、`api_key`） | 若服务端无 `/tokenize` 或不接受 vLLM 扩展字段，可加 `"api_style": "openai"` |

## 快速启动命令

先用 `--dry-run` 预览硬件探测结果与即将执行的 vLLM 命令（无需加载权重）：

```bash
python3 deploy/serve_vllm.py \
  --model OpenETA7/ETA0-27B-PLACEHOLDER \
  --dry-run
```

### 1. 标准 vLLM 启动

```bash
python3 deploy/serve_vllm.py \
  --model OpenETA7/ETA0-27B-PLACEHOLDER \
  --port 18680
```

### 2. DFlash2 投机解码启动（含单卡 RTX 5090 本地高速部署）

```bash
python3 deploy/serve_vllm.py \
  --model OpenETA7/ETA0-27B-PLACEHOLDER \
  --draft OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER \
  --port 18680
```

> 单卡 32 GB（如 RTX 5090）若使用 BF16 权重出现显存不足，可改用 FP8 / 量化版本权重或追加 `--quantization fp8 --max-model-len 65536 --max-num-seqs 1`。
>
> DFlash2 模式依赖支持 `method="dflash"` 投机解码的 vLLM 运行时；若当前环境的官方 vLLM 尚未包含 DFlash2 支持，可去掉 `--draft` 立即回退到标准 vLLM 启动。

### 3. 启动后对接 `first_run.py` 与 GUI Gate

服务通过 512×512 红图视觉验收后，终端会输出 `ETA0_READY ...` 与 `export ETA0_EXECUTOR_CLIENT=...`。在另一个终端执行：

```bash
export ETA0_EXECUTOR_CLIENT=~/.cache/eta0-system/serve-18680/client.json
python3 deploy/first_run.py --gate 8765
```

若需要跨机器访问 27B 服务，启动时追加 `--host 0.0.0.0`，并将 `client.json` 复制到运行仿真的机器（或通过 SSH 端口转发访问 `127.0.0.1:18680`）。

## 关键参数说明

- `--python /path/to/vllm-venv/bin/python`：指定已安装 `vllm` 的解释器路径（默认使用当前 `python3`）。
- `--client-file /path/to/client.json`：自定义输出的私有连接文件路径（权限自动设为 `0600`）。
- `--num-speculative-tokens 7`：DFlash2 每次投机步数，默认与评测配置一致为 `7`。
- `--extra-vllm-arg ...`：可重复传入，透传任意额外 vLLM 参数。
- `--skip-canary`：跳过启动时的红图视觉 Canary 检查（仅调试用，正式部署建议保留）。
