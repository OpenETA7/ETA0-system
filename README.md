# ETA0 System

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

ETA0-27B (`fullmix step2500`) receives a complete user goal and autonomously observes, plans, queries native task progress, and executes actions within a single attempt. In the default `request` mode, an optional upper multimodal model inspects the initial dual-camera views before execution starts, asks clarifying questions when genuinely ambiguous, grounds the instruction, and optionally selects one verified example from a demonstration library. If an attempt exhausts its budget, the upper model can revise the handoff from the terminal scene, and 27B continues from the same physical environment until native benchmark success. When the upper endpoint returns empty, truncated, or invalid output, it retries up to 4 additional times within its own budget; if still unavailable, 27B continues with the original instruction and confirmed user answers with zero demos.

[27B Serving Guide (vLLM / DFlash2)](docs/serving.md) · [Request Harness & Paired Evaluation](docs/request-harness.md) · [Legacy In-Attempt Supervision](docs/executor-led.md) · [Base Harness Guide](README.base.md)

27B can autonomously call `check_task_completion` to inspect which native goal conditions the environment last observed as satisfied before choosing its next action. All tasks share the same tool protocol. To evaluate this capability alone, compare `--mode base --completion-query` against `--mode base --no-completion-query` (both 27B-only). The formal 40-task × 5-initial-state comparison is documented in [Progress Query Ablation](docs/progress-query-ablation.md).

## Model Weights & Inference Serving (HuggingFace)

| Weights | Purpose | HuggingFace Repository (placeholders prior to release) |
|---|---|---|
| **ETA0-27B** | Main embodied executor (supports standard vLLM and DFlash2) | `https://huggingface.co/OpenETA7/ETA0-27B-PLACEHOLDER` |
| **ETA0-27B-DFlash2** | Companion DFlash2 speculative decoding draft weights (**fast single-GPU RTX 5090 local deployment**) | `https://huggingface.co/OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER` |

This repository ships a portable, cluster-agnostic launcher [deploy/serve_vllm.py](deploy/serve_vllm.py) that detects local NVIDIA GPUs and VRAM, selects tensor parallelism and context length automatically, verifies multimodal inference with a 512×512 visual canary, and writes a private `client.json`:

```bash
# Option 1: Standard vLLM (auto TP and context length based on detected GPUs)
python3 deploy/serve_vllm.py --model OpenETA7/ETA0-27B-PLACEHOLDER

# Option 2: vLLM + DFlash2 speculative decoding (e.g. fast local single-GPU RTX 5090)
python3 deploy/serve_vllm.py \
  --model OpenETA7/ETA0-27B-PLACEHOLDER \
  --draft OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER
```

See the [27B Serving Guide](docs/serving.md) for single-RTX-5090 / multi-GPU profiles, quantization options, `--dry-run` inspection, and remote access.

## One-Prompt Deployment for Coding Agents

Send this prompt directly to any coding agent with terminal access (Claude Code, Codex, Cursor, Antigravity, etc.):

> Clone `OpenETA7/ETA0-system` and follow the README to deploy and verify ETA0 System on my machine: first inspect my OS, GPU/VRAM, Python, and rendering environment without assuming any specific hardware or serving stack; then based on my actual 27B setup (reusing an existing OpenAI-compatible endpoint, launching 27B locally or remotely with `deploy/serve_vllm.py` using standard vLLM or DFlash2, or verifying with the offline interactive demo if no 27B server is ready yet), ask me only for the missing required configuration, complete setup, run one end-to-end request (preferring `--gate` browser handoff review when available), and report back the detected environment, deployment path, enabled features, native benchmark goal, success flag, and receipt path.

### Coding Agent Deployment Decision Matrix (Hardware-Agnostic)

After inspecting the host (`uname`, `nvidia-smi` / Apple Silicon, Python / `uv`, network access, and existing endpoints), choose the matching path:

| Scenario | Host & Prerequisites | Recommended Path |
|---|---|---|
| **A. Existing reachable 27B service** | Any Linux / macOS machine with access to a local, LAN, or cloud 27B endpoint | Point `ETA0_EXECUTOR_CLIENT` to a private `client.json` and run `python3 deploy/first_run.py --gate 8765` |
| **B. Deploy 27B on your own GPU(s) (vLLM / DFlash2)** | Single RTX 5090, single 80 GB GPU, or multi-GPU NVIDIA workstation/server | Launch [deploy/serve_vllm.py](deploy/serve_vllm.py) (see [Serving Guide](docs/serving.md)) to auto-configure GPU settings and emit `client.json`; the runner adapts to the served context length (falling back to local token estimation if `/tokenize` is absent; add `--strict-serving-profile` to enforce pinned `262144`), then follow Path A |
| **C. No 27B server yet / laptop preview** | macOS / Linux / CPU-only machine, or verifying install & GUI flow first | Run the offline interactive workflow below: `examples/request_demo.py --gate 8765` |

### First Real LIBERO Request

Requires a reachable 27B OpenAI Chat Completions endpoint (exact token accounting via vLLM `/tokenize` when available, automatic conservative local estimation otherwise), Git, and `uv`. Offscreen rendering depends on platform:
- **Headless Linux**: defaults to Mesa offscreen rendering (`MUJOCO_GL=osmesa`; on Ubuntu install `libosmesa6` and `libgl1`), or set `MUJOCO_GL=egl` on GPU hosts.
- **macOS**: uses native OpenGL; Mesa is not required.

Run from the repository root (browse the built-in 40 LIBERO tasks with `--list-tasks` or preview the resolved command with `--dry-run`; see [examples/client.example.json](examples/client.example.json) and [examples/upper.example.json](examples/upper.example.json) for config schemas, and protect private files with `chmod 600`):

```bash
# Browse built-in tasks or preview a declarative run profile
python3 deploy/first_run.py --list-tasks --suite libero_goal
python3 deploy/first_run.py --profile configs/profiles/libero_goal_drawer.json --dry-run

# Run a real LIBERO request
export ETA0_EXECUTOR_CLIENT=/path/to/private/client.json
python3 deploy/first_run.py --profile configs/profiles/libero_object_default.json --gate 8765
```

The launcher creates a cached Python 3.11 environment, caches simulation package installation by dependency fingerprint (subsequent runs start offline without reinstalling; pass `--reinstall` to refresh), and fetches the pinned LIBERO revision. First-time setup requires access to GitHub and Python package indexes (Linux automatically uses the PyTorch CPU index). Set `ETA0_SIM_PYTHON` and `ETA0_LIBERO_ROOT` to reuse existing installations, and `ETA0_CACHE` to customize the cache root. Defaults to `libero_object` task 0, initial state 5, writing to a fresh `outputs/first-request-*` directory.

| Private Configuration | Enabled Capabilities |
|---|---|
| `ETA0_EXECUTOR_CLIENT` | Autonomous 27B execution; JSON with `base_url`, `model`, `api_key`, optional `max_model_len`, `api_style` (`"vllm"` or `"openai"`) |
| Add `ETA0_UPPER_CONFIG` (or `ETA0_UPPER_BASE_URL`/`MODEL`/`API_KEY`) | Visual upper-layer clarification & goal grounding; JSON may set `allowed_returned_models` |
| Add `ETA0_DEMO_LIBRARY` | Retrieval of verified successful demonstrations; path to `library.json` |
| Optional `--profile` / `--skills-dir` / `--skill` | Declarative task profiles & SHA-256 verified read-only Markdown skill cards (see [skills/](skills/), keeping strict Tool/Skill separation) |

**Visual GUI Gate**: Run `python3 deploy/first_run.py --gate 8765` and open the full session URL (including the `#<token>` fragment) printed in the terminal. On remote hosts, forward the port first via `ssh -N -L 8765:127.0.0.1:8765 <ssh-alias>`. The console provides a 5-stage pipeline stepper, aligned dual-camera viewports with click-to-zoom lightbox, scene telemetry, clarification thread, instruction diff badge with one-click restore, demo example toggle, **live execution camera updates with a safe stop button**, and copyable verification receipts. Omit `--gate` for direct headless execution.

**Multi-Embodiment & Real-Robot Ecosystem**: This repository focuses on the ETA0-27B request workbench and LIBERO evaluation, while [harness.py](harness.py) exposes pluggable `observation_writer`, `query_backend`, and `try_render_tcp_overlay` hooks. For dual-arm PiPER, Galaxea R1 Pro, or real Franka/DROID hardware adapters (`readback → preview → execute` 3-stage onboarding), see the sibling repository [OpenETA7/ETA-Harness](https://github.com/OpenETA7/ETA-Harness). Design rationale and historical experiments are documented in the [Nov 24 Design Proposal](docs/attempt-design-2026-11-24.md).

## Local Offline Preview

Requires Python 3.10+. The offline example pairs a scripted upper model with a two-step simulated robot (attempt 1 executes one step; attempt 2 continues from the resulting state). Inspect `RESULT.json` and `events.jsonl` in the output directory.

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python examples/request_demo.py --output outputs/first-request
# Optional: answer clarification and review handoff in the browser GUI
.venv/bin/python examples/request_demo.py --gate 8765 --output outputs/gui-demo
.venv/bin/python -m pytest -q
```

## Advanced CLI & Multi-Mode Evaluation

To invoke the LIBERO runner directly (e.g., 27B-only `base` mode or legacy `teacher` / `judge` advisory modes), pass any private `client.json` produced by [deploy/serve_vllm.py](deploy/serve_vllm.py) or your own OpenAI-compatible 27B endpoint:

```bash
export ETA0_EXECUTOR_CLIENT=/path/to/private/client.json
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode base --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-base
```

For upper multimodal models via environment variables, provide `ETA0_UPPER_BASE_URL` (ending in `/v1`), `ETA0_UPPER_MODEL`, and `ETA0_UPPER_API_KEY` (or pass `--boyue-config /path/to/upper.json` matching [examples/upper.example.json](examples/upper.example.json)). Transient upper-API failures (timeouts, disconnects, HTTP 408/429/5xx) retry up to 4 times on the exact same frames while the robot remains stationary:

```bash
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode teacher --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-teacher
```

Use `--planner`, `--supervisor`, `--teacher` (and `--no-*`) to compose legacy advisory roles, `--instruction` to override wording, and `--recover-length` to attempt one direct generation after truncated thinking. See [docs/executor-led.md](docs/executor-led.md) and [docs/libero40.md](docs/libero40.md) for simulator environment wiring and paired 40-task evaluation. Other benchmarks can integrate directly via `harness.run(benchmark, policy, ...)`.
