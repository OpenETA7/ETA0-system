"""Deploy the request harness and run one natively scored LIBERO episode."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
REV = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def checkout(cache):
    selected = os.getenv("ETA0_LIBERO_ROOT")
    root = Path(selected).expanduser() if selected else cache / ("LIBERO-" + REV[:10])
    if not selected and not root.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            staging = Path(temp) / "LIBERO"
            subprocess.run(["git", "init", "-q", str(staging)], check=True)
            git = ["git", "-C", str(staging)]
            url = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
            subprocess.run([*git, "fetch", "--depth=1", url, REV], check=True)
            subprocess.run([*git, "checkout", "--detach", "FETCH_HEAD"], check=True)
            staging.rename(root)
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if rev != REV:
        raise SystemExit(f"ETA0_LIBERO_ROOT requires revision {REV}, got {rev}")
    return root.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--gate", type=int, metavar="PORT", help="open a local review gate")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--initial-index", type=int, default=5)
    p.add_argument("--instruction")
    args = p.parse_args()
    client = os.getenv("ETA0_EXECUTOR_CLIENT")
    if not client or not Path(client).is_file():
        p.error("set ETA0_EXECUTOR_CLIENT to your existing 27B client.json")
    upper = os.getenv("ETA0_UPPER_CONFIG")
    library = os.getenv("ETA0_DEMO_LIBRARY")
    for path in [upper, library]:
        if path and not Path(path).is_file():
            p.error(f"configuration file missing: {path}")
    upper_vars = [os.getenv("ETA0_UPPER_" + k) for k in ("BASE_URL", "MODEL", "API_KEY")]
    if not upper and any(upper_vars) and not all(upper_vars):
        p.error("set all three ETA0_UPPER_BASE_URL, ETA0_UPPER_MODEL, ETA0_UPPER_API_KEY")
    use_upper = bool(upper or all(upper_vars))
    if library and not use_upper:
        p.error("ETA0_DEMO_LIBRARY requires an upper endpoint to select examples")
    output = args.output or ROOT / "outputs" / ("first-request-" + uuid4().hex[:12])
    output = output.resolve()
    if output.exists():
        p.error(f"select a new output directory: {output}")
    cache = Path(os.getenv("ETA0_CACHE", Path.home() / ".cache" / "eta0-system")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    root = checkout(cache)
    env = os.environ.copy()
    python = env.get("ETA0_SIM_PYTHON")
    if not python:
        uv = shutil.which("uv")
        if not uv:
            p.error("install uv or set ETA0_SIM_PYTHON to a prepared simulation interpreter")
        venv = cache / "sim-py311"
        python = str(venv / "bin/python")
        if not Path(python).is_file():
            subprocess.run([uv, "venv", "--python", "3.11", str(venv)], check=True)
        install = [uv, "pip", "install", "--python", python]
        if sys.platform.startswith("linux"):
            install += ["--torch-backend", "cpu"]
        subprocess.run([*install, f"{ROOT}[simulation,libero]"], check=True)
    env["ETA0_SIM_PYTHON"] = python
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "osmesa")
        env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
        env.setdefault("PYNPUT_BACKEND", "dummy")
    flags = [
        "--mode",
        "request",
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--initial-index",
        str(args.initial_index),
        "--libero-source-root",
        str(root),
        "--executor-client",
        client,
        "--output",
        str(output),
        "--recover-length",
    ]
    if upper:
        flags.extend(["--boyue-config", upper])
    if not use_upper:
        flags.extend(["--no-clarify", "--no-rewrite", "--no-retrieve"])
    if library:
        flags.extend(["--demo-library", library])
    if args.gate is not None:
        flags.extend(["--gate", str(args.gate)])
    elif use_upper and sys.stdin.isatty():
        flags.append("--interactive")
    if args.instruction:
        flags.extend(["--instruction", args.instruction])
    subprocess.run([python, str(ROOT / "eta0_run.py"), *flags], check=True, env=env)
    receipt = output / "rollout/RESULT.json"
    result = json.loads(receipt.read_text())
    run = json.loads((output / "run.json").read_text())
    print(
        json.dumps(
            dict(
                receipt=str(receipt),
                native_goal=run["native_goal"],
                features=run["request_features"],
                success=result["success"],
                reason=result["reason"],
                lower_calls=result["lower_chat_calls"],
                upper_calls=result["upper_calls"],
                seconds=result["episode_wall_s"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
