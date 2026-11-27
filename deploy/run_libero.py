"""Run a LIBERO episode, optionally reusing a frozen evaluation environment."""

import json
import os
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
env = os.environ.copy()
python = sys.executable
paths = [str(repo)]
plan = env.get("ETA0_RUNTIME_PLAN")
if plan:
    file = Path(plan)
    stage = json.loads(file.read_text())["stages"]["libero"]
    env.update(stage["environment"])
    paths.extend([str(file.resolve().parent / "source"), stage["physics_root"]])
    python = stage["python"]
else:
    python = env.get("ETA0_SIM_PYTHON", python)
env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
entry = repo / "eta0_run.py"
args = sys.argv[1:]
if args[:1] == ["--check-upper"]:
    entry = repo / "deploy/check_upper.py"
    args = args[1:]
os.execve(python, [python, str(entry), *args], env)
