"""Run one episode in the simulation runtime from the frozen LIBERO evaluation."""

import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/tests/inference/libero10-text-compact-20261119")
repo = Path(__file__).resolve().parents[1]
stage = json.loads((ROOT / "plan.json").read_text())["stages"]["libero"]
env = os.environ.copy()
env.update(stage["environment"])
paths = [str(repo), str(ROOT / "source"), stage["physics_root"], env.get("PYTHONPATH", "")]
env["PYTHONPATH"] = os.pathsep.join(filter(None, paths))
entry = repo / "eta0_run.py"
args = sys.argv[1:]
if args[:1] == ["--check-upper"]:
    entry = repo / "deploy/check_upper.py"
    args = args[1:]
os.execve(stage["python"], [stage["python"], str(entry), *args], env)
