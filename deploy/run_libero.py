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
os.execve(stage["python"], [stage["python"], str(repo / "eta0_run.py"), *sys.argv[1:]], env)
