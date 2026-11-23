#!/usr/bin/env python3
"""Preview one R27 episode; --execute runs the frozen two-role configuration."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("single_fast", "single_strong", "dual_async")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def command_for(args):
    config = json.loads((ROOT / "configs/reference.json").read_text())
    catalog = json.loads((ROOT / "configs/round21-task-catalog.json").read_text())
    task = next((t for t in catalog["tasks"]
                 if t["suite"] == args.suite and t["task_index"] == args.task), None)
    if task is None:
        raise ValueError("Task is not in the official 40-task catalog")
    if not 0 <= args.initial_state < task["official_state_count"]:
        raise ValueError("Official initial-state index is out of range")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    budget = config["budgets"]["long" if args.suite == "libero_10" else "standard"]
    command = [sys.executable, "-m", "embodied_harness.app",
               "--backend", "direct_libero", "--mode", args.mode,
               "--hierarchy", "milestones", "--no-identity-check",
               "--env-id", task["env_id"], "--mission", task["mission"],
               "--seed", str(args.initial_state), "--output", str(output),
               "--model-env", str(Path(args.model_env).expanduser().resolve()),
               "--memory", str(ROOT / "memory/verified-experience.json"),
               "--max-decisions", str(budget["max_decisions"]),
               "--timeout", str(budget["timeout_s"]), "--review-every", "3"]
    command.extend("--" + name.replace("_", "-")
                   for name, enabled in config["feature_flags_by_mode"][args.mode].items()
                   if enabled)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--mode", choices=MODES, default="dual_async")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-env", default=str(ROOT / ".model-env"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        command = command_for(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.execute:
        print(shlex.join(command))
        return 0
    if not Path(args.model_env).expanduser().is_file():
        parser.error("Create a local model-env file before execution")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
