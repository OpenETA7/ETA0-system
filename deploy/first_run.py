"""Deploy the request harness and run one natively scored LIBERO episode."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_cards import SkillLibrary, load_profile  # noqa: E402

REV = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CATALOG_PATH = ROOT / "eta0_sim/libero_catalog.json"


def load_catalog():
    return json.loads(CATALOG_PATH.read_text())["tasks"]


def resolve_task(suite, task_id, initial_index):
    tasks = load_catalog()
    suites = sorted({t["suite"] for t in tasks})
    if suite not in suites:
        raise ValueError(f"unknown suite {suite!r}; choose from {', '.join(suites)}")
    match = next((t for t in tasks if t["suite"] == suite and t["task"] == task_id), None)
    if match is None:
        max_task = max(t["task"] for t in tasks if t["suite"] == suite)
        raise ValueError(f"task {task_id} out of range for {suite} (0..{max_task})")
    limit = match["official_state_count"]
    if not 0 <= initial_index < limit:
        raise ValueError(
            f"initial-index {initial_index} out of range for {suite} task {task_id} "
            f"(0..{limit - 1})"
        )
    return match


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
    p.add_argument("--profile", type=Path, help="declarative JSON run profile")
    p.add_argument("--output", type=Path)
    p.add_argument("--gate", type=int, metavar="PORT", help="open a local review gate")
    p.add_argument("--suite")
    p.add_argument("--task", type=int)
    p.add_argument("--initial-index", type=int)
    p.add_argument("--instruction")
    p.add_argument("--skills-dir", type=Path, help="directory of read-only Markdown skill cards")
    p.add_argument("--skill", action="append", default=[], help="read-only skill card ID to attach")
    p.add_argument("--list-tasks", action="store_true", help="print built-in LIBERO task catalog")
    p.add_argument("--dry-run", action="store_true", help="preview resolved command without run")
    p.add_argument("--reinstall", action="store_true", help="refresh cached simulation packages")
    args = p.parse_args()

    if args.list_tasks:
        tasks = load_catalog()
        if args.suite:
            tasks = [t for t in tasks if t["suite"] == args.suite]
            if not tasks:
                p.error(f"unknown suite: {args.suite}")
        for item in tasks:
            print(
                f"{item['suite']:14s}  task={item['task']:2d}  "
                f"states=0..{item['official_state_count'] - 1:<2d}  {item['goal']}"
            )
        return

    profile = {}
    if args.profile is not None:
        try:
            profile = load_profile(args.profile, base_dir=ROOT)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            p.error(f"invalid profile {args.profile}: {exc}")

    suite = args.suite if args.suite is not None else profile.get("suite", "libero_object")
    task_id = args.task if args.task is not None else profile.get("task", 0)
    initial_index = (
        args.initial_index if args.initial_index is not None else profile.get("initial_index", 5)
    )
    instruction = args.instruction if args.instruction is not None else profile.get("instruction")
    gate_port = args.gate if args.gate is not None else profile.get("gate")
    skills_dir = args.skills_dir or (
        Path(profile["skills_dir"]) if profile.get("skills_dir") else None
    )
    skill_ids = list(args.skill) if args.skill else list(profile.get("skill_ids", ()))
    if skill_ids and skills_dir is None:
        default_skills = ROOT / "skills"
        if default_skills.is_dir():
            skills_dir = default_skills
        else:
            p.error("--skill requires --skills-dir")
    if skills_dir is not None:
        try:
            skill_lib = SkillLibrary(skills_dir)
            for sid in skill_ids:
                skill_lib.get(sid)
        except ValueError as exc:
            p.error(str(exc))

    try:
        catalog_entry = resolve_task(suite, task_id, initial_index)
    except ValueError as exc:
        p.error(str(exc))

    client = os.getenv("ETA0_EXECUTOR_CLIENT")
    client_ready = bool(client and Path(client).is_file())
    if not args.dry_run and not client_ready:
        p.error("set ETA0_EXECUTOR_CLIENT to your existing 27B client.json")
    upper = os.getenv("ETA0_UPPER_CONFIG")
    library = os.getenv("ETA0_DEMO_LIBRARY") or profile.get("demo_library")
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

    def build_flags(libero_root, client_path):
        flags = [
            "--mode",
            "request",
            "--suite",
            suite,
            "--task",
            str(task_id),
            "--initial-index",
            str(initial_index),
            "--libero-source-root",
            str(libero_root),
            "--executor-client",
            client_path,
            "--output",
            str(output),
            "--recover-length",
        ]
        for key, flag in (
            ("max_attempts", "--max-attempts"),
            ("max_actions", "--max-actions"),
            ("max_calls", "--max-calls"),
            ("wall_seconds", "--wall-seconds"),
        ):
            if key in profile:
                flags.extend([flag, str(profile[key])])
        if upper:
            flags.extend(["--boyue-config", upper])
        if not use_upper:
            flags.extend(["--no-clarify", "--no-rewrite", "--no-retrieve"])
        if library:
            flags.extend(["--demo-library", library])
        if skills_dir is not None:
            flags.extend(["--skills-dir", str(skills_dir)])
            for sid in skill_ids:
                flags.extend(["--skill", sid])
        if gate_port is not None:
            flags.extend(["--gate", str(gate_port)])
        elif use_upper and sys.stdin.isatty():
            flags.append("--interactive")
        if instruction:
            flags.extend(["--instruction", instruction])
        return flags

    if args.dry_run:
        preview_root = (
            Path(os.getenv("ETA0_LIBERO_ROOT")).expanduser()
            if os.getenv("ETA0_LIBERO_ROOT")
            else cache / ("LIBERO-" + REV[:10])
        )
        preview_python = os.getenv("ETA0_SIM_PYTHON") or str(cache / "sim-py311/bin/python")
        preview_flags = build_flags(preview_root, client or "<set ETA0_EXECUTOR_CLIENT>")
        plan = dict(
            dry_run=True,
            profile=profile.get("id") or (str(args.profile) if args.profile else None),
            suite=suite,
            task=task_id,
            initial_index=initial_index,
            native_goal=catalog_entry["goal"],
            instruction=instruction or catalog_entry["goal"],
            executor_client_ready=client_ready,
            upper_enabled=use_upper,
            demo_library=library,
            skills_dir=str(skills_dir) if skills_dir else None,
            skill_ids=skill_ids,
            gate=gate_port,
            output=str(output),
            command=[preview_python, str(ROOT / "eta0_run.py"), *preview_flags],
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

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
        stamp = venv / ".eta0-installed"
        fingerprint = hashlib.sha256(
            (str(ROOT) + "\n").encode() + (ROOT / "pyproject.toml").read_bytes()
        ).hexdigest()
        if not Path(python).is_file():
            subprocess.run([uv, "venv", "--python", "3.11", str(venv)], check=True)
        if args.reinstall or not stamp.is_file() or stamp.read_text().strip() != fingerprint:
            install = [uv, "pip", "install", "--python", python]
            if sys.platform.startswith("linux"):
                install += ["--torch-backend", "cpu"]
            subprocess.run([*install, f"{ROOT}[simulation,libero]"], check=True)
            stamp.write_text(fingerprint + "\n")
    env["ETA0_SIM_PYTHON"] = python
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "osmesa")
        env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
        env.setdefault("PYNPUT_BACKEND", "dummy")
    flags = build_flags(root, client)
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
