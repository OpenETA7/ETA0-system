"""Run and score the held-out four-suite request-harness comparison on existing replicas."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from statistics import mean

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODEL = "kimi-k3.1-pro"
EXECUTOR = os.environ.get("ETA0_SERVED_MODEL", "eta0-27b")
ARCHIVE = Path(os.environ["ETA0_DEMO_ARCHIVE"]) if os.environ.get("ETA0_DEMO_ARCHIVE") else None
ARMS = ("baseline", "dual")


def _sim_python():
    plan_path = os.environ.get("ETA0_RUNTIME_PLAN")
    if plan_path and Path(plan_path).is_file():
        return read(Path(plan_path))["stages"]["libero"]["python"]
    return os.environ.get("ETA0_SIM_PYTHON") or sys.executable


def read(path):
    return json.loads(path.read_text())


def score(pair):
    base, dual = [pair[a] for a in ARMS]
    for row in (base, dual):
        if row.get("status") != "scored":
            return "incomplete"
    if (
        not base["identity"]
        or base["identity"] != dual["identity"]
        or base.get("replica") != dual.get("replica")
    ):
        return "mismatch"
    return {
        (False, True): "win",
        (True, False): "loss",
        (True, True): "both_success",
        (False, False): "both_failed",
    }[base["success"], dual["success"]]


def cases():
    return [(suite, task) for task in range(10) for suite in SUITES]


def terminal_pairs(pairs):
    return [
        p
        for p in pairs
        if all(type(p[a].get("success")) is bool for a in ARMS)
        and p["baseline"].get("identity")
        and p["baseline"]["identity"] == p["dual"].get("identity")
        and p["baseline"].get("replica") is not None
        and p["baseline"].get("replica") == p["dual"].get("replica")
    ]


def report(output):
    pairs = []
    for suite, task in cases():
        pair = {"suite": suite, "task": task, "initial": 5}
        for arm in ARMS:
            file = output / suite / f"task-{task:02d}" / f"{arm}.json"
            pair[arm] = read(file) if file.is_file() else {"status": "pending"}
        pair["outcome"] = score(pair)
        pairs.append(pair)
    matched = [p for p in pairs if p["outcome"] not in {"incomplete", "mismatch"}]
    summary = {}
    for arm in ARMS:
        rows = [p[arm] for p in matched]
        summary[arm] = {
            k: mean(row[k] for row in rows) if rows else None
            for k in (
                "success",
                "first_attempt_success",
                "elapsed_s",
                "lower_wall_s",
                "upper_wall_s",
                "calls",
                "upper_calls",
                "completion_queries",
                "attempts",
                "selected_demos",
            )
        }
    return dict(
        matched=len(matched), terminal_pairs=len(terminal_pairs(pairs)), means=summary, pairs=pairs
    )


def render_report(result):
    lines = [
        "# 请求级双系统与 27B 配对结果",
        "",
        f"两组均已取得评分并通过配对校验的任务有 {result['matched']}/{len(result['pairs'])} 个。"
        "每项任务使用初态索引 5。",
        "",
        "| 配置 | 首轮成功 | 最终成功 | 平均总分钟 | 平均 27B 分钟 | 平均上层分钟 | 平均进度查询 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    comparable = [p for p in result["pairs"] if p["outcome"] not in ("incomplete", "mismatch")]
    for arm, label in zip(ARMS, ("27B", "Kimi + 27B"), strict=True):
        rows = [p[arm] for p in comparable]
        n = len(rows)
        if n:
            means = result["means"][arm]
            lines.append(
                f"| {label} | {sum(r['first_attempt_success'] for r in rows)}/{n} | "
                f"{sum(r['success'] for r in rows)}/{n} | {means['elapsed_s'] / 60:.1f} | "
                f"{means['lower_wall_s'] / 60:.1f} | {means['upper_wall_s'] / 60:.1f} | "
                f"{means['completion_queries']:.2f} |"
            )
    terminal = terminal_pairs(result["pairs"])
    if terminal:
        lines.extend(["", f"另按完整用户请求统计，已有 {len(terminal)} 对返回终态且配对一致。"])
        for arm, label in zip(ARMS, ("27B", "Kimi + 27B"), strict=True):
            rows = [p[arm] for p in terminal]
            failures = sum(r.get("reason") == "preparation_error" for r in rows)
            lines.append(
                f"{label} 完成 {sum(r['success'] for r in rows)}/{len(rows)}，"
                f"其中准备阶段失败 {failures} 次。"
            )
            fallbacks = [r for r in rows if r.get("preparation_fallbacks")]
            if fallbacks:
                lines.append(
                    f"{label} 有 {len(fallbacks)} 个请求启用过兜底，"
                    f"其中 {sum(r['success'] for r in fallbacks)} 个最终完成。"
                )
        lines.extend(
            [
                "",
                "| 完整请求 | 首轮成功 | 平均总分钟 | 平均 27B 分钟 | 平均上层分钟 | "
                "27B 调用总数 | 上层逻辑调用总数 | 实际进度查询总数 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm, label in zip(ARMS, ("27B", "Kimi + 27B"), strict=True):
            rows = [p[arm] for p in terminal]
            lines.append(
                f"| {label} | {sum(r['first_attempt_success'] for r in rows)}/{len(rows)} | "
                f"{mean(r['elapsed_s'] for r in rows) / 60:.1f} | "
                f"{mean(r['lower_wall_s'] for r in rows) / 60:.1f} | "
                f"{mean(r['upper_wall_s'] for r in rows) / 60:.1f} | "
                f"{sum(r['calls'] for r in rows)} | {sum(r['upper_calls'] for r in rows)} | "
                f"{sum(r['completion_queries'] for r in rows)} |"
            )
        for label, flags in (("转胜", (False, True)), ("退化", (True, False))):
            changed = [
                f"{p['suite']}/{p['task']:02d}"
                for p in terminal
                if tuple(p[a]["success"] for a in ARMS) == flags
            ]
            lines.append(
                f"\n{label} {len(changed)} 项"
                + ("，" + "、".join(changed) if changed else "")
                + "。"
            )
        both = [p for p in terminal if all(p[a]["success"] for a in ARMS)]
        if both:
            base_s, dual_s = (mean(p[a]["elapsed_s"] for p in both) for a in ARMS)
            lines.append(
                f"\n两组均成功的 {len(both)} 个初态中，27B 平均用时 {base_s / 60:.1f} 分钟，"
                f"双系统 {dual_s / 60:.1f} 分钟，"
                f"每个请求平均相差 {(dual_s - base_s) / 60:+.1f} 分钟。"
            )
    lines += [
        "",
        "| 任务 | 配对结果 | 27B / 双系统分钟 | 实际查询 | 双系统示例采用轮数 |",
        "|---|---|---:|---:|---:|",
    ]
    labels = dict(
        win="转胜",
        loss="退化",
        both_success="均成功",
        both_failed="均未完成",
        incomplete="未齐或未评分",
        mismatch="配对校验失败",
    )
    for pair in result["pairs"]:
        base, dual = (pair[a] for a in ARMS)
        name = base.get("task_name") or dual.get("task_name") or ""
        case = f"{pair['suite']} / {pair['task']:02d} {name}"
        label = labels[pair["outcome"]]
        if pair in terminal and pair["outcome"] == "incomplete":
            reasons = [p.get("reason", "unknown") for p in (base, dual)]
            lines.append(
                f"| {case} | {reasons[0]} / {reasons[1]} | "
                f"{base['elapsed_s'] / 60:.1f} / {dual['elapsed_s'] / 60:.1f} | "
                f"{base['completion_queries']} / {dual['completion_queries']} | "
                f"{dual['selected_demos']} |"
            )
        elif pair["outcome"] in ("mismatch", "incomplete"):
            lines.append(f"| {case} | {label} | — | — | — |")
        else:
            lines.append(
                f"| {case} | {label} | {base['elapsed_s'] / 60:.1f} / "
                f"{dual['elapsed_s'] / 60:.1f} | {base['completion_queries']} / "
                f"{dual['completion_queries']} | {dual['selected_demos']} |"
            )
    return "\n".join(lines) + "\n"


def read_outcome(out, lib, index, suite, task, arm, upper_model):
    result = read(out / "rollout/RESULT.json")
    native, config = read(out / "simulation/task.json"), read(out / "run.json")
    if (
        (native["suite"], native["task_id"], native["initial_index"], native["seed"])
        != (suite, task, 5, 1234)
        or config["executor"] != EXECUTOR
        or config["upper"] != (upper_model if arm == "dual" else None)
    ):
        raise ValueError("model or case mismatch")
    identity = [
        native["initial_state_sha256"],
        native["bddl_sha256"],
        result["initial_external_rgb_sha256"],
        config["pack_sha256"],
    ]
    if not all(isinstance(v, str) and len(v) == 64 for v in identity):
        raise ValueError("missing pair evidence")
    if arm == "dual" and result["library_sha256"] != hashlib.sha256(lib.read_bytes()).hexdigest():
        raise ValueError("frozen library differs")
    events = [json.loads(line) for line in (out / "rollout/events.jsonl").read_text().splitlines()]
    row = dict(
        status="scored"
        if result["reason"]
        in {
            "native_success",
            "action_budget",
            "call_budget",
            "wall_time_budget",
            "needs_clarification",
        }
        else "unscored",
        identity=identity,
        task_name=native["task_name"],
        library_sha256=result["library_sha256"],
        replica=index + 1,
        reason=result["reason"],
        attempts=len(result["attempts"]),
        selected_demos=sum(e["event"] == "attempt_started" and bool(e["demo_id"]) for e in events),
        **{
            k: result[k]
            for k in (
                "success",
                "first_attempt_success",
                "elapsed_s",
                "lower_wall_s",
                "upper_wall_s",
                "upper_calls",
                "calls",
                "completion_queries",
            )
        },
    )
    if "preparation_fallbacks" in result:
        row["preparation_fallbacks"] = result["preparation_fallbacks"]
    return row


def launch(args, lib, index, suite, task, arm):
    folder = args.target / suite / f"task-{task:02d}"
    folder.mkdir(exist_ok=True, parents=True)
    out = folder / arm
    scorefile = folder / f"{arm}.json"
    if args.resume_from:
        try:
            existing = read_outcome(out, lib, index, suite, task, arm, args.upper_model)
        except (OSError, ValueError, TypeError, KeyError):
            existing = None
        try:
            saved = read(scorefile)
        except (FileNotFoundError, json.JSONDecodeError):
            saved = None
        # An upper preparation failure is a terminal request with its own outcome.
        # Preserve it on preemption just like a natively scored request.
        if existing is not None:
            if saved is None:
                write_score(scorefile, existing)
            elif saved != existing:
                raise ValueError("scored record differs from native result")
            return
        if saved and saved.get("status") == "scored":
            raise ValueError("scored record lacks matching native result")
        archive = args.output / "interrupted" / suite / folder.name
        for partial in (out, folder / f"{arm}.log", scorefile):
            if partial.exists():
                archive.mkdir(parents=True, exist_ok=True)
                partial.rename(archive / partial.name)
    # A provider-wide Retry-After delays admission, before any episode clock starts.
    hold = args.output.parent / "preparation-mailbox/next-admission.json"
    while hold.is_file():
        remaining = read(hold)["epoch"] - time.time()
        if remaining <= 0:
            break
        time.sleep(min(30, remaining))
    cmd = [
        sys.executable,
        str(args.target.parent / "source/deploy/run_libero.py"),
        "--mode",
        "request",
        "--executor-client",
        str(args.remote_client[index % len(args.remote_client)]),
        "--suite",
        suite,
        "--task",
        str(task),
        "--initial-index",
        "5",
        "--seed",
        "1234",
        "--instruction-mode",
        "revised",
        "--completion-query",
        "--recover-length",
        "--max-attempts",
        "2",
        "--wall-seconds",
        "2700",
        "--max-actions",
        "120",
        "--max-calls",
        "360",
        "--upper-wall-seconds",
        "900",
        "--output",
        str(out),
    ]
    libero_root = os.environ.get("ETA0_LIBERO_ROOT") or (
        str(ARCHIVE / "LIBERO") if ARCHIVE is not None else None
    )
    if libero_root:
        cmd.extend(["--libero-source-root", libero_root])
    if arm == "dual":
        cmd += [
            "--clarify",
            "--rewrite",
            "--retrieve",
            "--demo-library",
            str(lib),
            "--boyue-config",
            str(args.boyue_config),
        ]
    else:
        cmd += ["--no-clarify", "--no-rewrite", "--no-retrieve"]
    with (folder / f"{arm}.log").open("x") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=6600)
            if proc.returncode:
                raise RuntimeError(f"episode exit {proc.returncode}")
            row = read_outcome(out, lib, index, suite, task, arm, args.upper_model)
        except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
            row = dict(status="unscored", error_type=type(exc).__name__)
    write_score(scorefile, row)


def write_score(path, row):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(row, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--report", action="store_true")
    p.add_argument("--markdown", action="store_true", help="render --report as a task table")
    p.add_argument("--upper-model", default=MODEL)
    p.add_argument("--boyue-config", type=Path)
    p.add_argument("--remote-client", type=Path, action="append")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.report:
        data = report(args.output)
        print(render_report(data) if args.markdown else json.dumps(data, indent=2))
        return
    if not args.remote_client or not args.boyue_config:
        p.error("a private profile and one or more existing executor clients are required")
    profile = read(args.boyue_config)
    profile = profile.get("boyue", profile)
    if profile.get("model") != args.upper_model or profile.get("allowed_returned_models") != [
        args.upper_model
    ]:
        p.error("the private profile must pin the requested upper model and returned-model ID")
    args.upper_base_url = profile["base_url"]
    args.upper_transport = "mac_relay" if "preparation_mailbox" in profile else "direct"
    args.output.mkdir(parents=True, exist_ok=False)
    args.target = args.resume_from or args.output
    with (args.target / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


def check_relay(profile_path, receipt):
    """Exercise the real two-view preparation contract from the CPU simulation zone."""
    import time
    from types import SimpleNamespace

    from PIL import Image

    from eta0_run import upper_endpoint
    from executor_led import EpisodeBudget
    from request_harness import prepare

    episode = ARCHIVE / "batch/initial-00/off/libero_goal/task-00/base"
    with (episode / "rollout/events.jsonl").open() as stream:
        ctx = next(
            event["context"]
            for line in stream
            if (event := json.loads(line))["event"] == "observation"
        )
    view = episode / "rollout/observation-000"
    client = upper_endpoint(profile_path, time.monotonic() + 160)

    def chat(messages, *, deadline, **options):
        client.deadline = deadline
        return client.chat(messages, **options)

    with Image.open(view / "external.png") as external, Image.open(view / "wrist_raw.png") as wrist:
        obs = SimpleNamespace(
            external_rgb=external,
            wrist_rgb=wrist,
            robot=SimpleNamespace(reference_frame=ctx["reference_frame"]),
        )
        budget = EpisodeBudget(2700, 160)
        instruction, _, _ = prepare(
            chat,
            obs,
            ctx["task"],
            [],
            None,
            None,
            [],
            rewrite=True,
            retrieve=False,
            clarify=False,
            ask=None,
            budget=budget,
            log=lambda *a, **k: None,
        )
    write_score(
        receipt,
        dict(
            model=client.model,
            calls=client.chat_calls,
            upper_s=budget.upper_elapsed,
            usable=bool(instruction),
            transport="mac_relay",
        ),
    )


def execute(args):
    source = args.target.parent / "source"
    if args.upper_transport == "mac_relay":
        subprocess.run(
            [
                _sim_python(),
                "-c",
                "import sys; from pathlib import Path; "
                "from deploy.request_comparison import check_relay; "
                "check_relay(Path(sys.argv[1]), Path(sys.argv[2]))",
                str(args.boyue_config),
                str(args.output / "upper-preflight.json"),
            ],
            cwd=source,
            check=True,
        )
    if args.resume_from:
        manifest = read(args.target / "manifest.json")
        expected = dict(
            upper=args.upper_model,
            executor=EXECUTOR,
            seed=1234,
            initial=5,
            cases=[list(c) for c in cases()],
        )
        if "upper_base_url" in manifest:
            expected["upper_base_url"] = args.upper_base_url
        if "upper_transport" in manifest:
            expected["upper_transport"] = args.upper_transport
        if any(manifest[k] != v for k, v in expected.items()):
            raise ValueError("preempted experiment uses different settings")
        for name, digest in manifest["sources"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("frozen executor source changed")
        lib = args.target / "demos/library.json"
        if hashlib.sha256(lib.read_bytes()).hexdigest() != manifest["library_sha256"]:
            raise ValueError("demo library changed before resume")
    else:
        if ARCHIVE is None:
            raise ValueError("set ETA0_DEMO_ARCHIVE to the baseline archive directory")
        python = _sim_python()
        demos = []
        for suite, task in cases():
            for initial in range(5):
                archive = (
                    ARCHIVE
                    / f"batch/initial-{initial:02d}/off"
                    / suite
                    / f"task-{task:02d}"
                    / "base"
                )
                result = read(archive / "rollout/RESULT.json")
                if result["success"] and result["reason"] == "native_success" and result["actions"]:
                    demos.append(archive)
                    break
        lib = args.output / "demos"
        subprocess.run(
            [
                python,
                str(source / "deploy/build_demo_library.py"),
                "--output",
                str(lib),
                *map(str, demos),
            ],
            check=True,
        )
        lib = lib / "library.json"
    # Place the pinned OpenETA package ahead of site packages in the simulation runtime.
    pack = source / "openeta_bench"
    if not pack.exists():
        pack.symlink_to(ARCHIVE / "source/openeta_bench", target_is_directory=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if not args.resume_from:
        manifest = dict(
            suites=SUITES,
            cases=cases(),
            initial=5,
            seed=1234,
            demo_sources=[str(path.relative_to(ARCHIVE)) for path in demos],
            library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
            upper=args.upper_model,
            upper_base_url=args.upper_base_url,
            upper_transport=args.upper_transport,
            executor=EXECUTOR,
            sources={
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(source.glob("*.py"))
            },
        )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    route_file = args.target / "replica-clients.json"
    routes = [str(p) for p in args.remote_client]
    if route_file.exists() and read(route_file) != routes:
        raise ValueError("resume requires the same executor replicas in the same order")
    if not route_file.exists():
        write_score(route_file, routes)
    library = read(lib)
    with ExitStack() as stack:
        pools = [stack.enter_context(ThreadPoolExecutor(max_workers=2)) for _ in args.remote_client]
        jobs = []
        for n, (suite, task) in enumerate(cases()):
            idx = (n + task) % len(pools)
            rows = [r for r in library["examples"] if (r["suite"], r["task_id"]) != (suite, task)]
            subset = lib.with_name(f"{suite}-{task:02d}.json")
            subset_data = json.dumps(dict(library, examples=rows), indent=2)
            if subset.exists():
                if subset.read_text() != subset_data:
                    raise ValueError("frozen per-task demo subset changed")
            else:
                subset.write_text(subset_data)
            for arm in ARMS if n % 2 == 0 else reversed(ARMS):
                jobs.append(pools[idx].submit(launch, args, subset, idx, suite, task, arm))
        for job in jobs:
            job.result()
    final = report(args.target)
    (args.target / "report.json").write_text(json.dumps(final, indent=2))
    (args.target / "results-zh.md").write_text(render_report(final))
    if final["terminal_pairs"] != len(cases()):
        raise SystemExit("missing or mismatched request results; see per-arm diagnostics")


if __name__ == "__main__":
    main()
