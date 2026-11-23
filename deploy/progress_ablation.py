"""Prepare, run or report the executor-only paired progress-query experiment."""

import argparse
import fcntl
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

if __package__:
    from . import libero40 as batch
    from .serve import MODEL, TARGET
else:
    import libero40 as batch
    from serve import MODEL, TARGET

ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "experiments/libero40-progress-states.json"
ARMS = ("off", "on")
PACK = "1325eeaca2d9eb257b788bf3732108426798a4fadb819cbb0f1a639fda7bd94f"


def manifest():
    tasks = {(t["suite"], t["task"]): t for t in json.loads(STATES.read_text())}
    pairs = []
    for initial in range(5):
        for case in batch.cases():
            t = tasks[case]
            pairs.append(
                dict(
                    suite=t["suite"],
                    task=t["task"],
                    name=t["name"],
                    initial=initial,
                    state=t["initial_state_sha256"][initial],
                    bddl=t["bddl_sha256"],
                    first=ARMS[(initial + t["task"]) % 2],
                )
            )
    return dict(
        model=MODEL,
        target=TARGET,
        libero_commit="8f1084e3132a39270c3a13ebe37270a43ece2a01",
        draft="/mnt/model/Qwen3.8-27B-DFlash2",
        mode="base",
        instruction_mode="revised",
        pack_sha256=PACK,
        seed=batch.SEED,
        wall_seconds=batch.WALL,
        max_calls=360,
        max_actions=120,
        recover_length=True,
        arms=list(ARMS),
        pairs=pairs,
    )


def locations(output, pair, arm):
    root = output / f"initial-{pair['initial']:02d}" / arm
    folder = root / pair["suite"] / f"task-{pair['task']:02d}"
    return root, folder


def load(path):
    return json.loads(path.read_text())


def read_trial(folder, p, arm):
    path = folder / "base.result.json"
    if not path.exists():
        return None, "unfinished" if folder.exists() else "pending"
    row = load(path)
    if row["returncode"] != 0 or type(row["success"]) is not bool:
        return None, "unscored"
    run = load(folder / "base/run.json")
    native = load(folder / "base/simulation/task.json")
    expected = dict(
        mode="base",
        executor=MODEL,
        upper=None,
        instruction_mode="revised",
        completion_query=arm == "on",
        progress_access="on_request" if arm == "on" else "hidden",
        roles=dict(planner=False, supervisor=False, teacher=False),
        pack_sha256=PACK,
        wall_seconds=batch.WALL,
        max_actions=120,
        max_calls=360,
        recover_length=True,
    )
    if any(run.get(k) != v for k, v in expected.items()):
        return None, "configuration_mismatch"
    if (
        native["suite"],
        native["task_id"],
        native["task_name"],
        native["initial_index"],
        native["seed"],
    ) != (p["suite"], p["task"], p["name"], p["initial"], batch.SEED):
        return None, "case_mismatch"
    if row["bddl_sha256"] != p["bddl"] or row["initial_state_sha256"] != p["state"]:
        return None, "state_or_task_hash_mismatch"
    queries = row["completion_queries"]
    if (
        type(queries) is not int
        or queries < 0
        or queries != row.get("completion_query_events")
        or (arm == "off" and queries)
        or row.get("upper_models")
        or row.get("upper_wall_s") != 0
        or not isinstance(row.get("initial_view_sha256"), str)
        or len(row["initial_view_sha256"]) != 64
        or type(row.get("replica")) is not int
    ):
        return None, "unexpected_tool_or_missing_evidence"
    if (
        not isinstance(row["elapsed_s"], (int, float))
        or not math.isfinite(row["elapsed_s"])
        or row["elapsed_s"] < 0
    ):
        return None, "invalid_time"
    return row, "scored"


def stats(pairs):
    n = len(pairs)
    wins = [i for i, a, b in pairs if b["success"] and not a["success"]]
    losses = [i for i, a, b in pairs if a["success"] and not b["success"]]
    times = [b["elapsed_s"] - a["elapsed_s"] for _, a, b in pairs]
    both = [b["elapsed_s"] - a["elapsed_s"] for _, a, b in pairs if a["success"] and b["success"]]
    summary = dict(
        pairs=n,
        wins=wins,
        losses=losses,
        both_success=len(both),
        neither=n - len(wins) - len(losses) - len(both),
        delta_pp=round(100 * (len(wins) - len(losses)) / n, 2) if n else None,
        delta_s=round(mean(times), 2) if n else None,
        median_delta_s=round(median(times), 2) if n else None,
        both_success_delta_s=round(mean(both), 2) if both else None,
    )
    for idx, arm in enumerate(ARMS, 1):
        rows = [p[idx] for p in pairs]
        success = sum(r["success"] for r in rows)
        queries = sum(r["completion_queries"] for r in rows)
        summary[arm] = dict(
            success=success,
            rate=round(success / n, 4) if n else None,
            mean_s=round(mean(r["elapsed_s"] for r in rows), 2) if n else None,
            queries=queries,
            queried_episodes=sum(r["completion_queries"] > 0 for r in rows),
            queries_per_episode=round(queries / n, 2) if n else None,
        )
    return summary


def report(output, plan):
    groups, records, all_pairs = {}, [], []
    counts = Counter()
    for p in plan["pairs"]:
        key = p["suite"], p["task"], p["name"]
        group = groups.setdefault(key, [])
        arms, statuses = {}, {}
        for arm in ARMS:
            try:
                row, status = read_trial(locations(output, p, arm)[1], p, arm)
            except (ValueError, OSError, KeyError, TypeError):
                row, status = None, "unreadable_record"
            statuses[arm] = status
            counts[status] += 1
            if row:
                arms[arm] = row
        accepted = len(arms) == 2
        if accepted and (
            arms["on"]["initial_view_sha256"] != arms["off"]["initial_view_sha256"]
            or arms["on"]["replica"] != arms["off"]["replica"]
        ):
            accepted = False
            statuses["pair"] = "view_or_replica_mismatch"
        if accepted:
            pair = p["initial"], arms["off"], arms["on"]
            group.append(pair)
            all_pairs.append((f"{key[0]}/{key[1]:02d}/{p['initial']}", *pair[1:]))
        records.append(
            dict(
                suite=key[0],
                task=key[1],
                initial=p["initial"],
                accepted=accepted,
                status=statuses,
                arms=arms,
            )
        )
    tasks = []
    for (suite, task, name), pairs in sorted(groups.items()):
        s = stats(pairs)
        direction = (
            "待齐"
            if len(pairs) < 5
            else ("改善" if s["delta_pp"] > 0 else "退化" if s["delta_pp"] < 0 else "持平")
        )
        tasks.append(dict(suite=suite, task=task, name=name, direction=direction, **s))
    overall = stats(all_pairs)
    batch.write_json(
        output / "report.json",
        dict(overall=overall, trial_status=counts, tasks=tasks, pair_details=records),
    )

    def show(v):
        return "—" if v is None else str(v)

    lines = [
        "# 27B 进度查询配对消融",
        "",
        f"已核验 {overall['pairs']}/200 对。任务方向仅在 5 对齐全后标注。",
        "",
    ]

    def table_row(label, s, direction):
        def values(field, fmt=""):
            return " / ".join(format(s[a][field], fmt) for a in ARMS) if s["pairs"] else "—"

        def ids(k):
            return ", ".join(map(str, s[k])) or "—"

        return (
            f"| {label} | {s['pairs']} | {values('rate', '.0%')} | {show(s['delta_pp'])} | "
            f"{values('mean_s')} | {show(s['delta_s'])} | {values('queries')} | "
            f"{ids('wins')} | {ids('losses')} | {direction} |"
        )

    lines += [
        "| 任务 | 有效对 | 成功率 关 / 开 | 差值 pp | 平均秒 关 / 开 | 配对平均差秒 |"
        " 查询总数 关 / 开 | 改善初态 | 退化初态 | 净方向 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for t in tasks:
        lines.append(table_row(f"{t['suite']}/{t['task']:02d} {t['name']}", t, t["direction"]))
    if all_pairs:
        lines[4:4] = [
            f"整体成功率（关 / 开）为 {overall['off']['rate']:.1%} / {overall['on']['rate']:.1%}，"
            f"差 {overall['delta_pp']:+.2f} 个百分点。"
            f"平均端到端耗时差 {overall['delta_s']:+.2f} 秒。"
            f"实际查询总数（关 / 开）为 {overall['off']['queries']} / {overall['on']['queries']}。",
            "",
        ]
        directions = [
            f"{label}任务："
            + "、".join(f"{t['suite']}/{t['task']:02d}" for t in tasks if t["direction"] == label)
            for label in ("改善", "退化")
            if any(t["direction"] == label for t in tasks)
        ]
        if directions:
            lines[6:6] = [*directions, ""]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return overall["pairs"]


def run_queue(args, plan):
    if not args.client or len(args.client) > 4 or len(set(args.client)) != len(args.client):
        raise ValueError("supply one to four distinct existing 27B clients")
    profiles = [load(c) for c in args.client]
    if len({p["base_url"] for p in profiles}) != len(profiles):
        raise ValueError("replica URLs must be distinct")
    for c, p in zip(args.client, profiles, strict=True):
        if (p["target"], p["model"], p["draft"]) != (TARGET, MODEL, plan["draft"]):
            raise ValueError("executor target, model or draft mismatch")
        batch.ready(c)
    checkout = args.libero_source_root.resolve(strict=True)
    # Only nonsecret fields are persisted. Pin scheduling and code across resumptions.
    files = [
        "harness.py",
        "eta0_run.py",
        "libero_semantics.py",
        "rollout_context.py",
        "visual_query.py",
        "tcp_overlay.py",
        "tcp_overlay_adapter.py",
        "executor_led.py",
        "deploy/run_libero.py",
        "deploy/libero40.py",
        "deploy/progress_ablation.py",
    ]
    setup = dict(
        clients=[p["base_url"] for p in profiles],
        draft=profiles[0]["draft"],
        slots_per_replica=args.slots,
        libero_root=str(checkout),
        sources={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
    )
    path = args.output / "runtime.json"
    if path.exists() and load(path) != setup:
        raise ValueError("runtime differs from the existing study")
    batch.write_json(path, setup)

    def lane(index):
        slot = index % len(profiles)
        for p in plan["pairs"][index :: len(profiles) * args.slots]:
            for arm in (p["first"], "off" if p["first"] == "on" else "on"):
                root, folder = locations(args.output, p, arm)
                if folder.exists():
                    row, status = read_trial(folder, p, arm)
                    if row:
                        continue
                    if status not in {"unfinished", "unscored"}:
                        raise ValueError(f"inspect {folder}: {status}")
                    retry = 1
                    while (folder.parent / f"{folder.name}.attempt-{retry}").exists():
                        retry += 1
                    folder.rename(folder.parent / f"{folder.name}.attempt-{retry}")
                config = SimpleNamespace(
                    output=root,
                    initial_index=p["initial"],
                    libero_source_root=str(checkout),
                    completion_query=arm == "on",
                    progress_query_ablation=True,
                )
                batch.run_episode(config, args.client[slot], slot, (p["suite"], p["task"]), "base")

    lanes = len(profiles) * args.slots
    with ThreadPoolExecutor(lanes) as pool:
        list(pool.map(lane, range(lanes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for cmd in ("plan", "run", "report"):
        sub = commands.add_parser(cmd)
        sub.add_argument("--output", type=Path, required=True)
        if cmd == "run":
            sub.add_argument("--client", action="append", type=Path, required=True)
            sub.add_argument("--libero-source-root", type=Path, required=True)
            sub.add_argument("--slots", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    plan = manifest()
    args.output.mkdir(exist_ok=True, parents=True)
    path = args.output / "manifest.json"
    with ExitStack() as stack:
        if args.command in ("plan", "run"):
            lock = stack.enter_context((args.output / ".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and load(path) != plan:
            parser.error("output directory contains a different experiment")
        if args.command == "plan":
            batch.write_json(path, plan)
            print(f"{path}: 40 tasks, 200 pairs, 400 episodes")
            return
        if not path.is_file():
            parser.error("prepare this output directory with plan first")
        if args.command == "run":
            run_queue(args, plan)
        # Serialize report writers while allowing snapshots during execution.
        lock = stack.enter_context((args.output / ".report-lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX)
        verified = report(args.output, plan)
        print(f"Verified {verified} / 200 pairs")
        if args.command == "run" and verified != len(plan["pairs"]):
            raise SystemExit(
                "Incomplete experiment. See report.json for excluded or missing pairs."
            )


if __name__ == "__main__":
    main()
