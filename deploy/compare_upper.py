"""Compare three upper models and one shared baseline on matched LIBERO states."""

import argparse
import fcntl
import itertools
import json
import random
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__:
    from . import libero40 as batch
    from .audit_comparison import failure_type
else:
    import libero40 as batch
    from audit_comparison import failure_type


def schedule(first, count, replicas):
    lanes = [[] for _ in range(replicas)]
    for offset in range(count):
        for i, case in enumerate(batch.cases()):
            slot = (i + i // len(batch.SUITES) + offset) % replicas
            lanes[slot].append((first + offset, case))
    return lanes


def comparisons(rows, arms):
    report = {}
    for left, right in itertools.combinations(arms, 2):
        pair = [
            dict(r, mode="base" if r["arm"] == left else "teacher")
            for r in rows
            if r["arm"] in (left, right)
        ]
        result = batch.summarize(pair)
        report[f"{left}_vs_{right}"] = dict(
            paired=result["paired_scored"],
            right_minus_left_pp=result["paired_success_delta_pp"],
            right_minus_left_seconds=result["paired_teacher_minus_base_wall"],
            hash_mismatches=result["initial_view_mismatches"],
            left=result["arms"]["base"],
            right=result["arms"]["teacher"],
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs=3, required=True)
    parser.add_argument("--initial-index", type=int, required=True)
    parser.add_argument("--initial-count", type=int, required=True)
    parser.add_argument("--remote-client", type=Path, action="append", required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--boyue-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-from", type=Path)
    parser.add_argument("--resume-from", type=Path)
    batch.harness_options(parser)
    args = parser.parse_args()
    settings = batch.harness_settings(args)
    if len(set(args.models)) != 3 or args.initial_index < 0 or args.initial_count < 1:
        parser.error("expected three distinct models and a valid initial-state range")
    if (
        not 1 <= args.replicas <= 4
        or len(set(args.remote_client)) != args.replicas
        or len(args.remote_client) != args.replicas
    ):
        parser.error("expected one distinct remote client per replica")
    args.output.mkdir(parents=True, exist_ok=False)
    arms = {"base": None, **{f"upper-{i}": model for i, model in enumerate(args.models, 1)}}
    lanes = schedule(args.initial_index, args.initial_count, args.replicas)
    for prior in filter(None, (args.repair_from, args.resume_from)):
        previous = json.loads((prior / "manifest.json").read_text())
        prior_settings = previous.get("executor_led")
        # Executor-led manifests written before the batch switch used the tool by default.
        previous_options = (
            {"completion_query": True, **prior_settings} if prior_settings is not None else {}
        )
        if any(
            getattr(args, k) is not None and getattr(args, k) != previous_options.get(k)
            for k in batch.harness_settings(args)
        ):
            parser.error("harness settings must match the source experiment")
        settings = prior_settings
        if (
            previous["arms"] != arms
            or previous["clients"] != list(map(str, args.remote_client))
            or previous["lanes"] != json.loads(json.dumps(lanes))
        ):
            parser.error("repair requires matching models, clients and case assignments")
    target = args.resume_from or args.output
    if args.resume_from:
        lease = (target / ".resume.lock").open("a")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if previous.get("repair_from") != (str(args.repair_from) if args.repair_from else None):
            parser.error("the resumed batch must have the same repair source")
    workers_per_replica = 2 if args.repair_from else 4
    batch.write_json(
        args.output / "manifest.json",
        dict(
            arms=arms,
            executor_led=settings,
            lanes=lanes,
            seed=batch.SEED,
            wall_s=batch.WALL,
            clients=list(map(str, args.remote_client)),
            initial_count=args.initial_count,
            initial_index=args.initial_index,
            upstream="cd6fcd91d19472ce24ea4dbd226930bafbf907c5",
            simulations=workers_per_replica * args.replicas,
            repair_from=str(args.repair_from) if args.repair_from else None,
            resume_from=str(args.resume_from) if args.resume_from else None,
        ),
    )
    for client in args.remote_client:
        batch.ready(client)
    use_upper = settings is None or any(settings[k] for k in ("planner", "supervisor", "teacher"))
    raw = json.loads(args.boyue_config.read_text())
    provider = raw.get("boyue", raw)
    with tempfile.TemporaryDirectory() as tmp:
        profiles = {"base": None}
        for arm, model in arms.items():
            if model is None:
                continue
            path = Path(tmp) / f"{arm}.json"
            path.write_text(
                json.dumps(
                    dict(
                        base_url=provider["base_url"],
                        api_key=provider["api_key"],
                        model=model,
                        allowed_returned_models=[model],
                    )
                )
            )
            path.chmod(0o600)
            if use_upper and not args.repair_from and not args.resume_from:
                subprocess.run(
                    [
                        sys.executable,
                        str(batch.HERE / "run_libero.py"),
                        "--check-upper",
                        str(path),
                        str(args.initial_index),
                    ],
                    check=True,
                    timeout=180,
                )
            profiles[arm] = path

        rows = []
        if args.resume_from:
            for file in sorted(target.glob("initial-*/*/*/task-*/*.result.json")):
                arm = file.parents[2].name
                rows.append(dict(json.loads(file.read_text()), arm=arm, requested_model=arms[arm]))

        lock = threading.Lock()

        preflight_locks = {arm: threading.Lock() for arm in arms}
        verified = set()

        def run_block(slot):
            if args.repair_from:
                # Avoid competing for the GPU with any first-pass case on this replica.
                final_index, (final_suite, final_task) = lanes[slot][-1]
                last = args.repair_from / f"initial-{final_index:02d}"
                markers = [
                    last
                    / arm
                    / final_suite
                    / f"task-{final_task:02d}"
                    / ("base.result.json" if arm == "base" else "teacher.result.json")
                    for arm in arms
                ]
                while not all(path.is_file() for path in markers):
                    time.sleep(1800)
            if args.repair_from or args.resume_from:
                # The original run already exercised all three models. Check the provider
                # again when GPU capacity becomes available, rather than while it is idle.
                for arm, profile in profiles.items():
                    if profile is None or not use_upper:
                        continue
                    with preflight_locks[arm]:
                        if arm in verified:
                            continue
                        while True:
                            try:
                                subprocess.run(
                                    [
                                        sys.executable,
                                        str(batch.HERE / "run_libero.py"),
                                        "--check-upper",
                                        str(profile),
                                        str(args.initial_index),
                                    ],
                                    check=True,
                                    timeout=180,
                                )
                            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                                batch.write_json(
                                    args.output / f"{arm}-preflight.json",
                                    {"ready": False, "time": time.time()},
                                )
                                time.sleep(1800)
                            else:
                                verified.add(arm)
                                batch.write_json(
                                    args.output / f"{arm}-preflight.json",
                                    {"ready": True, "time": time.time()},
                                )
                                break
            with ThreadPoolExecutor(workers_per_replica) as workers:
                for initial, case in lanes[slot]:
                    arms_order = list(arms)
                    random.Random(f"{batch.SEED}/{initial}/{case}").shuffle(arms_order)
                    pending = {}
                    for arm in arms_order:
                        if args.repair_from:
                            folder = (
                                args.repair_from
                                / f"initial-{initial:02d}"
                                / arm
                                / case[0]
                                / f"task-{case[1]:02d}"
                            )
                            old_mode = "base" if arm == "base" else "teacher"
                            first = json.loads((folder / f"{old_mode}.result.json").read_text())
                            if first["success"] is not None or not failure_type(
                                folder / f"{old_mode}.log"
                            ).startswith("transport_"):
                                continue
                        output = target / f"initial-{initial:02d}" / arm
                        mode = "base" if arm == "base" else "teacher"
                        folder = output / case[0] / f"task-{case[1]:02d}"
                        if args.resume_from and (folder / f"{mode}.result.json").exists():
                            continue
                        if args.resume_from:
                            archive = args.output / "interrupted" / folder.relative_to(target)
                            for orphan in (folder / mode, folder / f"{mode}.log"):
                                if orphan.exists():
                                    archive.mkdir(parents=True, exist_ok=True)
                                    orphan.rename(archive / orphan.name)

                        config = argparse.Namespace(
                            output=output,
                            initial_index=initial,
                            boyue_config=profiles[arm],
                            harness_settings=settings,
                        )
                        prior = args.repair_from or args.resume_from
                        if prior:
                            config.episode_runner = prior.parent / "source/deploy/run_libero.py"
                        pending[arm] = workers.submit(
                            batch.run_episode, config, args.remote_client[slot], slot, case, mode
                        )
                    if not pending:
                        continue
                    block = [
                        dict(future.result(), arm=arm, requested_model=arms[arm])
                        for arm, future in pending.items()
                    ]
                    with lock:
                        rows.extend(block)
                        batch.write_json(target / "episodes.json", rows)
                        batch.write_json(target / "report.json", comparisons(rows, arms))

        with ThreadPoolExecutor(args.replicas) as workers:
            list(workers.map(run_block, range(args.replicas)))
        print("LIBERO40_COMPARE_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
