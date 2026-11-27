"""Run 40 paired LIBERO tasks on four colocated TP2 replicas and write the comparison."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODES = ("base", "teacher")
SEED = 1234
HERE = Path(__file__).resolve().parent
WALL = 2700


def harness_options(parser):
    for role in ("planner", "supervisor", "teacher"):
        parser.add_argument("--" + role, action=argparse.BooleanOptionalAction)
    parser.add_argument("--upper-wall-seconds", type=int)
    parser.add_argument("--libero-source-root")
    parser.add_argument("--completion-query", action=argparse.BooleanOptionalAction)
    parser.add_argument("--upstream-base", action=argparse.BooleanOptionalAction, default=None)


def harness_settings(args):
    settings = dict(
        planner=True,
        supervisor=True,
        teacher=False,
        upper_wall_seconds=900,
        libero_source_root=None,
        completion_query=True,
        upstream_base=False,
    )
    settings.update({k: getattr(args, k) for k in settings if getattr(args, k, None) is not None})
    if settings["upper_wall_seconds"] < 0:
        raise ValueError("upper wall budget must be nonnegative")
    return settings


def cases():
    rng = random.Random(SEED)
    task_ids = [rng.sample(range(10), 10) for _ in SUITES]
    return [(suite, task_ids[j][i]) for i in range(10) for j, suite in enumerate(SUITES)]


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def ready(client):
    profile = json.loads(client.read_text())
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(profile["base_url"] + "/models")
    req.add_unredirected_header("Authorization", "Bearer " + profile["api_key"])
    with http.open(req, timeout=10) as response:
        if profile["model"] not in {m["id"] for m in json.load(response)["data"]}:
            raise RuntimeError("unexpected executor model")


def stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def start_replicas(root, existing, stack, replicas):
    if existing:
        ready(existing)
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if gpu_ids == [""]:
        count = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        )
        gpu_ids = count.split()
    expected = 2 * (replicas - bool(existing))
    if len(gpu_ids) != expected or len(set(gpu_ids)) != expected:
        raise RuntimeError(f"expected {expected} distinct GPUs, found {len(gpu_ids)}")
    clients = [existing] if existing else []
    workers = []
    model_path = os.environ.get("ETA0_MODEL_PATH")
    if expected and not model_path:
        raise RuntimeError("set ETA0_MODEL_PATH to launch local TP2 replicas")
    draft_path = os.environ.get("ETA0_DFLASH_DRAFT")
    for i in range(expected // 2):
        session = root / f"replica-{i + 1}"
        session.mkdir()
        client = session / "client.json"
        devices = ",".join(gpu_ids[2 * i : 2 * i + 2])
        log = stack.enter_context((session / "startup.log").open("w"))
        cmd = [
            sys.executable,
            "-u",
            str(HERE / "serve_vllm.py"),
            "--model",
            model_path,
            "--tensor-parallel-size",
            "2",
            "--session-dir",
            str(session),
            "--port",
            str(18800 + 10 * i),
        ]
        if draft_path:
            cmd.extend(["--draft", draft_path])
        proc = subprocess.Popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": devices},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stack.callback(stop_group, proc)
        workers.append(proc)
        clients.append(client)
    deadline = time.monotonic() + 2100
    while not all(p.is_file() for p in clients):
        for i, proc in enumerate(workers, 1):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"replica {i} exited during startup, see {root}/replica-{i}/startup.log"
                )
        if time.monotonic() >= deadline:
            raise TimeoutError("replicas did not become ready")
        time.sleep(3)
    for client in clients:
        ready(client)
    return clients


def run_episode(args, client, slot, case, mode):
    suite, task = case
    folder = args.output / suite / f"task-{task:02d}"
    episode = folder / mode
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(getattr(args, "episode_runner", HERE / "run_libero.py")),
        "--mode",
        mode,
        "--suite",
        suite,
        "--task",
        str(task),
        "--initial-index",
        str(args.initial_index),
        "--seed",
        str(SEED),
        "--output",
        str(episode),
        "--executor-client",
        str(client),
        "--recover-length",
        "--wall-seconds",
        str(WALL),
        "--max-actions",
        "120",
        "--max-calls",
        "360",
    ]
    settings = getattr(args, "harness_settings", harness_settings(args))
    if getattr(args, "progress_query_ablation", False):
        cmd.extend(["--progress-query-ablation", "--instruction-mode", "revised"])
    upper_wall = 0
    if settings is not None:
        if settings.get("libero_source_root"):
            cmd.extend(["--libero-source-root", settings["libero_source_root"]])
        # Older manifests kept the runner's default. Frozen upstream has no such tool.
        if "completion_query" in settings and (mode != "base" or not settings.get("upstream_base")):
            cmd.append(
                "--completion-query" if settings["completion_query"] else "--no-completion-query"
            )
        if mode == "teacher":
            for role in ("planner", "supervisor", "teacher"):
                cmd.append(("--" if settings[role] else "--no-") + role)
            upper_wall = settings["upper_wall_seconds"]
            cmd.extend(["--upper-wall-seconds", str(upper_wall)])
    if mode == "teacher":
        cmd += ["--boyue-config", str(args.boyue_config)]
    elif settings is None or settings.get("upstream_base"):
        cmd += ["--upstream-base"]
    print(f"LIBERO40_START {suite}/{task} {mode} replica={slot}", flush=True)
    start = time.monotonic()
    with (folder / f"{mode}.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=WALL + upper_wall + 150)
        except subprocess.TimeoutExpired:
            stop_group(proc)
    wall = time.monotonic() - start
    result_path = episode / "rollout/RESULT.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if not result and not (episode / "rollout/observation-000").exists():
        raise RuntimeError(f"simulation initialization failed, see {folder}/{mode}.log")
    image = episode / "rollout/observation-000/external.png"
    view_hash = hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else None
    meta_file = episode / "simulation/task.json"
    metadata = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    events = episode / "rollout/events.jsonl"
    upper_times = Counter()
    upper_models = set()
    progress_events = 0
    if events.is_file():
        with events.open() as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # a killed process may leave a partial final event
                if entry["event"] == "upper_response":
                    upper_times[entry["role"]] += entry["elapsed_s"]
                    upper_models.add(entry["model"])
                elif entry["event"] == "completion_query":
                    progress_events += 1
    row = dict(
        suite=suite,
        task=task,
        mode=mode,
        replica=slot,
        initial_index=args.initial_index,
        seed=SEED,
        elapsed_s=round(wall, 3),
        upper_response_s=round(sum(upper_times.values()), 3),
        upper_role_s={role: round(seconds, 3) for role, seconds in upper_times.items()},
        upper_models=sorted(upper_models),
        returncode=proc.returncode,
        initial_view_sha256=view_hash,
        initial_state_sha256=metadata.get("initial_state_sha256"),
        bddl_sha256=metadata.get("bddl_sha256"),
        reason=result.get("reason", "incomplete"),
        success=result.get("success"),
        lower_wall_s=result.get("lower_wall_s"),
        upper_wall_s=result.get("upper_wall_s"),
        lower_chat_s=result.get("lower_chat_s"),
        lower_chat_calls=result.get("lower_chat_calls"),
        calls=result.get("calls"),
        completion_queries=result.get("completion_queries"),
        actions=result.get("actions"),
    )
    if getattr(args, "progress_query_ablation", False):
        row["completion_query_events"] = progress_events
    write_json(folder / f"{mode}.result.json", row)
    print("LIBERO40_TRIAL " + json.dumps(row), flush=True)
    return row


def timing(values):
    if not values:
        return None
    return dict(
        mean_s=round(statistics.mean(values), 2),
        median_s=round(statistics.median(values), 2),
        p90_s=round(sorted(values)[math.ceil(0.9 * len(values)) - 1], 2),
    )


def summarize(rows):
    arms = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        valid = [r for r in selected if r["success"] is not None]
        wins = sum(r["success"] is True for r in valid)
        arms[mode] = dict(
            attempted=len(selected),
            native_scored=len(valid),
            native_successes=wins,
            success_per_attempt=wins / len(selected) if selected else None,
            success_per_scored=wins / len(valid) if valid else None,
            end_to_end=timing([r["elapsed_s"] for r in selected]),
            scored_time=timing([r["elapsed_s"] for r in valid]),
            reasons=dict(Counter(r["reason"] for r in selected)),
        )
    by_case = {}
    for r in rows:
        by_case.setdefault((r["suite"], r["task"], r["initial_index"], r["seed"]), {})[
            r["mode"]
        ] = r
    paired = Counter()
    deltas = []
    mismatches = []
    for key, arms_by_mode in by_case.items():
        if len(arms_by_mode) != 2:
            continue
        b, t = [arms_by_mode[m] for m in MODES]
        if None in (b["success"], t["success"]):
            continue
        if any(
            not b[k] or b[k] != t[k]
            for k in ("initial_state_sha256", "bddl_sha256", "initial_view_sha256")
        ):
            mismatches.append(key)
            continue
        paired[
            {
                (True, True): "both",
                (False, False): "neither",
                (True, False): "base_only",
                (False, True): "teacher_only",
            }[b["success"], t["success"]]
        ] += 1
        deltas.append(t["elapsed_s"] - b["elapsed_s"])
    n = sum(paired.values())
    return dict(
        arms=arms,
        paired_scored=n,
        paired_outcomes=dict(paired),
        paired_success_delta_pp=round(100 * (paired["teacher_only"] - paired["base_only"]) / n, 2)
        if n
        else None,
        paired_teacher_minus_base_wall=timing(deltas),
        initial_view_mismatches=mismatches,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicas", type=int, choices=[1, 2, 3, 4], default=4)
    p.add_argument("--executor-client", type=Path)
    p.add_argument("--remote-client", type=Path, action="append", default=[])
    for name in ["boyue-config", "output"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--initial-index", required=True, type=int)
    harness_options(p)
    args = p.parse_args()
    args.harness_settings = harness_settings(args)
    if args.remote_client and (
        len(set(args.remote_client)) != args.replicas
        or len(args.remote_client) != args.replicas
        or args.executor_client
    ):
        p.error("remote execution requires one distinct client file per replica")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = cases()
    lanes = 2 * args.replicas
    write_json(
        args.output / "manifest.json",
        dict(
            tasks=tasks,
            executor_led=args.harness_settings,
            modes=MODES,
            seed=SEED,
            initial_index=args.initial_index,
            upstream="cd6fcd91d19472ce24ea4dbd226930bafbf907c5",
            replicas=args.replicas,
            sim_workers=2 * lanes,
            remote_clients=[str(p) for p in args.remote_client],
            wall_s=WALL,
        ),
    )
    if any(args.harness_settings[r] for r in ("planner", "supervisor", "teacher")):
        subprocess.run(
            [
                sys.executable,
                str(HERE / "run_libero.py"),
                "--check-upper",
                str(args.boyue_config),
                str(args.initial_index),
            ],
            check=True,
            timeout=180,
        )
    with ExitStack() as stack:
        if args.remote_client:
            clients = args.remote_client
            deadline = time.monotonic() + 3600
            for client in clients:
                while not client.is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"remote TP2 did not publish {client}")
                    time.sleep(3)
                ready(client)
        else:
            clients = start_replicas(args.output, args.executor_client, stack, args.replicas)
        print(f"LIBERO40_READY {args.replicas} TP2 replicas, {2 * lanes} episode slots", flush=True)
        results = []
        lock = threading.Lock()
        start = time.monotonic()

        def lane(idx):
            slot = idx % len(clients)
            with concurrent.futures.ThreadPoolExecutor(2) as pair:
                for i in range(idx, len(tasks), lanes):
                    # Balance suites across fixed replica slots as the lanes progress.
                    replica = slot
                    start_idx = i - idx
                    width = min(lanes, len(tasks) - start_idx)
                    case = tasks[start_idx + (idx + i // lanes) % width]
                    modes = MODES if (idx + i // lanes) % 2 else MODES[::-1]
                    pending = [
                        pair.submit(run_episode, args, clients[replica], replica, case, m)
                        for m in modes
                    ]
                    rows = [f.result() for f in pending]
                    with lock:
                        results.extend(rows)
                        report = summarize(results)
                        report["completed_episodes"] = len(results)
                        report["batch_elapsed_s"] = round(time.monotonic() - start, 2)
                        write_json(args.output / "report.json", report)

        with concurrent.futures.ThreadPoolExecutor(lanes) as pool:
            list(pool.map(lane, range(lanes)))
        print(
            "LIBERO40_REPORT " + json.dumps(json.loads((args.output / "report.json").read_text())),
            flush=True,
        )


if __name__ == "__main__":
    main()
