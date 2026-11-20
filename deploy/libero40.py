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
        if profile["model"] != "eta0-system-fullmix-2500" or profile["model"] not in {
            m["id"] for m in json.load(response)["data"]
        }:
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
    for i in range(expected // 2):
        session = root / f"replica-{i + 1}"
        session.mkdir()
        client = session / "client.json"
        devices = ",".join(gpu_ids[2 * i : 2 * i + 2])
        log = stack.enter_context((session / "startup.log").open("w"))
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(HERE / "serve.py"),
                "--session",
                str(session),
                "--port",
                str(18800 + 10 * i),
            ],
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
        str(HERE / "run_libero.py"),
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
    if mode == "teacher":
        cmd += ["--boyue-config", str(args.boyue_config)]
    else:
        cmd += ["--upstream-base"]
    print(f"LIBERO40_START {suite}/{task} {mode} replica={slot}", flush=True)
    start = time.monotonic()
    with (folder / f"{mode}.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=WALL + 150)
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
    upper_s = 0
    if events.is_file():
        with events.open() as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # a killed process may leave a partial final event
                if entry["event"] == "upper_response":
                    upper_s += entry["elapsed_s"]
    row = dict(
        suite=suite,
        task=task,
        mode=mode,
        replica=slot,
        initial_index=args.initial_index,
        seed=SEED,
        elapsed_s=round(wall, 3),
        upper_response_s=round(upper_s, 3),
        returncode=proc.returncode,
        initial_view_sha256=view_hash,
        initial_state_sha256=metadata.get("initial_state_sha256"),
        bddl_sha256=metadata.get("bddl_sha256"),
        reason=result.get("reason", "incomplete"),
        success=result.get("success"),
        calls=result.get("calls"),
        actions=result.get("actions"),
    )
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
    p.add_argument("--replicas", type=int, choices=[2, 3, 4], default=4)
    p.add_argument("--executor-client", type=Path)
    for name in ["boyue-config", "output"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--initial-index", required=True, type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = cases()
    lanes = 2 * args.replicas
    write_json(
        args.output / "manifest.json",
        dict(
            tasks=tasks,
            modes=MODES,
            seed=SEED,
            initial_index=args.initial_index,
            upstream="cd6fcd91d19472ce24ea4dbd226930bafbf907c5",
            replicas=args.replicas,
            sim_workers=2 * lanes,
            wall_s=WALL,
        ),
    )
    with ExitStack() as stack:
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
