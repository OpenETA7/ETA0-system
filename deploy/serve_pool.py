"""Serve colocated TP2 replicas for CPU simulations."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

from libero40 import WALL, ready, start_replicas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    lifetime = p.add_mutually_exclusive_group(required=True)
    lifetime.add_argument("--batch", type=Path)
    lifetime.add_argument("--persistent", action="store_true", help="keep the workspace serving")
    p.add_argument("--existing", type=Path)
    p.add_argument("--total-replicas", type=int, choices=(2, 3, 4), required=True)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        clients = start_replicas(args.root, args.existing, stack, args.total_replicas)
        print("ETA0_REMOTE_POOL_READY", len(clients), flush=True)
        if args.persistent:
            while True:
                time.sleep(60)
                for client in clients:
                    ready(client)
        last_progress = time.time()
        while True:
            report = args.batch / "report.json"
            if report.is_file():
                if json.loads(report.read_text())["completed_episodes"] == 80:
                    return
                last_progress = max(last_progress, report.stat().st_mtime)
            if time.time() - last_progress > WALL + 900:
                raise TimeoutError("CPU batch stopped publishing completed task pairs")
            time.sleep(20)


if __name__ == "__main__":
    main()
