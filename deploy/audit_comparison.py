"""Collect native outcomes and infrastructure failures from a four-arm LIBERO run."""

import json
import re
import sys
from pathlib import Path


def failure_type(log):
    with log.open(errors="replace") as stream:
        tail = list(stream)[-80:]
    errors = [s.strip() for s in tail if re.match(r"^[\w.]+(?:Error|Exception):", s)]
    error = errors[-1] if errors else ""
    if "timed out" in error or error.startswith("TimeoutError:"):
        return "transport_timeout"
    if "EOF occurred in violation of protocol" in error or "Connection reset" in error:
        return "transport_disconnect"
    status = re.search(r"returned HTTP (408|429|500|502|503|504)\b", error)
    if status:
        return "transport_http_" + status[1]
    if error.startswith(("ValueError:", "KeyError:", "TypeError:")):
        return "invalid_response"
    return "unclassified"


def audit_result(file):
    row = json.loads(file.read_text())
    arm = file.parents[2].name
    episode = file.parent / row["mode"]
    meta = json.loads((episode / "simulation/task.json").read_text())
    row.update(
        arm=arm,
        task_name=meta["task_name"],
        instruction=meta["instruction"],
        recorded_success=row["success"],
        score_source="rollout",
    )
    if row["success"] is not None:
        return row
    row["failure_type"] = failure_type(file.with_name(row["mode"] + ".log"))
    row["score_source"] = "unknown"
    events = [
        json.loads(line) for line in (episode / "rollout/events.jsonl").read_text().splitlines()
    ]
    if not events or events[-1]["event"] != "infrastructure_error":
        return row
    executions = [e for e in events if e["event"] == "execution"]
    started = sum(e["event"] == "action_started" for e in events)
    if started != len(executions) or started != events[-1].get("actions_completed"):
        return row
    latest = episode / "simulation/latest.json"
    if executions and latest.is_file():
        last = json.loads(latest.read_text())
        if (
            last.get("feedback") == executions[-1]["feedback"]
            and type(last.get("native_success")) is bool
        ):
            row.update(success=last["native_success"], score_source="last_native_action")
    elif not executions and not latest.exists() and type(meta.get("initial_success")) is bool:
        row.update(success=meta["initial_success"], score_source="native_initial_state")
    return row


if __name__ == "__main__":
    root = Path(sys.argv[1])
    files = sorted(root.glob("initial-*/*/*/task-*/*.result.json"))
    print(json.dumps([audit_result(f) for f in files]))
