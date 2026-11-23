"""Evaluate episode-level alert placement, holding all initial states of one task out."""

import argparse
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .replay_stalls import replay
else:
    from replay_stalls import replay

from openeta_bench.packs import load_pack

from executor_led import StallDetector


def score(rows, rule):
    positives = sum(not row["success"] for row in rows)
    negatives = len(rows) - positives
    tp = sum(not row["success"] and bool(row[rule]) for row in rows)
    fp = sum(row["success"] and bool(row[rule]) for row in rows)
    balanced = ((tp / positives) + (1 - fp / negatives)) / 2
    return dict(failures_alerted=tp, successes_alerted=fp, balanced_accuracy=balanced)


def audit(traces, reference):
    def key(row):
        return row["suite"], row["task"], row["initial"]

    if any(r["arm"] != "base" or type(r["success"]) is not bool for r in traces):
        raise ValueError("expected scored original-harness episodes")
    pack = load_pack("libero")
    old = {key(r): r for r in reference["episodes"] if r["arm"] == "base"}
    if not reference.get("inferred_single") or {key(r) for r in traces} != old.keys():
        raise ValueError("expected matching base episodes and inferred-single reference")
    configs = {f"{mm}mm_{count}of8": (mm, count) for mm in (20, 30, 40) for count in (3, 4)}
    rows = []
    for trace in traces:
        actions = 0
        last_decision = 0
        for event, *_ in trace["events"]:
            if event == "execution":
                actions += 1
            elif event == "response":
                last_decision = actions
        row = {k: trace[k] for k in ("suite", "task", "initial", "success", "name")}
        row.update(actions=actions, old=old[key(trace)]["hits"], age40=last_decision >= 40)
        for name, (mm, count) in configs.items():
            detector = StallDetector(tracking_mm=mm, tracking_count=count)
            row[name] = replay(trace, pack, 3, 8, True, detector=detector)["hits"]
        rows.append(row)
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError("duplicate rollout")
    folds = Counter()
    for suite, task in sorted({(r["suite"], r["task"]) for r in rows}):
        training = [r for r in rows if (r["suite"], r["task"]) != (suite, task)]

        def rank(name, training=training):
            s = score(training, name)
            return s["balanced_accuracy"], -s["successes_alerted"], configs[name]

        selected = max(configs, key=rank)
        folds[selected] += 1
        for r in rows:
            if (r["suite"], r["task"]) == (suite, task):
                r.update(heldout=r[selected], heldout_rule=selected)
    return dict(
        rows=rows,
        folds=folds,
        scores={k: score(rows, k) for k in ("old", *configs, "heldout", "age40")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.events.open() as stream:
        result = audit(
            [json.loads(line) for line in stream], json.loads(args.reference.read_text())
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("folds", "scores")}, indent=2))


if __name__ == "__main__":
    main()
