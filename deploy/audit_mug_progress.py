"""Offline-only comparison of supervision rules for LIBERO's two mugs and two plates."""

import argparse
import json
from itertools import groupby
from pathlib import Path
from statistics import median


def known_pair(values):
    if (
        isinstance(values, (tuple, list))
        and len(values) == 2
        and all(type(value) is bool for value in values)
    ):
        return tuple(values)
    return None


class PlacementReview:
    """Only the optional wrong-support rule receives the two extra audit predicates."""

    def __init__(self):
        self.partial = None
        self.phase = "need_open"
        self.releases = 0
        self.review_after = None
        self.reviewed = False
        self.jaw = None
        self.wrong_since = [None, None]
        self.wrong_reported = [False, False]
        self.last_tick = -1

    def observe(self, tick, goals, commands, crossed=None, finish=False):
        if tick <= self.last_tick:
            raise ValueError("samples must advance simulation time")
        self.last_tick = tick
        goals = known_pair(goals)
        crossed = known_pair(crossed)
        partial = goals
        if partial is None or sum(partial) != 1:
            partial = None
        if partial is None or partial != self.partial:
            self.phase = "need_open"
            self.releases = 0
            self.review_after = None
            self.reviewed = False
        else:
            # Only commands after the first one-of-two receipt can count.
            for cmd in commands:
                if cmd == "open" and self.phase == "need_release":
                    self.releases += 1
                    if self.releases >= 2:
                        self.review_after = tick + 10  # Allow contact to settle for 0.5 s.
                if cmd == "open":
                    self.phase = "need_close"
                elif self.phase == "need_close" and cmd == "close":
                    self.phase = "need_release"
                    self.review_after = None
        self.partial = partial
        for cmd in commands:
            if cmd in {"open", "close"}:
                self.jaw = cmd
        hints = []
        if (
            partial
            and not self.reviewed
            and (finish or self.review_after is not None and tick >= self.review_after)
        ):
            hints.append("partial_finish" if finish else "partial_after_two_releases")
            self.reviewed = True
        for mug in range(2):
            # Separate two sightings after release, never combine different cups.
            observed = (
                goals is not None
                and crossed is not None
                and crossed[mug] is True
                and goals[mug] is False
                and self.jaw == "open"
                and "close" not in commands
            )
            if not observed:
                self.wrong_since[mug] = None
                self.wrong_reported[mug] = False
            elif self.wrong_since[mug] is None:
                self.wrong_since[mug] = tick
            elif tick - self.wrong_since[mug] >= 10 and not self.wrong_reported[mug]:
                hints.append(("white_on_wrong_plate", "yellow_white_on_wrong_plate")[mug])
                self.wrong_reported[mug] = True
        return hints


def demo_samples(rows, stride):
    """Group recorded 20 Hz controls without using a future state or gripper command."""
    commands = []
    for idx, row in enumerate(rows):
        value = row["cmd"]
        if value is not None:
            commands.append("close" if value > 0 else "open" if value < 0 else "hold")
        if idx % stride == 0 or idx == len(rows) - 1:
            yield dict(row, tick=idx, action=idx, commands=[k for k, _ in groupby(commands)])
            commands.clear()


def audit(samples):
    candidate = PlacementReview()
    alerts = []
    partial_at = success_at = None
    longest = 0
    last_partial = None
    start = None
    lost_first_placement = False
    for row in samples:
        goals, tick = known_pair(row["flags"]), row["tick"]
        if goals == (True, True):
            success_at = tick
            break
        partial = goals if goals and sum(goals) == 1 else None
        if partial_at is not None and goals == (False, False):
            lost_first_placement = True
        if partial != last_partial:
            start = tick if partial else None
        if partial:
            if partial_at is None:
                partial_at = tick
            longest = max(longest, tick - start)
        last_partial = partial
        for reason in candidate.observe(
            tick, goals, row["commands"], crossed=row.get("cross"), finish=row.get("finish", False)
        ):
            alerts.append(dict(action=row["action"], tick=tick, reason=reason))
    return dict(
        first_partial=partial_at,
        first_success=success_at,
        longest_partial_ticks=longest,
        saw_zero_after_partial=lost_first_placement,
        alerts=alerts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    d = args.directory
    demos = json.loads((d / "demo-states.json").read_text())
    if len({r["demo"] for r in demos}) != len(demos):
        raise ValueError("duplicate demonstration")
    out = {"demos": {}, "summary": {}, "reconstructed": {}}
    for stride in (1, 10, 20, 30):
        runs = {x["demo"]: audit(demo_samples(x["rows"], stride)) for x in demos}
        passed = [r for r in runs.values() if r["first_success"] is not None]
        gaps = sorted(
            (r["first_success"] - r["first_partial"]) / 20
            for r in passed
            if r["first_partial"] is not None
        )
        out["demos"][stride] = runs
        out["summary"][stride] = dict(
            available=len(runs),
            observed_success=len(passed),
            saw_partial=len(gaps),
            saw_zero_after_partial=sum(r["saw_zero_after_partial"] for r in passed),
            alert_on_partial=len(gaps),
            alert_after_2s=sum(r["longest_partial_ticks"] >= 40 for r in passed),
            alert_after_5s=sum(r["longest_partial_ticks"] >= 100 for r in passed),
            candidate_alerts_on_success=sum(bool(r["alerts"]) for r in passed),
            candidate_alerts_all=sum(bool(r["alerts"]) for r in runs.values()),
            first_partial_to_success_s=[min(gaps), median(gaps), max(gaps)] if gaps else None,
        )
    for i in range(3):
        rows = json.loads((d / f"reconstructed-{i}.json").read_text())
        out["reconstructed"][i] = audit(rows)
    (d / "review.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("summary", "reconstructed")}, indent=2))


if __name__ == "__main__":
    main()
