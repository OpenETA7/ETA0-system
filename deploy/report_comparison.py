"""Render matched improvements and total end-to-end timing for the four-arm study."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MATCH = ("replica", "seed", "initial_view_sha256", "initial_state_sha256", "bddl_sha256")
CASES_PER_TASK = 5


def identity(r):
    return (r["suite"], r["task"], r["initial_index"], r["arm"])


def matches(a, b):
    return all(a.get(k) is not None and a[k] == b.get(k) for k in MATCH)


def aggregate(rows, arms):
    expected = set(arms)
    grouped = defaultdict(dict)
    for r in rows:
        suite, task, initial, arm = identity(r)
        group = grouped[suite, task, initial]
        if arm not in expected or arm in group:
            raise ValueError(f"Unexpected or duplicate row {identity(r)}")
        group[arm] = r
        if r.get("upper_models") and set(r["upper_models"]) != {arms[arm]}:
            raise ValueError(f"Returned model mismatch at {identity(r)}")

    tasks = []
    for suite, task in sorted({k[:2] for k in grouped}):
        for arm in list(arms)[1:]:
            wins, losses, same, deltas = [], [], [], []
            name = ""
            for initial in range(CASES_PER_TASK):
                group = grouped.get((suite, task, initial), {})
                b, t = group.get("base"), group.get(arm)
                if not b or not t or None in (b["success"], t["success"]) or not matches(b, t):
                    continue
                name = b["instruction"]
                if t["success"] == b["success"]:
                    same.append(initial)
                elif t["success"]:
                    wins.append(initial)
                else:
                    losses.append(initial)
                deltas.append((t["elapsed_s"] - b["elapsed_s"]) / 60)
            tasks.append(
                dict(
                    suite=suite,
                    task=task,
                    instruction=name,
                    model=arms[arm],
                    paired=len(deltas),
                    wins=wins,
                    losses=losses,
                    net=len(wins) - len(losses),
                    mean_delta_min=round(statistics.mean(deltas), 2) if deltas else None,
                )
            )
    valid = sum(
        set(g) == expected
        and all(r["success"] is not None and matches(r, g["base"]) for r in g.values())
        for g in grouped.values()
    )
    return tasks, valid


def render(out, rows, arms, stem, label):
    tasks, valid = aggregate(rows, arms)
    with (out / (stem + "-tasks.csv")).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "suite",
                "task",
                "instruction",
                "model",
                "paired",
                "wins",
                "losses",
                "net",
                "mean_delta_min",
            ),
        )
        writer.writeheader()
        writer.writerows(tasks)
    lines = [
        f"# {label}",
        "",
        "每项任务使用五个初态（0–4）。模型间只比较任务定义、初态、首帧、随机种子和底层副本一致的轨迹。",
        "",
        "| 条件 | 已返回 | 有终点判定 | 成功 | 成功率（按全部返回） |"
        " 平均分钟 | 中位分钟 | P90 分钟 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, model in arms.items():
        selected = [r for r in rows if r["arm"] == arm]
        valid_arm = sum(type(r["success"]) is bool for r in selected)
        if not selected:
            continue
        dur = sorted(r["elapsed_s"] / 60 for r in selected)
        wins = sum(r["success"] is True for r in selected)
        lines.append(
            f"| {model or '原 harness'} | {len(selected)} | {valid_arm} | {wins} |"
            f" {wins / len(selected):.1%} | {statistics.mean(dur):.2f} |"
            f" {statistics.median(dur):.2f} |"
            f" {dur[(9 * len(dur) + 9) // 10 - 1]:.2f} |"
        )
    lines += [
        "",
        f"四组均取得判定且配对一致的任务初态有 {valid}/200 个，已返回 {len(rows)}/800 条。",
        "",
        "## 相对原 harness 的逐任务改善",
        "",
        "每个任务须完成五个有效配对才列入改善表。"
        "新增成功是原版失败而该模型成功的初态，丢失成功是相反方向。净增等于两者之差。",
        "",
        "| 上层模型 | Suite / 任务 | LIBERO 指令 | 新增成功初态 |"
        " 丢失成功初态 | 净增 / 5 | 平均耗时差（分钟） |",
        "|---|---|---|---|---|---:|---:|",
    ]
    improved = sorted(
        (t for t in tasks if t["paired"] == CASES_PER_TASK and t["net"] > 0),
        key=lambda r: (r["model"], -r["net"], r["suite"], r["task"]),
    )

    def ids(v):
        return ", ".join(map(str, v)) or "—"

    for t in improved:
        lines.append(
            f"| {t['model']} | {t['suite']} / {t['task']:02} | {t['instruction']} |"
            f" {ids(t['wins'])} | {ids(t['losses'])} | +{t['net']} | {t['mean_delta_min']:+.2f} |"
        )
    if not improved:
        lines += ["", "当前满足五个有效配对且成功数净增的任务为 0。"]
    lines += [
        "",
        f"所有任务的增益、退步、持平及耗时差见 [{stem}-tasks.csv]({stem}-tasks.csv)。",
        "",
        "## 中断与计时",
        "",
    ]
    for arm, model in arms.items():
        rr = [r for r in rows if r["arm"] == arm]
        err = Counter(r["failure_type"] for r in rr if r.get("failure_type"))
        if rr:
            recovered = sum(
                r["score_source"] in ("native_initial_state", "last_native_action") for r in rr
            )
            lines.append(
                f"- {model or '原 harness'} 有 {sum(err.values())} 次异常退出。"
                f"终点判定恢复自已执行动作或初态的有 {recovered} 次。"
            )
    lines += ["", "耗时为仿真进程启动至退出的墙钟，包含排队和 API 重试。提前中断会缩短所报耗时。"]
    (out / (stem + ".md")).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    rows = json.loads((out / "first-pass.json").read_text())
    arms = json.loads((out / "manifest.json").read_text())["arms"]
    render(out, rows, arms, "comparison", "LIBERO 四组首轮对照")
    retry_file = out / "repair.json"
    if not retry_file.exists():
        return
    retries = {identity(r): r for r in json.loads(retry_file.read_text())}
    original = {identity(r): r for r in rows}
    if any(
        k not in original
        or not original[k].get("failure_type", "").startswith("transport_")
        or not matches(original[k], r)
        for k, r in retries.items()
    ):
        raise ValueError("Supplement contains an ineligible or unmatched trial")
    merged = [retries.get(identity(r), r) for r in rows]
    render(out, merged, arms, "comparison-with-repair", "每项网络中断补测一次后的对照")
    with (out / "comparison-with-repair.md").open("a") as f:
        hours = sum(original[k]["elapsed_s"] for k in retries) / 3600
        f.write(
            f"\n补测替换了 {len(retries)} 个网络异常项目，无论补测成功或失败均选用第二次结果。"
            f"表中时间为所选尝试的耗时。首轮尝试另累计耗时 {hours:.2f} episode 小时。"
            "补测每个副本至多并发两个 episode，首轮为四个。\n"
        )


if __name__ == "__main__":
    main()
