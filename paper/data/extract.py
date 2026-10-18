"""Collect the evaluation results behind the paper into paper/data (run from the repository root, any Python 3):

    python3 paper/data/extract.py

For every run used in the paper it copies the per-episode result lines ([eval] / [compose]) and the run summary out
of the full evaluation log (outputs/, not in git; simulator warnings dropped) into paper/data/<experiment>/<run>.log,
and writes paper/data/summary.csv: success rates and two-sided sign tests on the paired episodes of each comparison.
"""

import csv
import re
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent

RUNS = {  # experiment -> {run name: evaluation log}
    "reliability": {
        "baseline_still": "outputs/stage_rc_eval2/new_fused/find_verdict_first_still.log",
        "baseline_push": "outputs/stage_rc_eval2/new_fused/find_verdict_first_push.log",
        "final_still": "outputs/eval_gate/still.log",
        "final_push": "outputs/eval_gate/push.log",
    },
    "exp1_mono_vs_skills": {
        "mono_still": "outputs/exp1/eval_mono/still.log",
        "mono_push": "outputs/exp1/eval_mono/push.log",
        "skills_still": "outputs/eval_gate/still.log",
        "skills_push": "outputs/eval_gate/push.log",
    },
    "exp2_add_skill": {
        "stack_skill_100": "outputs/exp2/stack_skill_100/still.log",
        "stack_skill_300": "outputs/exp2/stack_skill_300/still.log",
        "stack_skill_place300": "outputs/exp2/stack_skill_place300/still.log",
        "stack_skill_300_gate": "outputs/exp2/stack_skill_300_gate/still.log",
        "stack_reuse_place": "outputs/exp2/stack_reuse_place/still.log",
        "stack_mono_ft": "outputs/exp2/stack_mono_ft/still.log",
        "place_mono_ft": "outputs/exp2/place_mono_ft/still.log",
        "stack_mono_replay": "outputs/exp2/stack_mono_replay/still.log",
        "place_mono_replay": "outputs/exp2/place_mono_replay/still.log",
    },
    "exp3_composed": {f"{c}_{a}": f"outputs/exp3/{c}_{a}/still.log"
                      for c in ("pairs", "plate_stack", "tower") for a in ("skills", "mono")},
    "exp4_task_vs_skill": {f"{s}_{k}": f"outputs/exp4/{s}_{k}/still.log"
                           for s in ("seen_task", "seen_skill") for k in ("bb", "ry", "bw", "rg")}
                          | {f"{s}_{k}": f"outputs/exp4/{s}_{k}/still.log"
                             for s in ("unseen_task", "unseen_skill") for k in ("bo", "rp")},
}

COMPARISONS = [  # (label, experiment, run A, run B)
    ("final vs baseline, stationary", "reliability", "final_still", "baseline_still"),
    ("final vs baseline, push", "reliability", "final_push", "baseline_push"),
    ("monolith vs skills, stationary", "exp1_mono_vs_skills", "mono_still", "skills_still"),
    ("monolith vs skills, push", "exp1_mono_vs_skills", "mono_push", "skills_push"),
    ("stack 300 vs 100 demos", "exp2_add_skill", "stack_skill_300", "stack_skill_100"),
    ("stack from place vs from scratch", "exp2_add_skill", "stack_skill_place300", "stack_skill_300"),
    ("monolith+replay vs stack from place", "exp2_add_skill", "stack_mono_replay", "stack_skill_place300"),
] + [(f"task vs skill, {s} {k}", "exp4_task_vs_skill", f"{s}_task_{k}", f"{s}_skill_{k}")
     for s, ks in (("seen", ("bb", "ry", "bw", "rg")), ("unseen", ("bo", "rp"))) for k in ks]

KEEP = re.compile(r"^\[(eval|compose)\]")


def episodes(text):
    out = {}
    for line in text.splitlines():
        if m := re.match(r"\[eval\] +(\d+) ", line):
            out.setdefault(int(m.group(1)), []).append(line)
    return out


def ok(lines, key):
    """An episode counts as a success if its last sub-task line reports it (composed tasks: see [compose])."""
    return f"{key}=ok" in lines[-1]


def sign_test(w, l):
    n, k = w + l, min(w, l)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


texts = {}
for exp, runs in RUNS.items():
    (OUT / exp).mkdir(parents=True, exist_ok=True)
    for name, path in runs.items():
        src = ROOT / path
        if not src.exists():
            print(f"[data] missing {path}")
            continue
        text = src.read_text(errors="ignore")
        texts[(exp, name)] = text
        kept = [ln for ln in text.splitlines() if KEEP.match(ln)]
        (OUT / exp / f"{name}.log").write_text("\n".join(kept) + "\n")

rows = []
for label, exp, a, b in COMPARISONS:
    if (exp, a) not in texts or (exp, b) not in texts:
        continue
    A, B = episodes(texts[(exp, a)]), episodes(texts[(exp, b)])
    keys = sorted(set(A) & set(B))
    for key, metric in (("pick", "pick"), ("place", "full")):
        ra = sum(ok(A[e], key) for e in keys)
        rb = sum(ok(B[e], key) for e in keys)
        w = sum(ok(A[e], key) and not ok(B[e], key) for e in keys)
        l_ = sum(ok(B[e], key) and not ok(A[e], key) for e in keys)
        rows.append({"comparison": label, "metric": metric, "episodes": len(keys), "A": a, "A_success": ra,
                     "B": b, "B_success": rb, "A_better": w, "B_better": l_, "sign_test_p": f"{sign_test(w, l_):.3g}"})
# Experiment 3: composed success from the [compose] lines (all sub-tasks on target in the final scene).
for c in ("pairs", "plate_stack", "tower"):
    a_, b_ = f"{c}_mono", f"{c}_skills"
    if ("exp3_composed", a_) not in texts or ("exp3_composed", b_) not in texts:
        continue

    def comp(text):
        return {int(m.group(1)): m.group(2) == "SUCCESS"
                for m in re.finditer(r"^\[compose\] +(\d+) .*?: (SUCCESS|FAIL)", text, re.M)}

    A, B = comp(texts[("exp3_composed", a_)]), comp(texts[("exp3_composed", b_)])
    keys = sorted(set(A) & set(B))
    w = sum(A[e] and not B[e] for e in keys)
    l_ = sum(B[e] and not A[e] for e in keys)
    rows.append({"comparison": f"monolith+replay vs skills, composed {c}", "metric": "composed", "episodes": len(keys),
                 "A": a_, "A_success": sum(A[e] for e in keys), "B": b_, "B_success": sum(B[e] for e in keys),
                 "A_better": w, "B_better": l_, "sign_test_p": f"{sign_test(w, l_):.3g}"})
with open(OUT / "summary.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print(f"[data] {sum(1 for _ in OUT.rglob('*.log'))} run logs, {len(rows)} comparison rows -> {OUT}")
