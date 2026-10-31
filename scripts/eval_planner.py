"""Score the planner LLM (so101_lang/planner.py) on the instruction test set (tests/planner_cases.json).

No simulator needed: each case gives a scene (cubes, plates, held cube) and the expected skill calls, or null when
the planner must refuse. Deterministic (temperature 0). Needs ``ollama serve`` running.

    python scripts/eval_planner.py [--model qwen3-vl:8b] [--cat multi] [--out outputs/planner/qwen3-vl.json]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_lang.planner import DEFAULT_MODEL, Planner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cases", default=str(Path(__file__).resolve().parents[1] / "tests/planner_cases.json"))
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--cat", default=None, help="only this category")
parser.add_argument("--cpu", action="store_true", help="run the LLM on the CPU (GPU busy with the simulator)")
parser.add_argument("--out", default=None)
args = parser.parse_args()


def groups(calls):
    """Split calls into pick-groups (a pick and the place / stack that follows)."""
    out, cur = [], []
    for c in calls:
        if c[0] == "pick" and cur:
            out.append(tuple(cur))
            cur = []
        cur.append(tuple(c))
    return out + ([tuple(cur)] if cur else [])


def correct(case, plan):
    accepted = [case["expect"]] + [v for k, v in case.items() if k.startswith("alt_expect")]
    for exp in accepted:
        if exp is None:
            if not plan.ok:
                return True
        elif plan.ok:
            got = [tuple(c) for c in plan.calls]
            exp = [tuple(c) for c in exp]
            if got == exp or (case.get("any_order") and Counter(groups(got)) == Counter(groups(exp))):
                return True
    return False


data = json.loads(Path(args.cases).read_text())
planner = Planner(model=args.model, gpu=not args.cpu)
results, by_cat = [], defaultdict(list)
for case in data["cases"]:
    if args.cat and case["cat"] != args.cat:
        continue
    scene = {**data["default_scene"], **case.get("scene", {})}
    plan = planner.plan(case["instruction"], scene["cubes"], scene["plates"], case.get("holding"))
    ok = correct(case, plan)
    by_cat[case["cat"]].append(ok)
    got = plan.calls if plan.ok else f"REFUSED: {plan.error}"
    print(f"[planner] {'OK ' if ok else 'BAD'} {plan.seconds:5.1f}s {len(plan.raw)}x [{case['cat']}] "
          f"{case['instruction']!r} -> {got}", flush=True)
    results.append({**case, "plan": plan.calls, "error": plan.error, "correct": ok, "seconds": round(plan.seconds, 2),
                    "attempts": len(plan.raw), "raw": plan.raw})
n = len(results)
secs = sorted(r["seconds"] for r in results)
print(f"[planner] {args.model}: {sum(r['correct'] for r in results)}/{n} correct; "
      + ", ".join(f"{c} {sum(v)}/{len(v)}" for c, v in by_cat.items())
      + f"; median {secs[n // 2]:.1f}s, max {secs[-1]:.1f}s per plan")
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
