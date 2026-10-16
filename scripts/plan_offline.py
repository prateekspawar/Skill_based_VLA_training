"""Plan composed-task instructions with the planner LLM offline and score the plans (experiment 3).

Reads the ``[compose]`` lines of an ``eval_policy.py --compose ... --planner oracle`` log (instruction and outcome per
seeded episode), asks the planner (Ollama, CPU) for skill calls, and compares them with the oracle plan parsed from
the instruction template. The plan does not depend on the robot's execution, so an episode whose LLM plan equals
the oracle plan executes exactly like the oracle run: end-to-end success = oracle success AND plan correct.
Run in ~/lerobot-env with ``ollama serve`` up:

    ~/lerobot-env/bin/python scripts/plan_offline.py outputs/exp3/pairs/still.log [--out plans.json]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_lang import language  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log")
parser.add_argument("--out", default=None)
parser.add_argument("--model", default="qwen3-vl:8b")
parser.add_argument("--url", default="http://localhost:11434")
args = parser.parse_args()

SYSTEM = ("You are the task planner of a one-arm robot. Turn the user's instruction into a sequence of skill calls. Use "
          "only these skills (JSON schema):\n{schema}\nObjects in the scene: {objects}. The gripper holds one cube at a "
          "time: every pick is followed by a place or a stack of that cube. Reply only with JSON: "
          '{{"calls": [{{"skill": name, "args": {{param: value}}}}]}}.\n'
          'Examples:\nput the green cube on the white plate -> {{"calls": [{{"skill": "pick", "args": {{"object": '
          '"green cube"}}}}, {{"skill": "place", "args": {{"target": "white plate"}}}}]}}\nstack the yellow cube on the '
          'red cube -> {{"calls": [{{"skill": "pick", "args": {{"object": "yellow cube"}}}}, {{"skill": "stack", "args": '
          '{{"object": "red cube"}}}}]}}')
CUBES = [f"{c} cube" for c in ("red", "green", "blue", "yellow")]


def oracle(instr):
    """(skill, object) pairs of the instruction templates used by eval_policy --compose."""
    o = r"(\w+ (?:cube|plate))"
    if m := re.fullmatch(rf"put the {o} on the {o} and the {o} on the {o}", instr):
        x, a, y, b = m.groups()
        return [("pick", x), ("place", a), ("pick", y), ("place", b)]
    if m := re.fullmatch(rf"put the {o} on the {o}, then stack the {o} on top of it", instr):
        x, a, y = m.groups()
        return [("pick", x), ("place", a), ("pick", y), ("stack", x)]
    if m := re.fullmatch(rf"stack the {o} on the {o}, then put the {o} on top of the {o}", instr):
        y, x, z, _ = m.groups()
        return [("pick", y), ("stack", x), ("pick", z), ("stack", y)]
    raise ValueError(instr)


def plan(instr, plates):
    objects = CUBES + plates
    body = {"model": args.model, "stream": False, "options": {"temperature": 0, "num_gpu": 0, "num_predict": 8000},
            "messages": [{"role": "system", "content": SYSTEM.format(schema=language.skills_schema(),
                                                                     objects=", ".join(objects))},
                         {"role": "user", "content": instr}]}
    req = urllib.request.Request(args.url + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=3000).read())
    content = r["message"]["content"]
    try:
        calls = json.loads(content[content.index("{"): content.rindex("}") + 1]).get("calls", [])
        got = [(c.get("skill"), next(iter(c.get("args", {}).values()), None)) for c in calls if c.get("skill") != "find"]
    except ValueError:
        got = None
    return got, content, r.get("eval_count")


episodes = []
for line in open(args.log, errors="ignore"):
    if m := re.match(r"\[compose\] +(\d+) '([^']*)': (SUCCESS|FAIL).*scene plates: ([^|]*)", line):
        episodes.append((int(m.group(1)), m.group(2), m.group(3) == "SUCCESS", [x.strip() for x in m.group(4).split(",")]))
results = []
for ep, instr, exec_ok, plates in episodes:
    t0 = time.time()
    got, raw, tokens = plan(instr, plates)
    ok = got == oracle(instr)
    results.append({"episode": ep, "instruction": instr, "plan": got, "plan_ok": ok, "exec_ok": exec_ok,
                    "end_to_end": ok and exec_ok, "tokens": tokens, "seconds": round(time.time() - t0, 1),
                    "raw": None if ok else raw[-500:]})
    print(f"[plan] {ep:3d} {'OK ' if ok else 'BAD'} {time.time() - t0:5.1f}s {instr} -> {got}", flush=True)
n = len(results)
print(f"[plan] {args.log}: plans correct {sum(r['plan_ok'] for r in results)}/{n}; execution (oracle plans) "
      f"{sum(r['exec_ok'] for r in results)}/{n}; end-to-end {sum(r['end_to_end'] for r in results)}/{n}")
if args.out:
    Path(args.out).write_text(json.dumps(results, indent=1))
