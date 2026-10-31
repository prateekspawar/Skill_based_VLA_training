"""Planner LLM: user instruction -> validated skill calls for the objects in the scene.

The LLM (Ollama, no training) only says what the instruction wants, as goals "cube X on Y" (Y a plate, another cube,
or "gripper" to just hold it), in the order the user asked. The reply is constrained to a JSON schema whose fields
are enums of the scene's objects, so it cannot name objects that are not there; "table" and "unspecified" are allowed
so it can say so, and are always refused. :func:`goals_to_calls` then orders the goals (a cube's support first, the
held cube first) and turns them into ``pick`` / ``place`` / ``stack`` calls, which :func:`validate` checks against
the skill preconditions. An invalid answer is sent back once with the error. Instructions the skills cannot do (other
actions, objects not in the scene, ambiguous references) come back as ``Plan.error``.

``find`` is not planned: the executor runs it before every ``pick`` / ``place`` / ``stack`` (see :func:`with_finds`).
Standard library only, so it runs in both Python environments.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field

from . import language

PLANNED = ("pick", "place", "stack")
GRIPPER = "gripper"  # goal "on": pick the cube up and keep holding it
NOT_OBJECTS = ("table", "unspecified")  # allowed in the reply so the LLM can say it, always refused
ARG = {"pick": "object", "place": "target", "stack": "object"}  # the one parameter of each planned skill
DEFAULT_MODEL = os.environ.get("SO101_PLANNER_MODEL", "qwen3-vl:8b-instruct")
DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

SYSTEM = """You read instructions for a one-arm robot with a two-finger gripper and say what they want as goals.
The robot can only do this, with these skills:
{skills}
So every goal is one cube put on one plate or on top of another cube, or picked up and held ("on": "gripper").
Rules:
- Give one goal per cube the instruction wants moved, in the order the user asked for. Cubes that are not mentioned
  stay where they are.
- Words like block, box or brick mean a cube; dish, tray or saucer mean a plate. Match colours to the scene's objects.
- "X on top of Y", "X on Y", "Y at the bottom, X above" all mean the goal {{"cube": X, "on": Y}}.
- For "each" / "every" / "all" instructions, give goals only for the cubes the condition applies to in this scene
  (for example, cubes without a plate of their own colour stay where they are).
- If the instruction needs an action these skills cannot do (for example push, rotate, hand over), a place that is not
  a plate or a cube (use "table"), an object that is not in the scene, or is ambiguous (for example "the plate" when
  there are two plates: use "unspecified"), set "possible" to false and explain why in "reason".
For each goal also copy the instruction's own words for the cube and for where it goes ("cube_words", "on_words").
Reply only with JSON: {{"possible": true or false, "reason": "...", "goals": [{{"cube_words": ..., "cube": ..., "on_words": ..., "on": ...}}]}}.

Examples (scene: red, green, blue, yellow cube; pink plate, green plate):
"put the red block on the pink plate" -> {{"possible": true, "reason": "", "goals": [{{"cube_words": "the red block", "cube": "red cube", "on_words": "the pink plate", "on": "pink plate"}}]}}
"stack the yellow cube on the blue cube" -> {{"possible": true, "reason": "", "goals": [{{"cube_words": "the yellow cube", "cube": "yellow cube", "on_words": "the blue cube", "on": "blue cube"}}]}}
"pick up the green cube" -> {{"possible": true, "reason": "", "goals": [{{"cube_words": "the green cube", "cube": "green cube", "on_words": "", "on": "gripper"}}]}}
"push the red cube to the left" -> {{"possible": false, "reason": "pushing is not a skill", "goals": []}}"""


@dataclass
class Plan:
    calls: list[tuple[str, str]]  # (skill, argument), e.g. ("pick", "red cube")
    error: str | None = None  # why the instruction cannot be planned (calls is then empty)
    goals: list[tuple[str, str]] = field(default_factory=list)  # (cube, on) as the LLM gave them
    raw: list[str] = field(default_factory=list)  # LLM replies, one per attempt
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def skills_text() -> str:
    return "\n".join(f"- {n}({ARG[n]}): {language.SKILLS[n].description}" for n in PLANNED)


def response_schema(cubes: list[str], plates: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "possible": {"type": "boolean"},
            "reason": {"type": "string"},
            "goals": {"type": "array", "items": {"type": "object", "required": ["cube_words", "cube", "on_words", "on"],
                                                  "properties": {
                "cube_words": {"type": "string"},
                "cube": {"enum": list(cubes) + ["unspecified"]},
                "on_words": {"type": "string"},
                "on": {"enum": list(plates) + list(cubes) + [GRIPPER, *NOT_OBJECTS]}}}},
        },
        "required": ["possible", "reason", "goals"],
    }


GENERIC = {"cube", "cubes", "block", "box", "brick", "plate", "dish", "tray", "saucer", "one", "it", "thing", "object"}


def ambiguous(words: str, name: str, cubes: list[str], plates: list[str], holding: str | None) -> bool:
    """True if ``words`` (the instruction's words for ``name``) are only a generic noun like "the plate" while the
    scene has several objects of that kind (the LLM then picked one by guessing). A pronoun ("it") counts too: the
    LLM resolves it unreliably (held-out set: "... and stack the blue cube on it" -> on the plate, not the cube),
    so refusing is safer than guessing."""
    tokens = [t for t in words.lower().replace(",", " ").split() if t not in ("the", "a", "an", "that", "this")]
    if not tokens or any(t not in GENERIC for t in tokens):
        return False  # a colour, a quantifier ("both", "all") or a description: not just a bare noun
    if name == holding and tokens in (["it"], ["one"]):
        return False  # "it" = the cube in the gripper
    return len(cubes if name in cubes else plates) > 1


def goals_to_calls(goals: list[tuple[str, str]], cubes: list[str], plates: list[str],
                   holding: str | None = None) -> tuple[list[tuple[str, str]], str | None]:
    """Order the goals and turn them into skill calls. Returns (calls, None) or ([], why not)."""
    for cube, on in goals:
        if cube in NOT_OBJECTS or on in NOT_OBJECTS:
            return [], f"{cube} on {on}: {'the table is not a target' if 'table' in (cube, on) else 'ambiguous'}"
        if cube not in cubes or (on not in plates and on not in cubes and on != GRIPPER):
            return [], f"{cube} on {on}: not in the scene"
        if on == cube:
            return [], f"{cube} cannot go on itself"
    moved = [c for c, _ in goals]
    if len(set(moved)) != len(moved):
        return [], "a cube has two goals"
    if sum(on == GRIPPER for _, on in goals) > 1:
        return [], "only one cube can be held"
    if any(on in moved and dict(goals)[on] == GRIPPER for _, on in goals):
        return [], "cannot stack on a cube that stays in the gripper"
    if holding is not None and holding not in moved:
        return [], f"the gripper holds the {holding} and the instruction does not say where to put it"
    # Order: the held cube first (it must be put down before anything else), then repeatedly the first goal (in the
    # user's order) whose support is a plate or a cube that is already in its final place.
    todo, done, ordered = list(goals), set(), []
    if holding is not None:
        i = moved.index(holding)
        if goals[i][1] in moved:
            return [], f"the held {holding} must go on a cube that has to be moved first"
        ordered.append(todo.pop(i))
        done.add(holding)
    while todo:
        ready = [g for g in todo if (g[1] not in moved or g[1] in done) and (g[1] != GRIPPER or len(todo) == 1)]
        if not ready:
            return [], "the goals form a cycle"
        ordered.append(ready[0])
        todo.remove(ready[0])
        done.add(ready[0][0])
    calls = []
    for cube, on in ordered:
        if cube != holding:
            calls.append(("pick", cube))
        if on != GRIPPER:
            calls.append(("place" if on in plates else "stack", on))
    return calls, validate(calls, cubes, plates, holding)


def validate(calls: list[tuple[str, str]], cubes: list[str], plates: list[str], holding: str | None = None) -> str | None:
    """None if the calls can be executed in order from the given gripper state, else the first problem."""
    held = holding
    for i, (skill, arg) in enumerate(calls, 1):
        if skill == "pick":
            if arg not in cubes:
                return f"call {i}: pick needs a cube in the scene, got {arg!r}"
            if held is not None:
                return f"call {i}: pick {arg} while still holding the {held}"
            held = arg
        elif skill in ("place", "stack"):
            if held is None:
                return f"call {i}: {skill} with an empty gripper"
            if skill == "place" and arg not in plates:
                return f"call {i}: place needs a plate in the scene, got {arg!r}"
            if skill == "stack" and (arg not in cubes or arg == held):
                return f"call {i}: stack needs another cube in the scene, got {arg!r} while holding the {held}"
            held = None
        else:
            return f"call {i}: unknown skill {skill!r}"
    return None


def with_finds(calls: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    """Executor steps: each planned call as (skill, kwargs for language.skill_text), a find of its object first."""
    steps = []
    for skill, arg in calls:
        steps.append(("find", {"object": arg}))
        steps.append((skill, {ARG[skill]: arg}))
    return steps


def next_goal(goals: list[tuple], met: list[bool], tries: list[int], max_tries: int) -> tuple[int | None, str | None]:
    """Re-planning: which goal to run next, given which goals the scene now meets.

    ``goals`` are (cube, support, ...) in plan order (a support's own goal before the goals that stack on it);
    ``met`` says which hold in the scene now, ``tries`` how often each has been run. The LLM is not asked again:
    the goals do not change when a step fails, only which are left. Returns (index, None), (None, None) when every
    goal is met, or (None, reason) when the executor should stop.
    """
    if all(met):
        return None, None
    unmet_cubes = {g[0] for g, m in zip(goals, met) if not m}
    failed = [i for i, m in enumerate(met) if not m and tries[i] >= max_tries]
    for i, (g, m) in enumerate(zip(goals, met)):
        if m or g[1] in unmet_cubes or i in failed:  # met, support cube not in its final place yet, or given up
            continue
        return i, None
    if failed:
        return None, f"goal {failed[0]} not met after {tries[failed[0]]} attempts"
    return None, "no unmet goal has its support in place"


class Planner:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, gpu: bool = True, retries: int = 1,
                 timeout: float = 600.0):
        self.model, self.url, self.gpu, self.retries, self.timeout = model, url, gpu, retries, timeout
        # Ollama cannot constrain the reply of a model that always reasons first (its reasoning is cut off and the
        # answer comes back empty): for those the schema is only described in the prompt and the JSON is parsed.
        req = urllib.request.Request(url + "/api/show", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        self.thinking = "thinking" in json.loads(urllib.request.urlopen(req, timeout=30).read()).get("capabilities", [])

    def _chat(self, messages: list[dict], schema: dict) -> str:
        body = {"model": self.model, "stream": False, "think": False, "messages": messages,
                **({} if self.thinking else {"format": schema}),
                "options": {"temperature": 0, "num_predict": 8000, "num_ctx": 16384, **({} if self.gpu else {"num_gpu": 0})}}
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        msg = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())["message"]
        return msg.get("content") or ""

    def plan(self, instruction: str, cubes: list[str], plates: list[str], holding: str | None = None) -> Plan:
        """Skill calls for ``instruction`` in a scene with these cubes and plates (names as in language.SKILLS)."""
        t0 = time.time()
        scene = (f"Scene: cubes: {', '.join(cubes) or 'none'}; plates: {', '.join(plates) or 'none'}. "
                 f"The gripper holds: {'the ' + holding if holding else 'nothing'}.")
        messages = [{"role": "system", "content": SYSTEM.format(skills=skills_text())},
                    {"role": "user", "content": f"{scene}\nInstruction: {instruction}"}]
        raw, error, goals = [], "no reply", []
        for _ in range(1 + self.retries):
            content = self._chat(messages, response_schema(cubes, plates))
            raw.append(content)
            try:
                reply = json.loads(content[content.index("{"): content.rindex("}") + 1])
                goals = [(g["cube"], g["on"]) for g in reply.get("goals", [])]
                words = [(g.get("cube_words", ""), g["cube"], g.get("on_words", ""), g["on"]) for g in reply.get("goals", [])]
            except (ValueError, KeyError, TypeError) as e:
                error = f"unreadable reply ({e})"
            else:
                if not reply.get("possible", True):
                    return Plan([], reply.get("reason") or "not possible", goals, raw, time.time() - t0)
                if not goals:
                    error = "no goals"
                else:
                    vague = [w for cw, c, ow, o in words for w, n in ((cw, c), (ow, o))
                             if n in cubes + plates and ambiguous(w, n, cubes, plates, holding)]
                    if vague:
                        return Plan([], f"ambiguous: {vague[0]!r} could be several objects", goals, raw, time.time() - t0)
                    calls, error = goals_to_calls(goals, cubes, plates, holding)
                    if error is None:
                        return Plan(calls, None, goals, raw, time.time() - t0)
                    if any(x in NOT_OBJECTS for g in goals for x in g):  # the LLM itself said it cannot be done
                        return Plan([], error, goals, raw, time.time() - t0)
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"That answer is invalid: {error}. Reply again."}]
        return Plan([], f"invalid plan: {error}", goals, raw, time.time() - t0)
