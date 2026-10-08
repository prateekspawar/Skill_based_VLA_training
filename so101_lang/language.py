"""Object vocabulary and the robot skill registry, shared by the sim, data export, planner and benchmark.

Pure Python (no Isaac / torch imports) so both venvs can use it.

The planner LLM may only emit calls to the skills in :data:`SKILLS`. Each call is rendered into a
short natural-language step (``skill_text``) which is what the VLA is trained and prompted with, so
one VLA learns every skill and the planner composes them.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field

# Slot order matters: cube ``i`` in the scene is ``CUBE_COLORS[i]``, plate ``j`` is ``PLATE_COLORS[j]``.
CUBE_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.08, 0.08),
    "green": (0.10, 0.65, 0.15),
    "blue": (0.08, 0.25, 0.85),
    "yellow": (0.95, 0.80, 0.05),
}
PLATE_COLORS: dict[str, tuple[float, float, float]] = {
    "white": (0.92, 0.92, 0.92),
    "pink": (1.0, 0.45, 0.70),
}
# Evaluation override of the plate colours, per slot (e.g. SO101_PLATE_COLORS=red,blue -> slot 0 red, slot 1
# blue). Layouts, tasks (by slot) and everything else stay the same, so runs compare one-to-one.
_PALETTE = {**CUBE_COLORS, **PLATE_COLORS, "orange": (1.0, 0.45, 0.05), "purple": (0.50, 0.15, 0.75),
            "brown": (0.45, 0.25, 0.10)}
# Plate colours sampled per episode when plate-colour randomisation is on (env cfg / SO101_RANDOM_PLATE_COLORS=1):
# includes the cube colours on purpose, so policies rely on the target point and shape rather than colour.
# Black / gray are left out (the table is dark gray).
PLATE_PALETTE = ["white", "pink", "red", "blue", "green", "yellow", "orange", "purple", "brown"]
if os.environ.get("SO101_PLATE_COLORS"):
    PLATE_COLORS = {c: _PALETTE[c] for c in os.environ["SO101_PLATE_COLORS"].split(",")}
CUBE_NAMES = list(CUBE_COLORS)
PLATE_NAMES = list(PLATE_COLORS)
NUM_CUBES = len(CUBE_NAMES)
NUM_PLATES = len(PLATE_NAMES)

# Rotational symmetry about the vertical axis [deg] per object kind. Target yaws are reported modulo this period
# (perception cannot tell symmetric orientations apart): cubes 90, plates 360 (round: yaw is always 0).
# An asymmetric object would get 360, i.e. its full unbounded orientation.
YAW_SYMMETRY_DEG = {"cube": 90.0, "plate": 360.0}


def fold_yaw(yaw: float, kind: str) -> float:
    """Fold a yaw [rad] into [-period/2, period/2) for the object kind's symmetry period."""
    period = math.radians(YAW_SYMMETRY_DEG[kind])
    return (yaw + period / 2) % period - period / 2


# Object references the planner may use as skill arguments.
OBJECT_REFS = [f"{c} cube" for c in CUBE_NAMES]
TARGET_REFS = [f"{p} plate" for p in PLATE_NAMES]
ALL_TARGET_REFS = list(dict.fromkeys(TARGET_REFS + [f"{c} plate" for c in PLATE_PALETTE]))


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    params: dict[str, list[str]]  # parameter name -> allowed values
    canonical: str  # template the executor uses at run time
    train_templates: list[str] = field(default_factory=list)  # paraphrases seen in training (canonical included)
    heldout_templates: list[str] = field(default_factory=list)  # paraphrases reserved for the benchmark


SKILLS: dict[str, Skill] = {
    "pick": Skill(
        name="pick",
        description="Grasp the given object from the table and lift it. The gripper must be empty.",
        params={"object": OBJECT_REFS},
        canonical="pick up the {object}",
        train_templates=["pick up the {object}", "grab the {object}", "grasp the {object} and lift it"],
        heldout_templates=["get hold of the {object}", "lift the {object} off the table"],
    ),
    "place": Skill(
        name="place",
        description="Put the object currently held in the gripper onto the given target and release it.",
        params={"target": ALL_TARGET_REFS},
        canonical="place it on the {target}",
        train_templates=["place it on the {target}", "put it down on the {target}", "set it on the {target}"],
        heldout_templates=["drop it onto the {target}", "release it over the {target}"],
    ),
    "find": Skill(
        name="find",
        description="Move the arm until the camera has a clear view of the given object, then report its 9-D pose "
        "(position + 6-D rotation, head frame). Run once per object before pick / place.",
        params={"object": OBJECT_REFS + ALL_TARGET_REFS},
        canonical="find the {object}",
        train_templates=["find the {object}", "look for the {object}", "locate the {object}"],
        heldout_templates=["where is the {object}", "search for the {object}"],
    ),
}

# (cube, plate) pairs never sampled as a training episode; used by the "new combination" benchmark suite.
# Defined by slot so a plate-colour override keeps the same task sampling: (yellow, plate 1), (green, plate 0).
_HELDOUT_SLOTS = {(3, 1), (1, 0)}
HELDOUT_PAIRS = {(CUBE_NAMES[c], PLATE_NAMES[p]) for c, p in _HELDOUT_SLOTS}


def skill_text(skill: str, template: str | None = None, **args: str) -> str:
    """Render a skill call such as ``skill_text("pick", object="red cube")`` into the VLA step text."""
    s = SKILLS[skill]
    if set(args) != set(s.params):
        raise ValueError(f"{skill} expects {sorted(s.params)}, got {sorted(args)}")
    for k, v in args.items():
        if v not in s.params[k]:
            raise ValueError(f"{skill}: {k}={v!r} not in {s.params[k]}")
    return (template or s.canonical).format(**args)


def random_skill_text(skill: str, rng: random.Random | None = None, heldout: bool = False, **args: str) -> str:
    rng = rng or random
    s = SKILLS[skill]
    return skill_text(skill, rng.choice(s.heldout_templates if heldout else s.train_templates), **args)


def skills_schema() -> str:
    """JSON description of the callable skills, given to the planner LLM as its function list."""
    return json.dumps(
        [
            {"name": s.name, "description": s.description, "parameters": {k: {"enum": v} for k, v in s.params.items()}}
            for s in SKILLS.values()
        ],
        indent=2,
    )


def cube_ref(cube_idx: int) -> str:
    return OBJECT_REFS[cube_idx]


def plate_ref(plate_idx: int, colors=None) -> str:
    """Name of plate slot ``plate_idx``; ``colors`` = this episode's plate colour names when randomised."""
    return f"{colors[plate_idx]} plate" if colors is not None else TARGET_REFS[plate_idx]


def goal_instruction(cube_idx: int, plate_idx: int, colors=None) -> str:
    """High-level user instruction for one pick-and-place episode (what the planner receives)."""
    return f"put the {cube_ref(cube_idx)} on the {plate_ref(plate_idx, colors)}"


def train_pairs() -> list[tuple[int, int]]:
    return [
        (c, p)
        for c in range(NUM_CUBES)
        for p in range(NUM_PLATES)
        if (CUBE_NAMES[c], PLATE_NAMES[p]) not in HELDOUT_PAIRS
    ]


def heldout_pairs() -> list[tuple[int, int]]:
    return [(CUBE_NAMES.index(c), PLATE_NAMES.index(p)) for c, p in sorted(HELDOUT_PAIRS)]
