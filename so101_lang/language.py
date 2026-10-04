"""Object vocabulary and instruction templates shared by the sim, data export, planner and benchmark.

Pure Python (no Isaac / torch imports) so both venvs can use it.
"""

import random

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
CUBE_NAMES = list(CUBE_COLORS)
PLATE_NAMES = list(PLATE_COLORS)
NUM_CUBES = len(CUBE_NAMES)
NUM_PLATES = len(PLATE_NAMES)

# Canonical template: the planner always emits steps in this exact form, so it is what the
# executor policy must understand best.
CANONICAL_TEMPLATE = "put the {cube} cube on the {plate} plate"

# Paraphrases seen during training (canonical included).
TRAIN_TEMPLATES = [
    CANONICAL_TEMPLATE,
    "place the {cube} cube on the {plate} plate",
    "pick up the {cube} cube and put it on the {plate} plate",
    "move the {cube} block onto the {plate} plate",
]

# Paraphrases never seen in training, used by the "reworded instruction" benchmark suite.
HELDOUT_TEMPLATES = [
    "could you set the {cube} block down on the {plate} dish",
    "grab the {cube} one and drop it onto the {plate} plate",
    "the {cube} cube belongs on the {plate} plate",
]

# (cube, plate) pairs held out of training for the "new combination" benchmark suite.
HELDOUT_PAIRS = {("yellow", "pink"), ("green", "white")}


def instruction(cube_idx: int, plate_idx: int, template: str = CANONICAL_TEMPLATE) -> str:
    return template.format(cube=CUBE_NAMES[cube_idx], plate=PLATE_NAMES[plate_idx])


def random_train_instruction(cube_idx: int, plate_idx: int, rng: random.Random | None = None) -> str:
    rng = rng or random
    return instruction(cube_idx, plate_idx, rng.choice(TRAIN_TEMPLATES))


def train_pairs() -> list[tuple[int, int]]:
    return [
        (c, p)
        for c in range(NUM_CUBES)
        for p in range(NUM_PLATES)
        if (CUBE_NAMES[c], PLATE_NAMES[p]) not in HELDOUT_PAIRS
    ]


def heldout_pairs() -> list[tuple[int, int]]:
    return [(CUBE_NAMES.index(c), PLATE_NAMES.index(p)) for c, p in sorted(HELDOUT_PAIRS)]
