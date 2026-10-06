"""Skill verdict labels: two extra action dims ``[done, success]`` per frame.

    running   [0, 0]
    succeeded [1, 1]   the last SUCCESS_TAIL frames of a skill segment that achieved its goal
    failed    [1, 0]   from the moment the failure is detected (the robot then holds still)

Successful demonstrations (gen_demos.py, find successes) carry no labels; :func:`default_status` derives them.
Failure demonstrations store their own ``status`` dataset (gen_fail_demos.py, gen_find_demos.py --absent_prob).
"""

from __future__ import annotations

import numpy as np

STATUS_NAMES = ["status.done", "status.success"]
SUCCESS_TAIL = 5


def default_status(skill: np.ndarray, tail: int = SUCCESS_TAIL) -> np.ndarray:
    """Status of a fully successful episode: the last ``tail`` frames of every contiguous skill segment -> [1, 1]."""
    skill = np.asarray(skill).reshape(-1)
    status = np.zeros((len(skill), 2), dtype=np.float32)
    ends = np.nonzero(np.r_[skill[1:] != skill[:-1], True])[0]  # last frame of each segment
    starts = np.r_[0, ends[:-1] + 1]
    for s, e in zip(starts, ends):
        status[max(e + 1 - tail, s) : e + 1] = 1.0
    return status
