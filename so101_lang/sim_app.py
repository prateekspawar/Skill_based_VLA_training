"""Shared Isaac Lab launch boilerplate for the scripts in ``scripts/`` (``~/isaacsim`` venv only).

Usage::

    parser = argparse.ArgumentParser()
    parser.add_argument(...)                       # script-specific flags
    args, env_cfg = sim_app.parse(parser, task)    # adds --num_envs / launcher flags, resolves the cfg
    with sim_app.launch(env_cfg, args):
        env = sim_app.make_env(task, env_cfg)
        ...
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import gymnasium as gym

# Register our tasks before the config is resolved.
import so101_lang.envs  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def parse(parser: argparse.ArgumentParser, task: str, argv: list[str] | None = None):
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    add_launcher_args(parser)
    # Headless by default; pass --viz kit to open a window.
    parser.set_defaults(device=None)
    args, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    env_cfg, _ = resolve_task_config(task, "")
    env_cfg.scene.num_envs = args.num_envs
    args.device = env_cfg.sim.device
    return args, env_cfg


@contextlib.contextmanager
def launch(env_cfg, args):
    with launch_simulation(env_cfg, args):
        yield


def make_env(task: str, env_cfg):
    return gym.make(task, cfg=env_cfg).unwrapped
