"""Several skill models on one shared, frozen SmolVLA backbone, swapped in place (~/lerobot-env).

All skill checkpoints are fine-tuned from the same SmolVLA base with the vision encoder and VLM frozen
(``train_expert_only``), so ~350 M backbone parameters are byte-identical across them and only ~100 M differ
(action expert, state/action projections, depth encoder). One model is loaded fully; for every other skill only
its trainable tensors are kept on the GPU. Activating a skill copies those tensors into the live model
(tens of milliseconds), and switches to that skill's pre/post-processors (its own normalization statistics).

    skills = SkillModels({"pick": ckpt_a, "place": ckpt_b, "find": ckpt_c}, depth_kind="fast_foundationstereo")
    policy, pre, post = skills.activate("place")
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from safetensors import safe_open

from .policy_server import RENAME, load_policy


class SkillModels:
    def __init__(self, checkpoints: dict[str, str], depth_kind: str, calibration: str | None = None, verify: bool = True):
        names = list(checkpoints)
        first = names[0]
        self.cfg, self.policy, pre, post, self.depth = load_policy(checkpoints[first], depth_kind, calibration)
        self.params = dict(self.policy.named_parameters())
        # Skill-specific = trainable parameters (the frozen backbone is shared).
        self.skill_keys = [n for n, p in self.params.items() if p.requires_grad]
        frozen_keys = [n for n, p in self.params.items() if not p.requires_grad]
        n_skill = sum(self.params[k].numel() for k in self.skill_keys)
        n_frozen = sum(self.params[k].numel() for k in frozen_keys)
        self.procs = {first: (pre, post)}
        self.weights = {first: {k: self.params[k].detach().clone() for k in self.skill_keys}}
        for name in names[1:]:
            path = Path(checkpoints[name])
            with safe_open(str(path / "model.safetensors"), "pt", device="cuda") as f:
                if verify:  # the shared backbone must really be identical
                    bad = [k for k in frozen_keys if not torch.equal(f.get_tensor(k).to(self.params[k].dtype), self.params[k].data)]
                    if bad:
                        raise ValueError(f"{name}: {len(bad)} backbone tensors differ from {first} (e.g. {bad[0]})")
                self.weights[name] = {k: f.get_tensor(k).to(self.params[k].dtype) for k in self.skill_keys}
            cfg = PreTrainedConfig.from_pretrained(path)
            cfg.pretrained_path = str(path)
            self.procs[name] = make_pre_post_processors(
                cfg,
                pretrained_path=str(path),
                preprocessor_overrides={"device_processor": {"device": "cuda"},
                                        "rename_observations_processor": {"rename_map": RENAME}},
            )
        self.active = first
        self.last_swap_ms = 0.0
        print(f"[skills] {names}: shared backbone {n_frozen / 1e6:.0f}M params, {n_skill / 1e6:.0f}M per skill", flush=True)

    @torch.no_grad()
    def activate(self, skill: str):
        if skill not in self.weights:
            raise KeyError(f"no model for skill '{skill}' (have {list(self.weights)})")
        if skill != self.active:
            t0 = time.perf_counter()
            for k, v in self.weights[skill].items():
                self.params[k].data.copy_(v)
            torch.cuda.synchronize()
            self.last_swap_ms = 1000 * (time.perf_counter() - t0)
            self.active = skill
        pre, post = self.procs[skill]
        return self.policy, pre, post
