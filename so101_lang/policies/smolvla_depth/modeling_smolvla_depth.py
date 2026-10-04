"""SmolVLA + native depth tokens.

A small conv encoder turns the (3, 64, 64) head-frame point map (plus a validity channel) into an 8x8 grid
of tokens in the VLM hidden size. They are inserted into the prefix right before the state token (same
bidirectional attention block as image/language tokens) through a zero-initialized gate, so a freshly
initialized branch leaves the pretrained SmolVLA behaviour unchanged and training grows its influence.
"""

from __future__ import annotations

import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching

from .configuration_smolvla_depth import SmolVLADepthConfig


class DepthTokenEncoder(nn.Module):
    def __init__(self, hidden: int, channels: int, grid: int, size: int, scale: float):
        super().__init__()
        self.scale, self.grid = scale, grid
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(4, c // 4, 3, stride=2, padding=1), nn.GroupNorm(8, c // 4), nn.GELU(),
            nn.Conv2d(c // 4, c // 2, 3, stride=2, padding=1), nn.GroupNorm(8, c // 2), nn.GELU(),
            nn.Conv2d(c // 2, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU(),
            nn.AdaptiveAvgPool2d(grid),
        )
        self.proj = nn.Linear(c, hidden)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.gate = nn.Parameter(torch.zeros(hidden))  # zero-init: branch starts switched off

    def forward(self, pointmap: torch.Tensor) -> torch.Tensor:
        valid = (pointmap.abs().sum(1, keepdim=True) > 0).to(pointmap.dtype)
        x = torch.cat([pointmap / self.scale, valid], dim=1)
        feats = self.net(x).flatten(2).transpose(1, 2)  # (B, grid*grid, C)
        return (self.proj(feats) + self.pos) * self.gate


class VLAFlowMatchingDepth(VLAFlowMatching):
    """embed_prefix appends depth tokens computed from ``self._pointmap`` (set by the policy per call)."""

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        pointmap = getattr(self, "_pointmap", None)
        if pointmap is None:
            return embs, pad_masks, att_masks
        tok = self.depth_encoder(pointmap.to(device=embs.device, dtype=torch.float32)).to(embs.dtype)
        n = tok.shape[1]
        # Insert before the trailing state token, inside the bidirectional (att=0) block.
        embs = torch.cat([embs[:, :-1], tok, embs[:, -1:]], dim=1)
        ones = torch.ones(pad_masks.shape[0], n, dtype=pad_masks.dtype, device=pad_masks.device)
        pad_masks = torch.cat([pad_masks[:, :-1], ones, pad_masks[:, -1:]], dim=1)
        zeros = torch.zeros(att_masks.shape[0], n, dtype=att_masks.dtype, device=att_masks.device)
        att_masks = torch.cat([att_masks[:, :-1], zeros, att_masks[:, -1:]], dim=1)
        return embs, pad_masks, att_masks


class SmolVLADepthPolicy(SmolVLAPolicy):
    config_class = SmolVLADepthConfig
    name = "smolvla_depth"

    def __init__(self, config: SmolVLADepthConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model.__class__ = VLAFlowMatchingDepth
        hidden = self.model.vlm_with_expert.config.text_config.hidden_size
        self.model.depth_encoder = DepthTokenEncoder(
            hidden, config.depth_channels, config.depth_token_grid, config.pointmap_size, config.pointmap_scale
        )

    def _set_pointmap(self, batch):
        pm = batch.get(self.config.pointmap_key)
        if pm is not None:
            if pm.ndim == 3:  # single unbatched observation
                pm = pm.unsqueeze(0)
            elif pm.ndim == 5:  # (B, T, 3, H, W) observation history -> latest frame
                pm = pm[:, -1]
        self.model._pointmap = pm

    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        self._set_pointmap(batch)
        return super().forward(batch, noise=noise, time=time, reduction=reduction)

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        self._set_pointmap(batch)
        return super()._get_action_chunk(batch, noise, **kwargs)
