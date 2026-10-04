"""SmolVLA with a native depth branch: a metric XYZ point map (head frame) becomes extra prefix tokens."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

POINTMAP_KEY = "observation.pointmap"


@PreTrainedConfig.register_subclass("smolvla_depth")
@dataclass
class SmolVLADepthConfig(SmolVLAConfig):
    pointmap_key: str = POINTMAP_KEY
    pointmap_size: int = 64  # input grid (H = W)
    depth_channels: int = 256  # encoder width
    depth_token_grid: int = 8  # tokens = grid * grid
    pointmap_scale: float = 1.0  # metres per unit fed to the encoder
    use_depth_branch: bool = True  # False: plain SmolVLA inputs (for no-depth baselines with the options below)
    # Learn the first N action dims as offsets from the current state (both normalized), i.e. predict
    # a_norm - s_norm and add s_norm back at inference. Exactly invertible; 9 = TCP position + rot6d.
    relative_action_dims: int = 0

    def validate_features(self) -> None:
        super().validate_features()
        # The point map is consumed by the depth branch with fixed metric scaling, so keep it out of the
        # generic STATE normalization (which would otherwise normalize it per pixel).
        self.input_features.pop(self.pointmap_key, None)
        # Note: with --policy.path the declared state shape comes from the base checkpoint (6-D) while the
        # real 10-D state is padded to max_state_dim, so relative_action_dims is checked at runtime instead.
