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

    def validate_features(self) -> None:
        super().validate_features()
        # The point map is consumed by the depth branch with fixed metric scaling, so keep it out of the
        # generic STATE normalization (which would otherwise normalize it per pixel).
        self.input_features.pop(self.pointmap_key, None)
