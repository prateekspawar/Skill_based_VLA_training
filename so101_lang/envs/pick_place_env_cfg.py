"""SO-101 language-conditioned pick-and-place environment configs.

Reuses the seated SO-101, the soft gripper actuator and the absolute-pose IK action from
Isaac Lab's ``IsaacContrib-Stack-Cube-SO101-IK-Abs-v0`` config, and swaps in a scene with
four coloured cubes, two coloured plates, a front camera and a wrist camera.

Two action variants share everything else:

* :class:`SO101PickPlaceIkEnvCfg` - ``[pos_xyz, quat_xyzw, gripper_closedness]`` in the robot base
  frame; driven by the scripted expert.
* :class:`SO101PickPlaceJointEnvCfg` - absolute joint-position targets for the 6 joints (radians);
  what the learned policies output, matching LeRobot's SO-101 action space.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import numpy as np

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab_tasks.contrib.stack.config.so101 import stack_ik_abs_env_cfg

from so101_lang import language

from . import mdp

##
# Geometry [m]. The table top is at world z ~= 0 and the robot base sits at the env origin
# facing +x; the reachable workspace is roughly x in [0.12, 0.32], |y| < 0.15.
##
CUBE_SIZE = 0.03
CUBE_Z = CUBE_SIZE / 2 + 0.001
PLATE_RADIUS = 0.045
PLATE_HEIGHT = 0.008
PLATE_Z = PLATE_HEIGHT / 2 + 0.001
PLATE_TOP_Z = PLATE_HEIGHT + 0.001
WORKSPACE_X = (0.13, 0.30)
WORKSPACE_Y = (-0.13, 0.13)

CAMERA_RES = 256


def look_at_world_quat(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, ...]:
    """XYZW quaternion that points a ``convention="world"`` camera (forward +X, up +Z) from eye to target."""
    fwd = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    fwd /= np.linalg.norm(fwd)
    yaw = math.atan2(fwd[1], fwd[0])
    pitch = -math.asin(fwd[2])  # +pitch about Y tilts +X down
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    # q = Rz(yaw) * Ry(pitch)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


FRONT_CAM_EYE = (0.62, 0.0, 0.40)
FRONT_CAM_TARGET = (0.20, 0.0, 0.0)

_OBJECT_RIGID_PROPS = [
    PhysxRigidBodyCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
]


def _cube_cfg(i: int) -> RigidObjectCfg:
    color_name = language.CUBE_NAMES[i]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Cube_{i}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15 + 0.04 * i, 0.3, CUBE_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=language.CUBE_COLORS[color_name], roughness=0.6
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.2, dynamic_friction=1.0),
            rigid_props=_OBJECT_RIGID_PROPS,
            mass_props=sim_utils.MassCfg(mass=0.02),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            semantic_tags=[("class", f"{color_name}_cube")],
        ),
    )


def _plate_cfg(j: int) -> RigidObjectCfg:
    color_name = language.PLATE_NAMES[j]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Plate_{j}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15 + 0.1 * j, -0.3, PLATE_Z)),
        spawn=sim_utils.CylinderCfg(
            radius=PLATE_RADIUS,
            height=PLATE_HEIGHT,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=language.PLATE_COLORS[color_name], roughness=0.4
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=0.9),
            # Kinematic: plates stay put but can be teleported on reset.
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
            mass_props=sim_utils.MassCfg(mass=0.2),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            semantic_tags=[("class", f"{color_name}_plate")],
        ),
    )


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Table, SO-101, four cubes, two plates, front + wrist cameras."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    cube_0 = _cube_cfg(0)
    cube_1 = _cube_cfg(1)
    cube_2 = _cube_cfg(2)
    cube_3 = _cube_cfg(3)
    plate_0 = _plate_cfg(0)
    plate_1 = _plate_cfg(1)

    front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        update_period=0.0,
        height=CAMERA_RES,
        width=CAMERA_RES,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 3.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=FRONT_CAM_EYE, rot=look_at_world_quat(FRONT_CAM_EYE, FRONT_CAM_TARGET), convention="world"
        ),
    )
    # At the lens of the stock SO-101 wrist camera mount (gripper +Y side), looking along the jaws (gripper -Z) tilted 25 deg toward them.
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_cam",
        update_period=0.0,
        height=CAMERA_RES,
        width=CAMERA_RES,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.002, 0.056, -0.066), rot=(0.9763, 0.0, 0.0, 0.2164), convention="ros"),
    )


@configclass
class EventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_layout_and_task = EventTerm(
        func=mdp.reset_layout_and_task,
        mode="reset",
        params={
            "x_range": WORKSPACE_X,
            "y_range": WORKSPACE_Y,
            "cube_radius": CUBE_SIZE * 0.72,
            "plate_radius": PLATE_RADIUS,
            "cube_z": CUBE_Z,
            "plate_z": PLATE_Z,
            "pairs": language.train_pairs(),
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception + task id. Images are read straight from the camera sensors by the scripts."""

        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_vel = ObsTerm(func=base_mdp.joint_vel)
        task = ObsTerm(func=mdp.task_onehot)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Ground-truth object poses (robot base frame) for the scripted expert and metrics."""

        objects = ObsTerm(func=mdp.object_poses_b)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(func=mdp.any_cube_dropped)
    success = DoneTerm(
        func=mdp.target_on_plate,
        params={"plate_radius": PLATE_RADIUS, "plate_top_z": PLATE_TOP_Z, "cube_half": CUBE_SIZE / 2},
    )


@configclass
class SO101PickPlaceIkEnvCfg(stack_ik_abs_env_cfg.SO101CubeStackEnvCfg):
    """Pick-and-place with absolute end-effector pose actions (scripted expert)."""

    def __post_init__(self):
        super().__post_init__()
        # Keep the parent's seated robot, gripper actuator and EE frame; replace everything else.
        parent_scene = self.scene
        self.scene = PickPlaceSceneCfg(num_envs=parent_scene.num_envs, env_spacing=2.5, replicate_physics=True)
        self.scene.robot = parent_scene.robot
        self.scene.ee_frame = parent_scene.ee_frame
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        self.events = EventsCfg()
        self.observations = ObservationsCfg()
        self.terminations = TerminationsCfg()
        self.isaac_teleop = None

        self.episode_length_s = 20.0
        # Re-render after reset so the first camera frame of an episode shows the new layout.
        self.num_rerenders_on_reset = 2

        # Viewer looks at the workspace from the front-right.
        self.viewer.eye = (0.75, 0.45, 0.45)
        self.viewer.lookat = (0.2, 0.0, 0.05)


@configclass
class JointActionsCfg:
    arm_action: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    gripper_action: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
    )


@configclass
class SO101PickPlaceJointEnvCfg(SO101PickPlaceIkEnvCfg):
    """Pick-and-place with absolute joint-position actions (learned policies)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = JointActionsCfg()
