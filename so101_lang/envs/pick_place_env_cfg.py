"""SO-101 language-conditioned pick-and-place environment configs.

Reuses the seated SO-101, the soft gripper actuator and the absolute-pose IK action from
Isaac Lab's ``IsaacContrib-Stack-Cube-SO101-IK-Abs-v0`` config, and swaps in a scene with
four coloured cubes, two coloured plates and a stereo camera "head".

Two action variants share everything else:

* :class:`SO101PickPlaceIkEnvCfg` - ``[pos_xyz, quat_xyzw, gripper_closedness]`` in the robot base
  frame; driven by the scripted expert.
* :class:`SO101PickPlaceJointEnvCfg` - absolute joint-position targets for the 6 joints (radians);
  what the learned policies output, matching LeRobot's SO-101 action space.
"""

from __future__ import annotations

import os

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

CAMERA_RES = 512


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


# Stereo "head": two parallel pinhole cameras on one rigid mount, placed like a human head relative to
# its right arm -- behind, above and to the left of the arm's shoulder, looking down at the workspace.
HEAD_CENTER = (-0.12, 0.20, 0.45)  # midpoint between the two optical centres, env-local [m]
HEAD_TARGET = (0.21, -0.01, 0.0)  # point the head looks at
STEREO_BASELINE = 0.06  # [m], typical of compact stereo cameras (ZED Mini 63 mm, D435 50 mm)
STEREO_FOCAL_LENGTH = 32.0  # [mm]
STEREO_APERTURE = 20.955  # horizontal aperture [mm] -> ~36 deg horizontal FOV


def stereo_head_poses(center, target, baseline):
    """Left/right camera positions and their shared world-convention orientation (xyzw)."""
    quat = look_at_world_quat(center, target)
    fwd = np.asarray(target, dtype=float) - np.asarray(center, dtype=float)
    yaw = math.atan2(fwd[1], fwd[0])
    left_axis = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # camera +Y in world convention
    c = np.asarray(center, dtype=float)
    left = tuple((c + left_axis * baseline / 2).tolist())
    right = tuple((c - left_axis * baseline / 2).tolist())
    return left, right, quat


LEFT_CAM_POS, RIGHT_CAM_POS, HEAD_QUAT = stereo_head_poses(HEAD_CENTER, HEAD_TARGET, STEREO_BASELINE)


def _quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _matrix_to_quat_xyzw(r: np.ndarray) -> tuple[float, ...]:
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return (x, y, z, w)


def rendered_head_pose():
    """Where the stereo cameras are actually rendered: the training head pose moved by SO101_CAM_OFFSET
    (so101_lang.robots.virtual_head). Returns (left_pos, right_pos, quat_xyzw, head_in_env 4x4)."""
    from so101_lang.robots.virtual_head import real_in_virtual

    h = np.eye(4)
    h[:3, :3], h[:3, 3] = _quat_xyzw_to_matrix(HEAD_QUAT), HEAD_CENTER
    real = h @ real_in_virtual()
    half = real[:3, :3] @ np.array([0.0, STEREO_BASELINE / 2, 0.0])
    return (tuple((real[:3, 3] + half).tolist()), tuple((real[:3, 3] - half).tolist()),
            _matrix_to_quat_xyzw(real[:3, :3]), real)


RENDER_LEFT_CAM_POS, RENDER_RIGHT_CAM_POS, RENDER_HEAD_QUAT, _ = rendered_head_pose()


def _stereo_cam_cfg(name: str, pos) -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/StereoHead/{name}",
        update_period=0.0,
        height=CAMERA_RES,
        width=CAMERA_RES,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=STEREO_FOCAL_LENGTH, horizontal_aperture=STEREO_APERTURE, clipping_range=(0.05, 3.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=RENDER_HEAD_QUAT, convention="world"),
    )


def stereo_head_rig_cfgs() -> dict[str, AssetBaseCfg]:
    """Visual model of the stereo head for demos (SO101_SHOW_HEAD=1): a plate with two camera boards and lenses on a
    ball head and a post standing on the table, like the printed stereo plate in hardware/prints. Visual only (no
    collision). Every part lies behind the optical centres and within the cameras' 5 cm near clip, and the post stands
    behind and below their field of view, so the stereo images are unchanged (checked pixel for pixel)."""
    _, _, quat, head = rendered_head_pose()
    R, c = head[:3, :3], head[:3, 3]
    at = lambda p: tuple((R @ np.asarray(p, dtype=float) + c).tolist())  # head-frame point -> env frame
    look = lambda rgb, metal=0.0: sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, metallic=metal, roughness=0.5)
    half = STEREO_BASELINE / 2
    parts = {
        "plate": (sim_utils.CuboidCfg(size=(0.006, 0.10, 0.032), visual_material=look((0.12, 0.12, 0.13))),
                  at((-0.016, 0, 0)), quat),
    }
    for side, y in (("l", half), ("r", -half)):
        parts[f"board_{side}"] = (sim_utils.CuboidCfg(size=(0.002, 0.025, 0.024), visual_material=look((0.05, 0.35, 0.12))),
                                  at((-0.012, y, 0)), quat)
        parts[f"holder_{side}"] = (sim_utils.CuboidCfg(size=(0.006, 0.013, 0.013), visual_material=look((0.03, 0.03, 0.03))),
                                   at((-0.008, y, 0)), quat)
        parts[f"lens_{side}"] = (sim_utils.CylinderCfg(radius=0.0045, height=0.005, axis="X",
                                                       visual_material=look((0.02, 0.02, 0.05), 0.6)),
                                 at((-0.003, y, 0)), quat)
    ball = at((-0.032, 0, 0))
    parts["ball"] = (sim_utils.SphereCfg(radius=0.012, visual_material=look((0.2, 0.2, 0.2), 0.8)), ball, quat)
    post_h = ball[2] - 0.01
    parts["post"] = (sim_utils.CylinderCfg(radius=0.008, height=post_h, axis="Z", visual_material=look((0.6, 0.6, 0.62), 0.9)),
                     (ball[0], ball[1], post_h / 2), (0.0, 0.0, 0.0, 1.0))
    parts["foot"] = (sim_utils.CylinderCfg(radius=0.05, height=0.008, axis="Z", visual_material=look((0.1, 0.1, 0.1))),
                     (ball[0], ball[1], 0.004), (0.0, 0.0, 0.0, 1.0))
    return {f"head_rig_{k}": AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/HeadRig/{k}", spawn=spawn,
                                          init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot))
            for k, (spawn, pos, rot) in parts.items()}


def stereo_intrinsics() -> dict:
    """Pinhole intrinsics shared by both cameras (square pixels, principal point at the centre)."""
    f_px = STEREO_FOCAL_LENGTH / STEREO_APERTURE * CAMERA_RES
    return {"width": CAMERA_RES, "height": CAMERA_RES, "fx": f_px, "fy": f_px, "cx": CAMERA_RES / 2, "cy": CAMERA_RES / 2}

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
    """Table, SO-101, four cubes, two plates and a stereo head camera."""

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

    left_cam = _stereo_cam_cfg("left_cam", RENDER_LEFT_CAM_POS)
    right_cam = _stereo_cam_cfg("right_cam", RENDER_RIGHT_CAM_POS)


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
            # SO101_TASK_MODE=stack: put a cube on top of another cube (new skill "stack"); default place.
            "stack": os.environ.get("SO101_TASK_MODE", "place") == "stack",
        },
    )
    if os.environ.get("SO101_RANDOM_PLATE_COLORS", "1") != "0" and not os.environ.get("SO101_PLATE_COLORS"):
        # Per-episode plate colours from language.PLATE_PALETTE (names in env.pp_plate_colors). Default on;
        # SO101_RANDOM_PLATE_COLORS=0 gives the fixed white/pink plates, SO101_PLATE_COLORS a fixed override.
        randomize_plate_colors = EventTerm(
            func=mdp.randomize_plate_colors, mode="reset", params={"palette": language.PLATE_PALETTE}
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
    if os.environ.get("SO101_TASK_MODE", "place") == "stack":
        success = DoneTerm(func=mdp.target_on_cube, params={"cube_size": CUBE_SIZE})
    else:
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
        if os.environ.get("SO101_NO_CAMERA_MOUNT", "0") == "1":
            # The USD "Sensor" variant adds a wrist camera mount (mesh + collision) to the gripper; "none" drops it.
            self.scene.robot.spawn = self.scene.robot.spawn.replace(
                variants={**self.scene.robot.spawn.variants, "Sensor": "none"})
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        if os.environ.get("SO101_SHOW_HEAD", "0") == "1":  # demo: show the stereo head and its post
            for name, cfg in stereo_head_rig_cfgs().items():
                setattr(self.scene, name, cfg)

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
