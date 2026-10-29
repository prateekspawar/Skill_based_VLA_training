# SO-101 follower gripper (the one used in simulation, training and evaluation)

The simulator loads Isaac Lab's `SO101_CFG` (`isaaclab_assets/robots/so101.py`), whose USD
`Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda` is converted from TheRobotStudio's
official `so101_new_calib.urdf`. These files are that gripper, downloaded from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0, see `LICENSE`) on 2026-10-29.

| Folder | Files | Use |
|---|---|---|
| `../prints/` | `Wrist_Roll_Follower_SO101.stl` (fixed jaw + gripper servo housing, 65 x 52 x 105 mm), `Moving_Jaw_SO101.stl` (22 x 92 x 48 mm) | 3-D printing (`STL/SO101/Individual/`) |
| `step/` | the same two parts as STEP | CAD editing (`STEP/SO101/Follower_Specific/`) |
| `sim_meshes/` | `wrist_roll_follower_so101_v1.stl`, `moving_jaw_so101_v1.stl`, `so101_new_calib.urdf` | the meshes of links `gripper_link` and `moving_jaw_so101_v1_link` in the simulation model |

**Wrist camera mount.** The USD's sensor layer (`payloads/so101-camera_mount/`) also attaches a camera mount with a
32 x 32 mm camera board to the gripper link (`gripper/visuals/camera_mount`, plus a box + cylinder collision). It is
in every training image (the plate with the green board in the videos; `isaac_render_with_without_mount.png` shows
the head view with and without it), though our pipeline uses the head stereo pair, not this camera. The real
robot does not need it: without the mount (`SO101_NO_CAMERA_MOUNT=1`) success was 34/37 vs 33/37 episodes
(`docs/NOTES.md`, section 7), so no print file is provided. (It is 44 x 39 x 60 mm and matches neither of
TheRobotStudio's optional wrist-camera mounts.)

One fixed jaw, one moving jaw driven directly by the gripper servo (STS3215, joint `gripper`). The robot adapter
(`so101_lang/robots/so101.py`) puts the grasp point 9 cm below the gripper link (`tcp_offset`) and treats 1.745 rad
as fully open. Print settings and assembly: the SO-ARM100 repository README and the LeRobot SO-101 assembly guide.
`preview.png` renders the two print parts.
