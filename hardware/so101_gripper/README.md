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
the head view with and without it), though our pipeline uses the head stereo pair, not this camera.
`../prints/Camera_Mount_from_IsaacSim_USD.stl` (44 x 39 x 60 mm, units mm) is that printed part extracted from the
USD geometry (`preview_camera_mount.png`). It is not either of TheRobotStudio's optional wrist-camera mounts
(`SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module`: 66 x 43 x 35 mm; `Wrist_Cam_Mount_32x32_UVC_Module`: a full
follower body with mount), and no published source file for it was found; it is a render mesh, so check wall
thickness and screw holes before printing.

One fixed jaw, one moving jaw driven directly by the gripper servo (STS3215, joint `gripper`). The robot adapter
(`so101_lang/robots/so101.py`) puts the grasp point 9 cm below the gripper link (`tcp_offset`) and treats 1.745 rad
as fully open. Print settings and assembly: the SO-ARM100 repository README and the LeRobot SO-101 assembly guide.
`preview.png` renders the two print parts.
