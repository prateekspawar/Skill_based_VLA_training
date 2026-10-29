# Print files

All print files in one place: 3-D prints and the paper calibration boards (units mm). Sources and details are in the sibling folders.

| File | Part | Qty | Source |
|---|---|---|---|
| `Wrist_Roll_Follower_SO101.stl` | SO-101 gripper body + fixed jaw (the gripper used in simulation and training) | 1 | [`../so101_gripper/`](../so101_gripper/README.md), TheRobotStudio SO-ARM100 (Apache-2.0) |
| `Moving_Jaw_SO101.stl` | SO-101 moving jaw | 1 | same |
| `stereo_plate.stl` | Stereo head plate for two Arducam IMX219 boards, 60 mm baseline. **Measure the lens offset before printing** | 1 | [`../stereo_head/`](../stereo_head/README.md) |
| `cube.stl` | 30 mm task cube (red, green, blue, yellow) | 4 | [`../objects/`](../objects/README.md) |
| `plate_decagon.stl` | 90 mm task plate, 10-sided as rendered in simulation | 2+ | same |
| `handeye_board_plate.stl` | 98 x 98 x 3 mm backing plate for the hand-eye ChArUco board | 1 | [`../calibration/`](../calibration/README.md) |
| `handeye_jaw_clamp.stl` | Clamp that holds the board plate on the SO-101 fixed jaw during hand-eye calibration | 1 | same |
| `charuco_handeye_5x5_18mm.pdf` | Hand-eye ChArUco board, 5 x 5 squares of 18 mm; print on A4 at 100 %, glue on the board plate | 1 | [`../calibration/`](../calibration/README.md) |
| `charuco_stereo_8x11_22mm.pdf` | ChArUco board for the stereo cameras' own calibration, 8 x 11 squares of 22 mm; print on A4 at 100 %, glue on foam board | 1 | same |
