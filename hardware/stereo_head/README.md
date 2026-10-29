# Stereo head plate (two Arducam 8MP IMX219 V2.3 boards)

90 x 35 x 10 mm plate with locating pins for two camera boards, lens centres **60 mm apart** (the training
baseline, `STEREO_BASELINE` in `so101_lang/envs/pick_place_env_cfg.py`). Both boards are mounted upright with
their ribbons exiting **downward**.

| File | |
|---|---|
| `stereo_plate.scad` | parametric source (OpenSCAD) |
| `make_stereo_plate.py` | builds `../prints/stereo_plate.stl` from the same parameters without OpenSCAD (`uv run --with manifold3d --with numpy python hardware/stereo_head/make_stereo_plate.py`) |
| `../prints/stereo_plate.stl` | print file: lies on its back, pins up; no supports needed |
| `draw_stereo_plate.py`, `stereo_plate_drawing.png` | dimensioned front view and render |

**Design.** The boards follow the Raspberry Pi Camera V2 footprint: 25 x 23.86 mm, four 2.2 mm (M2) holes on a
21 x 12.5 mm pattern. Upright (ribbon down) the 21 mm spacing is horizontal, so the outer pins sit 40.5 mm from the
centre and the plate needs at least 86 mm; it is 90 mm. Pins are 1.9 mm on 4.5 mm x 3 mm standoffs (clearance for
parts on the back of the board). Two counterbored M4 holes on the centre line bolt the plate to a bracket that sets
the 49 degree downward tilt. Shallow crosses on the front face mark the two lens centres. Facing the lenses, the
camera's own left (training's left camera) is on the viewer's right.

**Measure before printing.** The lens position relative to the hole pattern (`lens_off_y` = 7.9 mm below the
pattern centre, `lens_off_x` = 0; lens to the board's ribbon edge 7.8 mm) was estimated from a photo: the lens sits
about 1.6 mm below the lower hole row. It sets the baseline and the plate's vertical placement, so measure it with
calipers (lens centre to the hole centres) and update both scripts.

**In software.** The images come out upright. The IMX219 standard lens covers about 62 x 49 degrees; crop the centre
to a 36 degree square and resize to 512 x 512 to match training. Calibrate the pair (intrinsics + stereo
extrinsics) with a checkerboard after mounting, and the head-to-base transform with a hand-eye calibration. The two
boards are not hardware-synchronised; grab both frames as close together as possible (the arm moves slowly, but a
30 ms offset still shifts a moving gripper by a few mm in depth).
