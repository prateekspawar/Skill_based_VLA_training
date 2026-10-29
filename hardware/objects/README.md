# Task objects at simulation size

Dimensions from `so101_lang/envs/pick_place_env_cfg.py`; colours from `so101_lang/language.py`. Generate with
`uv run --with manifold3d --with numpy python hardware/objects/make_objects.py`.

| File | Size | Print |
|---|---|---|
| `../prints/cube.stl` | 30 x 30 x 30 mm (`CUBE_SIZE`) | 4 cubes: red, green, blue, yellow. Simulated mass 20 g: PLA at about 30-40 % infill comes close (solid would be 33 g) |
| `../prints/plate_decagon.stl` | 90 mm across corners, 8 mm thick | The shape the models saw: the plate is a cylinder (`PLATE_RADIUS` 45 mm, `PLATE_HEIGHT` 8 mm) that Isaac Sim renders with 10 flat sides |

Colours in training (sRGB 0-255, matte):

* cubes: red (217, 20, 20), green (26, 166, 38), blue (20, 64, 217), yellow (242, 204, 13)
* plates: drawn per episode from white, pink, red, blue, green, yellow, orange, purple, brown, so any saturated
  colour works; pale colours (white, pink) caused the most wrong-plate detections

The table in training is dark grey; a dark matte surface gives the closest match.
