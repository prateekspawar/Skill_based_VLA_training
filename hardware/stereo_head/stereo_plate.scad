// Stereo head plate for two Arducam 8MP IMX219 V2.3 boards (Raspberry Pi Camera V2 footprint).
// Front view: X horizontal, Y vertical (up); the plate is printed lying on its back, pins up (+Z).
// Both boards are mounted upright with their ribbons pointing DOWN, so the images need no rotation. The 21 mm hole
// spacing then runs horizontally: the outer pins sit 40.5 mm from the centre, so the plate is 90 mm long (86 min).
// Origin: midway between the two lens centres. Units: mm.
// Regenerate hardware/prints/stereo_plate.stl from this file or from make_stereo_plate.py (keep the parameters in sync).

baseline     = 60;    // lens centre to centre (training: 0.06 m)
plate_l      = 90;
plate_h      = 35;
plate_t      = 10;

hole_dx      = 21.0;  // board hole spacing, horizontal when mounted ribbon-down (Pi Camera V2: 21 x 12.5)
hole_dy      = 12.5;  // vertical
// Lens centre relative to the centre of the 4-hole pattern (ribbon down): lens_off_x to the side, lens_off_y BELOW
// the pattern centre (toward the ribbon). Estimated from a photo (the lens sits about 1.6 mm below the lower hole
// row): MEASURE AND UPDATE.
lens_off_x   = 0.0;
lens_off_y   = 7.9;
board_h      = 23.86; // board height when mounted ribbon-down
lens_to_bottom = 7.8; // lens centre to the board's ribbon edge (photo estimate)

pin_d        = 1.9;   // board holes are 2.2 mm (M2); 1.9 leaves print tolerance
pin_h        = 2.5;   // above the standoff (board is 1 mm thick)
standoff_d   = 4.5;
standoff_h   = 3.0;   // clears parts on the back of the board

m4_d         = 4.4;   // two M4 through holes on the centre line, to bolt the plate to a bracket
m4_cbore_d   = 8.0;
m4_cbore_h   = 3.0;
m4_dy        = 8.0;   // +- from the plate centre

plate_cy = (board_h / 2) - lens_to_bottom;  // plate centred on the boards vertically

$fn = 48;

module pin() {
    cylinder(d = standoff_d, h = standoff_h);
    translate([0, 0, standoff_h]) cylinder(d = pin_d, h = pin_h - 0.4);
    translate([0, 0, standoff_h + pin_h - 0.4]) cylinder(d1 = pin_d, d2 = pin_d - 0.6, h = 0.4);  // chamfer
}

module board_pins(side) {  // side = -1 / +1
    cx = side * baseline / 2 - lens_off_x;
    cy = lens_off_y;
    for (sx = [-1, 1], sy = [-1, 1])
        translate([cx + sx * hole_dx / 2, cy + sy * hole_dy / 2, plate_t]) pin();
}

difference() {
    translate([-plate_l / 2, plate_cy - plate_h / 2, 0]) cube([plate_l, plate_h, plate_t]);
    for (sy = [-1, 1]) {
        translate([0, plate_cy + sy * m4_dy, -1]) cylinder(d = m4_d, h = plate_t + 2);
        translate([0, plate_cy + sy * m4_dy, plate_t - m4_cbore_h]) cylinder(d = m4_cbore_d, h = m4_cbore_h + 1);
    }
    // lens-centre marks (0.6 mm deep crosses on the front face, hidden under the boards)
    for (side = [-1, 1]) translate([side * baseline / 2, 0, plate_t - 0.6]) {
        translate([-3, -0.3, 0]) cube([6, 0.6, 1]);
        translate([-0.3, -3, 0]) cube([0.6, 6, 1]);
    }
}
board_pins(-1);
board_pins(1);
