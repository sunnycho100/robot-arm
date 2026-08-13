#!/usr/bin/env python3
"""Is the knob actually between the fingers? Ask the camera, not the servos.

The failure this catches, measured on the bench: a taught pose replayed with
the fingertips hovering at cap-top height, the jaws closed on air, and the
force check said "shut on nothing" while the xy search wandered around a miss
that was VERTICAL, which it never searches. Forward kinematics said 73.8 mm and
the video said "well above the knob", so the servo-side numbers cannot referee
this. The camera can.

    gap(frame, reference, cx, cy, cap_r) -> (gap_px, dx_px, seen)

The gripper is found by DIFFERENCE against a reference frame taken with the
arm parked, not by looking for dark pixels. Two simpler versions were tried on
real bench frames first and both misread a known-good grip as 93 px too high:
"dark rows" stopped at the servo body because the two fingers below it are
thin, and "dark mass connected to the top of the strip" lost the fingers where
a bright linkage screw split them from the body. Meanwhile the knob's own
skirt, the robot base and the shadows are all dark and all lie. Everything
static cancels in the difference; what remains in the strip is the arm.

gap_px    how far above the cap centre the gripper's lowest point sits.
          Positive = too high by that many pixels. <= 0 = the fingers reach
          down past the cap centre, where a real grip lives.
dx_px     lateral offset of that lowest part from the knob centre.
seen      False when nothing gripper-sized has entered the strip, meaning the
          miss is bigger than this window and the force search owns it.

Scale-free on purpose: callers convert px to mm with the cap's own on-screen
size (CAP_MM / 2*cap_r), so nothing here depends on where the camera stands.
"""
import numpy as np
import cv2

DIFF = 45                  # gray levels of change that count as "something moved in"
ROWS_ABOVE = 7.0           # strip reaches this many cap radii above the knob
HALF_WIDTH = 2.4           # strip half-width, in cap radii


def gap(frame, reference, cx, cy, cap_r):
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(cx - HALF_WIDTH * cap_r)), min(w, int(cx + HALF_WIDTH * cap_r))
    y0, y1 = max(0, int(cy - ROWS_ABOVE * cap_r)), min(h, int(cy + 1.2 * cap_r))
    now = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)
    ref = cv2.cvtColor(reference[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)
    moved = (np.abs(now - ref) > DIFF).astype(np.uint8)
    moved = cv2.morphologyEx(moved, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(moved, connectivity=8)
    keep = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= cap_r * cap_r]
    if not keep:                               # nothing gripper-sized came in
        return 0.0, 0.0, False
    bottom = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1
                 for i in keep)
    mask = np.isin(lab, keep)
    ys, xs = np.where(mask)
    low = ys > bottom - 2 * cap_r
    dx = float(x0 + xs[low].mean() - cx)
    return float(cy - (y0 + bottom)), dx, True


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        raise SystemExit('usage: gripcheck.py <reference.jpg> <frame.jpg> [...]')
    knobs = {'knob1': (404, 395, 18), 'knob2': (466, 431, 18),
             'knob3': (523, 390, 17)}
    ref = cv2.imread(sys.argv[1])
    for path in sys.argv[2:]:
        f = cv2.imread(path)
        print(path)
        for name, (cx, cy, r) in knobs.items():
            g, dx, seen = gap(f, ref, cx, cy, r)
            print(f'  {name}: ' + (f'gap {g:+6.1f} px  lateral {dx:+6.1f} px'
                                   if seen else 'no gripper near it'))
