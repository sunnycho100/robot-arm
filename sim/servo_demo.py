#!/usr/bin/env python3
"""Render the centring loop so it can be watched instead of read.

Produces two files in this folder:
  servo_steps.png    the camera's view at each iteration, target and gripper
                     marked, with the pixel error printed
  servo_track.png    the path the gripper took in the image, all 8 start
                     directions overlaid

    .venv/bin/python servo_demo.py [start_offset_mm]

Nothing here is a test; the gates live in servo_center.py. This is for
looking at, and for the presentation.
"""
import sys
import numpy as np
import cv2
import scene
import servo_center
from servo_center import Servo, TOL_PX

OFFSET_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
HOME = scene.knob_targets()['knob1']


def annotate(frame, gripper, target, err, step, box, note=''):
    """Mark up the frame and crop to `box`. The labels are drawn AFTER the
    crop on purpose: drawn before, they sit at the top of a 1280x960 frame
    and the crop throws them away."""
    x0, y0, x1, y1 = box
    vis = frame[y0:y1, x0:x1].copy()
    gx, gy = int(gripper[0]) - x0, int(gripper[1]) - y0
    tx, ty = int(target[0]) - x0, int(target[1]) - y0
    cv2.circle(vis, (tx, ty), 14, (0, 200, 255), 2)          # target: amber
    cv2.drawMarker(vis, (tx, ty), (0, 200, 255), cv2.MARKER_CROSS, 26, 2)
    cv2.circle(vis, (gx, gy), 9, (60, 255, 60), -1)          # gripper: green
    if err > TOL_PX:
        cv2.arrowedLine(vis, (gx, gy), (tx, ty), (255, 255, 255), 2,
                        tipLength=0.18)
    ok = err <= TOL_PX
    colour = (60, 255, 60) if ok else (255, 255, 255)
    label = f'step {step}   {err:.1f} px = {err * 0.208:.1f} mm'
    if note:
        label += f'   {note}'
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                colour, 2)
    return vis


def main():
    see, move, cam, data = servo_center._rig(noise_px=0.5, seed=3)
    s = Servo(see, move)
    s.calibrate(HOME)
    move(HOME)
    target = see()

    # one run, captured frame by frame
    start = np.array(HOME, float)
    start[0] += OFFSET_MM / 1000.0 * np.cos(np.radians(35))
    start[1] += OFFSET_MM / 1000.0 * np.sin(np.radians(35))

    # crop window: centred on the target, wide enough to hold a 10 mm excursion
    tx, ty = int(target[0]), int(target[1])
    half = 260
    x0, y0 = max(0, tx - half), max(0, ty - half)
    x1, y1 = min(cam.width, tx + half), min(cam.height, ty + half)
    box = (x0, y0, x1, y1)

    crops = []
    xyz = start.copy()
    move(xyz)
    Jinv = np.linalg.inv(s.J)
    for i in range(s.max_steps + 1):
        seen = see()
        err = float(np.linalg.norm(np.array(target) - np.array(seen)))
        note = 'on target' if err <= TOL_PX else ''
        crops.append(annotate(cam.render(data), seen, target, err, i, box, note))
        if err <= TOL_PX:
            break
        step = s.gain * (Jinv @ (np.array(target) - np.array(seen)))
        n = float(np.linalg.norm(step))
        if n > servo_center.MAX_STEP_M:
            step *= servo_center.MAX_STEP_M / n
        xyz[:2] += step
        move(xyz)

    cols = min(3, len(crops))
    rows = (len(crops) + cols - 1) // cols
    ch, cw = crops[0].shape[:2]
    sheet = np.full((rows * ch, cols * cw, 3), 24, np.uint8)
    for i, c in enumerate(crops):
        r, col = divmod(i, cols)
        sheet[r * ch:(r + 1) * ch, col * cw:(col + 1) * cw] = c
    cv2.imwrite(str(scene.HERE / 'servo_steps.png'), sheet)
    print(f'servo_steps.png: {len(crops)} iterations from {OFFSET_MM:.0f} mm out')

    # all eight directions, as paths in the image
    canvas = cam.render(data)[y0:y1, x0:x1].copy()
    canvas = (canvas * 0.45).astype(np.uint8)
    for k in range(8):
        th = k * np.pi / 4
        st = np.array(HOME, float)
        st[:2] += (OFFSET_MM / 1000.0) * np.array([np.cos(th), np.sin(th)])
        log = []
        s.center(st, target, log=log)
        pts = []
        xyz2 = st.copy()
        move(xyz2)
        pts.append(see())
        Ji = np.linalg.inv(s.J)
        for _ in range(s.max_steps):
            seen = see()
            e = np.array(target) - np.array(seen)
            if np.linalg.norm(e) <= TOL_PX:
                break
            stp = s.gain * (Ji @ e)
            nn = float(np.linalg.norm(stp))
            if nn > servo_center.MAX_STEP_M:
                stp *= servo_center.MAX_STEP_M / nn
            xyz2[:2] += stp
            move(xyz2)
            pts.append(see())
        col = tuple(int(v) for v in cv2.applyColorMap(
            np.uint8([[k * 30]]), cv2.COLORMAP_HSV)[0][0])
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, (int(a[0] - x0), int(a[1] - y0)),
                     (int(b[0] - x0), int(b[1] - y0)), col, 2)
        cv2.circle(canvas, (int(pts[0][0] - x0), int(pts[0][1] - y0)), 6, col, -1)
    cv2.drawMarker(canvas, (tx - x0, ty - y0), (255, 255, 255),
                   cv2.MARKER_CROSS, 30, 2)
    cv2.circle(canvas, (tx - x0, ty - y0), int(TOL_PX / 0.208 * 0.208) + 10,
               (0, 200, 255), 1)
    cv2.putText(canvas, f'8 starts, {OFFSET_MM:.0f} mm out, converging on the knob',
                (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(scene.HERE / 'servo_track.png'), canvas)
    print('servo_track.png: 8 approach paths')
    cam.close()


if __name__ == '__main__':
    main()
