#!/usr/bin/env python3
"""Find the gripper in the camera frame. Without this, servoing cannot run.

The centring loop needs the gripper's pixel position every iteration. In
simulation that came from projecting the marker's known 3D position, which is
fine for testing the control law and useless on a bench, where no such number
exists. This is the part that has to work on real pixels.

Two ways to mark the gripper, in order of preference:

    find(frame)                     a coloured sticker, by hue
    find(frame, tag_id=9)           an ArUco tag, by detection

The tag is more precise and self-identifying; the sticker survives blur,
motion and bad focus far better, and at 300 mm a small tag on a moving
gripper is often unreadable. Try the tag, fall back to the sticker.

Colour choice is not arbitrary. The scene contains an orange pedal, a blue
robot, grey metal and a wooden table, so the sticker has to be a hue none of
those occupy. Green and magenta are the two gaps; green is the default here
and magenta is the alternative if the bench ends up with anything green in
frame.
"""
import numpy as np
import cv2

# hue windows in OpenCV's 0-179 scale, chosen to avoid the pedal (orange, ~13)
# and the robot (blue, ~110)
MARKS = {
    'green':   ((35, 85), (90, 255), (60, 255)),
    'magenta': ((145, 172), (90, 255), (60, 255)),
}
MIN_AREA_FRAC = 2e-5      # a sticker smaller than this is noise
MAX_AREA_FRAC = 0.05


def find_colour(frame, colour='green', min_area_frac=MIN_AREA_FRAC):
    """-> (u, v, area_px, fill) of the best blob, or None."""
    if colour not in MARKS:
        raise ValueError(f'unknown marker colour {colour!r}, have {list(MARKS)}')
    (h0, h1), (s0, s1), (v0, v1) = MARKS[colour]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    area_px = frame.shape[0] * frame.shape[1]
    best = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (min_area_frac * area_px <= a <= MAX_AREA_FRAC * area_px):
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        fill = a / max(w * h, 1)
        if not (0.4 < w / max(h, 1) < 2.5):
            continue
        if best is None or a > best[2]:
            best = (float(cen[i][0]), float(cen[i][1]), int(a), float(fill))
    return best


def find_tag(frame, tag_id, dict_name='4x4_50'):
    """-> (u, v) of the tag centre, or None."""
    from compat import detect
    corners, ids = detect(frame, dict_name)
    for c, i in zip(corners, ids):
        if int(i) == int(tag_id):
            m = c.mean(axis=0)
            return float(m[0]), float(m[1])
    return None


def find(frame, tag_id=None, colour='green'):
    """Gripper pixel position, or None. Tag first, sticker second.

    Returns a dict so the caller can see WHICH cue answered: on the bench that
    distinction matters, because a run that silently drops from the tag to the
    sticker has quietly become less precise and should say so.
    """
    if tag_id is not None:
        uv = find_tag(frame, tag_id)
        if uv is not None:
            return dict(u=uv[0], v=uv[1], source='tag', conf=1.0)
    blob = find_colour(frame, colour)
    if blob is None:
        return None
    u, v, area, fill = blob
    return dict(u=u, v=v, source=f'{colour} mark', conf=round(fill, 2),
                area=area)


def draw(frame, hit, colour=(0, 255, 255)):
    vis = frame.copy()
    if hit is None:
        cv2.putText(vis, 'gripper not found', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return vis
    u, v = int(hit['u']), int(hit['v'])
    cv2.drawMarker(vis, (u, v), colour, cv2.MARKER_CROSS, 30, 2)
    cv2.putText(vis, hit['source'], (u + 16, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return vis


if __name__ == '__main__':
    import pathlib
    import sys

    # ---- synthetic: does it find the mark and reject the scene? ----------
    def scene_frame(mark_uv=None, r=9):
        img = np.zeros((480, 640, 3), np.uint8)
        img[:] = (150, 160, 170)                       # grey table
        cv2.rectangle(img, (180, 150), (460, 330), (40, 120, 240), -1)   # pedal
        cv2.rectangle(img, (20, 60), (150, 420), (200, 120, 40), -1)     # robot
        for cx, cy in ((250, 240), (320, 240), (390, 240)):
            cv2.circle(img, (cx, cy), 34, (25, 25, 28), -1)
            cv2.circle(img, (cx, cy), 22, (190, 192, 195), -1)
        if mark_uv:
            cv2.circle(img, mark_uv, r, (40, 230, 60), -1)
        return cv2.GaussianBlur(img, (5, 5), 0)

    empty = scene_frame()
    assert find(empty) is None, \
        'found a gripper in a frame containing only the pedal, robot and table'
    print('a scene with no marker correctly reports nothing')

    worst = 0.0
    for u, v in ((320, 200), (240, 300), (430, 180), (300, 120), (200, 380)):
        hit = find(scene_frame((u, v)))
        assert hit is not None, f'missed the marker at ({u}, {v})'
        err = float(np.hypot(hit['u'] - u, hit['v'] - v))
        worst = max(worst, err)
    print(f'5 marker positions found, worst centre error {worst:.2f} px')
    assert worst < 2.0, f'centre off by {worst:.2f} px'

    # a mark sitting ON the orange pedal must still be found: that is where
    # the gripper actually is when it matters
    hit = find(scene_frame((320, 240)))
    assert hit is not None and abs(hit['u'] - 320) < 3, \
        'lost the marker against the pedal, which is where it will always be'
    print('marker found against the pedal, the knobs and the robot body')

    # ---- the real test: a frame RENDERED by the simulated camera ----------
    sim = pathlib.Path(__file__).resolve().parents[1] / 'sim'
    sys.path.insert(0, str(sim))
    try:
        import mujoco, scene as sc, simcam
    except Exception as e:
        print(f'(simulation not available here: {e})')
        raise SystemExit(0)

    sc.reset()
    cam = simcam.SimCam(model=sc.MODEL)
    data = mujoco.MjData(sc.MODEL)
    worst_mm = 0.0
    misses = 0
    for name, target in sc.knob_targets().items():
        stand = np.array(target, float) + 0.03 * sc.knob_normal()
        sc._pose(data, stand)
        frame = cam.render(data)
        hit = find(frame)
        truth = cam.project(data.geom('gripper_marker').xpos)
        if hit is None:
            misses += 1
            continue
        err_px = float(np.hypot(hit['u'] - truth[0], hit['v'] - truth[1]))
        worst_mm = max(worst_mm, err_px * simcam.mm_per_px(cam, cam.cam.distance))
    print(f'found on {3 - misses}/3 rendered frames, worst {worst_mm:.2f} mm '
          f'from where the camera model says the marker is')
    assert misses == 0, f'the detector missed {misses} rendered frames'
    # The servoing loop is judged against 4 mm; a detector contributing a
    # sizeable fraction of that would be the dominant error, not a small one.
    assert worst_mm < 1.0, f'{worst_mm:.2f} mm of detector error is too much'
    cam.close()
    print('gripper self-checks passed')
