#!/usr/bin/env python3
"""Find the pedal plane from ArUco tags and rectify it to metric pixels.

The point: once tags of known physical size are on the pedal, a homography
maps the camera image onto a top-down view where one pixel is a fixed number
of millimetres. Knob detection then works in millimetres instead of pixels,
so it survives the camera being moved, and knob positions convert straight
into arm coordinates.

    view = locate(frame, {4: 25.0, 5: 25.0})    # id -> tag size in mm
    rect = view.rectify(frame)                  # metric top-down image
    x_mm, y_mm = view.to_mm(u, v)               # pixel -> pedal-frame mm

Works from a single tag (its 4 corners are enough for a homography) and gets
better with more. Self-check is synthetic: the physical tag in our bench
photos has a torn border and detects in zero dictionaries, so real frames
cannot validate this yet. Bench day replaces the tags; the maths does not
change.
"""
import numpy as np
import cv2
from compat import detect_any, draw_marker

MM_PER_PX = 0.2          # rectified image scale
PAD_MM = 60.0            # how far around the tags to keep in the rectified view


class PlaneView:
    """A homography between the camera image and the tag plane, in mm."""

    def __init__(self, H, ids, mm_per_px=MM_PER_PX, extent=None):
        self.H = H                       # image px -> plane mm
        self.ids = list(ids)
        self.mm_per_px = mm_per_px
        self.extent = extent             # (x0, y0, x1, y1) in mm

    def to_mm(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def to_px(self, x_mm, y_mm):
        p = np.linalg.inv(self.H) @ np.array([x_mm, y_mm, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def rectify(self, frame):
        """Top-down image where 1 px = self.mm_per_px mm."""
        x0, y0, x1, y1 = self.extent
        w = int(round((x1 - x0) / self.mm_per_px))
        h = int(round((y1 - y0) / self.mm_per_px))
        S = np.array([[1 / self.mm_per_px, 0, -x0 / self.mm_per_px],
                      [0, 1 / self.mm_per_px, -y0 / self.mm_per_px],
                      [0, 0, 1.0]])
        return cv2.warpPerspective(frame, S @ self.H, (w, h))

    def rect_to_mm(self, u, v):
        """Rectified-image pixel -> plane mm."""
        return self.extent[0] + u * self.mm_per_px, self.extent[1] + v * self.mm_per_px


def _tag_plane_corners(tag_id, size_mm, layout):
    """Where this tag's 4 corners sit in the plane frame, in mm.

    layout maps id -> (cx, cy) of the tag centre in plane mm. With no layout
    the first tag defines the origin and there is one tag only.
    """
    cx, cy = layout.get(tag_id, (0.0, 0.0))
    h = size_mm / 2.0
    # detectMarkers returns corners clockwise from top-left in tag frame
    return np.array([[cx - h, cy - h], [cx + h, cy - h],
                     [cx + h, cy + h], [cx - h, cy + h]], dtype=float)


def locate(frame, sizes_mm, layout=None, mm_per_px=MM_PER_PX, dicts=('4x4_50', '6x6_250')):
    """-> PlaneView, or None if no known tag is visible.

    sizes_mm: {marker_id: physical side length in mm}
    layout:   {marker_id: (cx_mm, cy_mm)} tag centres in the plane frame.
              Omit for a single tag (it becomes the origin).
    """
    corners, ids, _ = detect_any(frame, dicts)
    if not len(ids):
        return None
    layout = layout or {}
    src, dst = [], []
    seen = []
    for c, i in zip(corners, ids):
        if i not in sizes_mm:
            continue
        src.append(c)
        dst.append(_tag_plane_corners(i, sizes_mm[i], layout))
        seen.append(int(i))
    if not seen:
        return None
    src = np.vstack(src).astype(np.float32)
    dst = np.vstack(dst).astype(np.float32)
    H, _ = (cv2.findHomography(src, dst, cv2.RANSAC, 2.0) if len(seen) > 1
            else (cv2.getPerspectiveTransform(src, dst), None))
    if H is None:
        return None
    x0, y0 = dst.min(axis=0) - PAD_MM
    x1, y1 = dst.max(axis=0) + PAD_MM
    return PlaneView(H, seen, mm_per_px, (float(x0), float(y0), float(x1), float(y1)))


# ---------------------------------------------------------------- self-check
def _synthetic(tag_id, size_px, dict_name, quiet=40, warp=None, blur=0, noise=0,
               canvas=(720, 1280), seed=0):
    """A tag pasted on a background, optionally warped, blurred and noised.
    Returns (image, true_corners_in_image)."""
    rng = np.random.default_rng(seed)
    tag = draw_marker(dict_name, tag_id, size_px)
    card = np.full((size_px + 2 * quiet,) * 2, 255, np.uint8)
    card[quiet:quiet + size_px, quiet:quiet + size_px] = tag
    img = np.full(canvas, 120, np.uint8)
    y0 = (canvas[0] - card.shape[0]) // 2
    x0 = (canvas[1] - card.shape[1]) // 2
    img[y0:y0 + card.shape[0], x0:x0 + card.shape[1]] = card
    true = np.array([[x0 + quiet, y0 + quiet],
                     [x0 + quiet + size_px - 1, y0 + quiet],
                     [x0 + quiet + size_px - 1, y0 + quiet + size_px - 1],
                     [x0 + quiet, y0 + quiet + size_px - 1]], dtype=float)
    if warp is not None:
        img = cv2.warpPerspective(img, warp, (canvas[1], canvas[0]),
                                  borderValue=120)
        p = np.hstack([true, np.ones((4, 1))]) @ warp.T
        true = p[:, :2] / p[:, 2:]
    if blur:
        img = cv2.GaussianBlur(img, (blur * 2 + 1,) * 2, 0)
    if noise:
        img = np.clip(img.astype(int) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img, true


def _random_warp(rng, canvas=(720, 1280), amount=0.10):
    h, w = canvas
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-amount, amount, (4, 2)) * [w, h]
    return cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32))


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    SIZE_MM, SIZE_PX, TAG = 25.0, 240, 4

    # 1. corner recovery under 20 random perspective warps + blur + noise
    worst = 0.0
    misses = 0
    for k in range(20):
        W = _random_warp(rng)
        img, true = _synthetic(TAG, SIZE_PX, '4x4_50', warp=W,
                               blur=1, noise=4, seed=k)
        corners, ids, _ = detect_any(img, ('4x4_50',))
        if not len(ids):
            misses += 1
            continue
        err = np.linalg.norm(corners[0] - true, axis=1).max()
        worst = max(worst, err)
    print(f'20 warped/blurred/noisy tags: {20 - misses} detected, '
          f'worst corner error {worst:.2f} px')
    assert misses == 0, f'{misses} tags went undetected'
    assert worst < 1.0, f'corner error {worst:.2f} px exceeds the 1 px gate'

    # 2. scale recovery: a known tag must measure its own size back
    errs = []
    for k in range(20):
        W = _random_warp(rng)
        img, _ = _synthetic(TAG, SIZE_PX, '4x4_50', warp=W, blur=1, noise=4, seed=100 + k)
        view = locate(img, {TAG: SIZE_MM})
        assert view is not None, 'locate() failed on a detectable tag'
        c, _ids, _ = detect_any(img, ('4x4_50',))
        p = np.array([view.to_mm(*xy) for xy in c[0]])
        side = np.linalg.norm(p[0] - p[1])
        errs.append(abs(side - SIZE_MM) / SIZE_MM)
    print(f'scale recovery over 20 warps: worst {max(errs)*100:.2f}% error')
    assert max(errs) < 0.02, f'scale error {max(errs)*100:.1f}% exceeds the 2% gate'

    # 3. two-tag layout: a point between the tags lands where it should.
    # Keep the synthetic geometry self-consistent: at 160 px per 25 mm tag
    # the scale is 6.4 px/mm, so a 60 mm separation must be 384 px.
    img = np.full((720, 1280), 120, np.uint8)
    GAP_MM, PX_PER_MM = 60.0, 160 / 25.0
    gap_px = int(round(GAP_MM * PX_PER_MM))
    placed = {}
    for i, tid in enumerate([4, 5]):
        cx = 640 - gap_px // 2 + i * gap_px
        card = np.full((240, 240), 255, np.uint8)
        card[40:200, 40:200] = draw_marker('4x4_50', tid, 160)
        img[300:540, cx - 120:cx + 120] = card
        placed[tid] = (i * GAP_MM, 0.0)
    view = locate(img, {4: 25.0, 5: 25.0}, layout=placed)
    assert view is not None and sorted(view.ids) == [4, 5], 'two-tag locate failed'
    mid_x, mid_y = view.to_mm(640, 420)
    assert abs(mid_x - 30.0) < 1.5, f'midpoint x {mid_x:.2f} mm, expected 30'
    assert abs(mid_y) < 1.5, f'midpoint y {mid_y:.2f} mm, expected 0'
    rect = view.rectify(img)
    print(f'two tags 60 mm apart: midpoint reads ({mid_x:.2f}, {mid_y:.2f}) mm, '
          f'rectified {rect.shape[1]}x{rect.shape[0]} px')

    # 4. blank frame must return None, not crash
    assert locate(np.full((480, 640), 200, np.uint8), {4: 25.0}) is None
    print('aruco_locate self-checks passed')
