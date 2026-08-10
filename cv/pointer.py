#!/usr/bin/env python3
"""Read the white pointer line on a knob cap, and decide whether it turned.

Reading a knob is the analog-gauge problem: find the disc, find the needle,
take its angle. Gauge readers use HoughLines for the needle, but the DS-1
pointer is a short white tab on a grey cap, not a long thin line, so a
brightness-weighted centroid of the pointer pixels is both simpler and more
stable at this size.

Angles are degrees, 0 = pointing right (+x), increasing counter-clockwise,
which is the ordinary maths convention. Image y grows downward so the sign
of the y term is flipped when the angle is formed.

    ang = angle(frame, knob)                     # absolute, may be None
    d = turned(before, after, knob)              # signed change, degrees

Verification only needs the CHANGE between two frames, so absolute
calibration of "which angle is volume 0" is never required.
"""
import numpy as np
import cv2

INNER, OUTER = 0.35, 1.05    # search the pointer between these multiples of r
MIN_PIXELS = 12              # fewer bright pixels than this is not a pointer
MIN_CONTRAST = 12            # the tab must beat the cap by this much, in levels


def _annulus(shape, cx, cy, r_in, r_out):
    y, x = np.ogrid[:shape[0], :shape[1]]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return (d2 >= r_in ** 2) & (d2 <= r_out ** 2)


def angle(frame, knob, inner=INNER, outer=OUTER):
    """Pointer angle in degrees, or None if no pointer stands out.

    knob is a dict with cx, cy, r_px (what knobs2.find returns).
    """
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy, r = knob['cx'], knob['cy'], knob['r_px']
    ring = _annulus(g.shape, cx, cy, r * inner, r * outer)
    if ring.sum() < 50:
        return None
    vals = g[ring]
    # The tab is a small fraction of the annulus, so a fixed percentile either
    # lands inside the cap or clips the tab as its apparent size changes with
    # distance. Otsu adapts, but only if it is shown the right two
    # populations: run it on the annulus as a whole and it happily separates
    # the dark body from the cap and never mentions the tab. So drop
    # everything at or below the median first (that is the body and the dimmer
    # half of the cap), leaving cap-versus-tab as the only split available.
    mid = float(np.median(vals))
    bright = vals[vals > mid]
    if bright.size < MIN_PIXELS:
        return None
    cut, _ = cv2.threshold(bright.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # the pointer must really outshine the cap it sits on, not just be the
    # bright half of a flat patch
    if cut <= mid + MIN_CONTRAST:
        return None
    sel = ring & (g > cut)
    n = int(sel.sum())
    if n < MIN_PIXELS:
        return None
    ys, xs = np.nonzero(sel)
    w = g[ys, xs].astype(float) - cut                 # brightness-weighted
    if w.sum() <= 0:
        return None
    mx = float((xs * w).sum() / w.sum()) - cx
    my = float((ys * w).sum() / w.sum()) - cy
    if np.hypot(mx, my) < r * 0.08:                   # centroid on the axis
        return None
    return float(np.degrees(np.arctan2(-my, mx)) % 360.0)


def wrap(delta):
    """Fold a difference of angles into [-180, 180)."""
    return (float(delta) + 180.0) % 360.0 - 180.0


def turned(before, after, knob, knob_after=None):
    """Signed rotation in degrees between two frames, or None if unreadable."""
    a = angle(before, knob)
    b = angle(after, knob_after or knob)
    if a is None or b is None:
        return None
    return wrap(b - a)


def draw(frame, knob, ang, colour=(0, 0, 255)):
    vis = frame.copy()
    if ang is None:
        return vis
    cx, cy, r = knob['cx'], knob['cy'], knob['r_px']
    tip = (int(cx + r * np.cos(np.radians(ang))),
           int(cy - r * np.sin(np.radians(ang))))
    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
    cv2.line(vis, (int(cx), int(cy)), tip, colour, 3)
    cv2.putText(vis, f'{ang:.0f}', (int(cx) - 25, int(cy - r) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return vis


# ---------------------------------------------------------------- self-check
def _synth(ang_deg, r=60, size=200, blur=3, noise=6, seed=0):
    """A grey cap on a dark body with a white pointer tab at a known angle."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 30, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(r * 1.45), (25, 25, 28), -1)
    cv2.circle(img, (c, c), r, (185, 188, 190), -1)
    t = np.radians(ang_deg)
    tip = (int(c + r * 0.82 * np.cos(t)), int(c - r * 0.82 * np.sin(t)))
    cv2.circle(img, tip, max(4, r // 7), (245, 245, 245), -1)
    img = cv2.GaussianBlur(img, (blur * 2 + 1,) * 2, 0)
    if noise:
        img = np.clip(img.astype(int) + rng.normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return img, dict(cx=float(c), cy=float(c), r_px=float(r))


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # 1. absolute angle over 100 random targets
    errs = []
    misses = 0
    for k in range(100):
        true = float(rng.uniform(0, 360))
        img, knob = _synth(true, seed=k)
        got = angle(img, knob)
        if got is None:
            misses += 1
            continue
        errs.append(abs(wrap(got - true)))
    print(f'100 synthetic knobs: {misses} unreadable, '
          f'mean error {np.mean(errs):.2f} deg, worst {max(errs):.2f} deg')
    assert misses == 0, f'{misses} synthetic pointers were unreadable'
    assert max(errs) < 3.0, f'worst angle error {max(errs):.2f} deg exceeds 3 deg'

    # 2. the signed change is what verification actually uses
    for true_a, true_b in [(10, 100), (350, 20), (90, 0), (180, 181)]:
        ia, ka = _synth(true_a, seed=7)
        ib, _ = _synth(true_b, seed=8)
        d = turned(ia, ib, ka)
        assert abs(wrap(d - (true_b - true_a))) < 3.0, \
            f'{true_a}->{true_b}: read {d:.1f}, expected {wrap(true_b-true_a):.1f}'
    print('signed rotation correct across the 0/360 wrap')

    # 3. a cap with no pointer must return None, not a random angle
    plain = np.full((200, 200, 3), 30, np.uint8)
    cv2.circle(plain, (100, 100), 87, (25, 25, 28), -1)
    cv2.circle(plain, (100, 100), 60, (185, 188, 190), -1)
    plain = cv2.GaussianBlur(plain, (7, 7), 0)
    assert angle(plain, dict(cx=100.0, cy=100.0, r_px=60.0)) is None, \
        'a featureless cap produced an angle'
    print('featureless cap correctly returns None')

    # 4. real frames: the old detector printed its readings onto two photos,
    # so those numbers are the only ground truth we have off the bench.
    import json, pathlib
    docs = pathlib.Path(__file__).resolve().parents[2] / 'docs' / 'bench_photos'
    if (docs / 'manifest.json').exists():
        import knobs2
        m = json.load(open(docs / 'manifest.json'))
        for n, v in m.items():
            if v['orientation'] != 'topdown' or v['overlay']:
                continue
            frame = cv2.imread(str(docs / n))
            ks = knobs2.find(frame)
            angs = [angle(frame, k) for k in ks]
            got = [f'{a:.0f}' if a is not None else '--' for a in angs]
            if ks:
                print(f'  {n[:22]}: {got}')
    print('pointer self-checks passed')
