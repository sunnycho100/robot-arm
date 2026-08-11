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

INNER, OUTER = 0.95, 1.90    # search from the CAP EDGE outward, never inside it:
                             # inside the cap every angle is bright, so every
                             # angle scores a long run and the pointer stops
                             # standing out. knob.py searches cap_r+1 outward
                             # for the same reason.
MIN_PIXELS = 12              # fewer bright pixels than this is not a pointer
MIN_CONTRAST = 12            # the tab must beat the cap by this much, in levels


def _annulus(shape, cx, cy, r_in, r_out):
    y, x = np.ogrid[:shape[0], :shape[1]]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return (d2 >= r_in ** 2) & (d2 <= r_out ** 2)


def _ring_values(g, cx, cy, radius, n=180):
    """Sample the image around a circle. -> (angles_deg, values)."""
    th = np.radians(np.arange(n) * (360.0 / n))
    ys = np.clip(np.rint(cy + radius * np.sin(th)).astype(int), 0, g.shape[0] - 1)
    xs = np.clip(np.rint(cx + radius * np.cos(th)).astype(int), 0, g.shape[1] - 1)
    return th, g[ys, xs].astype(float)


def angle(frame, knob, inner=INNER, outer=OUTER, n=180):
    """Pointer angle in degrees, or None if no pointer stands out.

    knob is a dict with cx, cy, r_px (what knobs2.find returns).

    Angles are in image convention: y grows downward, so the angle grows
    CLOCKWISE on screen, matching scripts/knob.py. That agreement is not
    cosmetic. If one module measures clockwise and the other anticlockwise,
    the retry loop reads every correction as being in the wrong direction and
    drives the knob away from the target instead of toward it.

    Three bright things sit near a real pointer and none of them is it: the
    brushed cap's starburst, which is fixed by the LIGHTING and does not turn
    when the knob does; the pedal body; and the table beyond the pedal edge.
    Taking the brightest angle around a ring picks all three, and scripts/
    knob.py records that doing so put a knob 94 degrees out while reporting a
    contrast of 2.7, which reads as perfectly healthy. So brightness alone is
    not enough, and two further things are needed, both learned the hard way:

    Bright RELATIVE TO THIS KNOB'S OWN CAP. The pointer is white paint and
    saturates the sensor exactly as the aluminium beside it does; the
    impostors do not. Referencing each knob's own cap survives a lighting
    change, where a fixed threshold needs re-tuning.

    It ENDS. The pointer is painted on the skirt, so the skirt closes around
    it and its bright run has a finite length. The starburst, the pedal and
    the table all run off the knob and keep going, so an angle whose bright
    run is still going at the search limit scores zero.
    """
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy, r = knob['cx'], knob['cy'], knob['r_px']
    if r < 4:
        return None

    # this knob's own cap brightness, from well inside the cap
    _, cap_vals = _ring_values(g, cx, cy, max(2.0, r * 0.45), n=64)
    cap = float(np.median(cap_vals))
    cut = max(cap * 0.88, cap - 24.0)      # the pointer is as bright as the cap

    radii = np.arange(inner * r, outer * r, max(1.0, r * 0.06))
    if radii.size < 3:
        return None
    lit = np.empty((radii.size, n), dtype=bool)
    for i, rad in enumerate(radii):
        _, v = _ring_values(g, cx, cy, rad, n)
        lit[i] = v >= cut

    # score each angle by the length of its contiguous bright run outward; a
    # run still alive at the outer limit left the knob, so it scores nothing
    score = np.zeros(n)
    for a in range(n):
        col = lit[:, a]
        run = 0
        for i in range(radii.size):
            if col[i]:
                run += 1
            elif run:
                break
        score[a] = 0.0 if (run and col[-1]) else run

    best = float(score.max())
    if best < 2:
        return None
    # a genuine pointer is a narrow wedge, not half the rim
    if (score >= best * 0.6).sum() > n * 0.35:
        return None
    # centroid of the winning wedge, so the answer is not quantised to the
    # angular step
    peak = int(np.argmax(score))
    span = max(2, int(n * 0.06))
    idx = (np.arange(peak - span, peak + span + 1)) % n
    w = score[idx]
    if w.sum() <= 0:
        return None
    off = float((np.arange(-span, span + 1) * w).sum() / w.sum())
    return float(((peak + off) * (360.0 / n)) % 360.0)


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
           int(cy + r * np.sin(np.radians(ang))))
    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
    cv2.line(vis, (int(cx), int(cy)), tip, colour, 3)
    cv2.putText(vis, f'{ang:.0f}', (int(cx) - 25, int(cy - r) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return vis


# ---------------------------------------------------------------- self-check
def _synth(ang_deg, r=60, size=200, blur=3, noise=6, seed=0, starburst=True,
           background=90):
    """A knob as the camera actually sees one.

    Not just a tab on a matte disc. The cap carries a brushed starburst fixed
    at a lighting angle that does NOT rotate with the knob, and the frame has
    a bright background beyond the knob, because those are the two impostors
    that fooled the previous detector.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), background, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(r * 1.45), (25, 25, 28), -1)
    cv2.circle(img, (c, c), r, (185, 188, 190), -1)
    if starburst:
        # a fixed specular streak across the cap: as bright as the pointer,
        # never moves
        for k in (35.0, 215.0):
            t0 = np.radians(k)
            cv2.line(img, (c, c),
                     (int(c + r * 0.95 * np.cos(t0)), int(c + r * 0.95 * np.sin(t0))),
                     (250, 250, 250), max(2, r // 12))
    t = np.radians(ang_deg)
    # the pointer is painted on the SKIRT, past the cap edge, and it ends
    p0 = (int(c + r * 0.80 * np.cos(t)), int(c + r * 0.80 * np.sin(t)))
    p1 = (int(c + r * 1.30 * np.cos(t)), int(c + r * 1.30 * np.sin(t)))
    cv2.line(img, p0, p1, (250, 250, 250), max(3, r // 9))
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

    # 3. THE test: the starburst is a fixed lighting artefact, as bright as
    # the pointer, that does not rotate. If the reading follows the knob and
    # not the starburst, the reading is real. This is the exact failure
    # scripts/knob.py records (a knob reported 94 degrees out at a healthy
    # contrast of 2.7), so it gets a test rather than a comment.
    for true in (0.0, 35.0, 90.0, 200.0, 215.0, 300.0):
        img, knob = _synth(true, starburst=True, seed=11)
        got = angle(img, knob)
        assert got is not None, f'starburst frame at {true} deg was unreadable'
        err = abs(wrap(got - true))
        assert err < 6.0, (f'followed the starburst, not the pointer: pointer '
                           f'at {true:.0f}, read {got:.0f} ({err:.0f} deg out)')
    print('starburst rejected: the reading follows the knob, including when '
          'the pointer sits on top of the streak')

    # and it must not simply be ignoring bright things: with NO pointer but a
    # starburst present, there is nothing to report
    blank, kb = _synth(0.0, starburst=True, seed=12)
    import cv2 as _cv
    c = blank.shape[0] // 2
    _cv.circle(blank, (c, c), int(kb['r_px'] * 1.45), (25, 25, 28), -1)
    _cv.circle(blank, (c, c), int(kb['r_px']), (185, 188, 190), -1)
    for k in (35.0, 215.0):
        t0 = np.radians(k)
        _cv.line(blank, (c, c),
                 (int(c + kb['r_px'] * 0.95 * np.cos(t0)),
                  int(c + kb['r_px'] * 0.95 * np.sin(t0))), (250, 250, 250), 5)
    blank = _cv.GaussianBlur(blank, (7, 7), 0)
    assert angle(blank, kb) is None, \
        'a cap with only a starburst and no pointer produced an angle'
    print('a starburst with no pointer correctly reads as nothing')

    # 3b. a cap with no pointer must return None, not a random angle
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
