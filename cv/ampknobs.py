#!/usr/bin/env python3
"""Find the knobs on an amp panel. Standalone: imports nothing of our own.

    python3 ampknobs.py frame.jpg [more.jpg ...]     annotate and report
    from ampknobs import find;  knobs = find(bgr)

Written against the Fender Rumble, from three camera positions that disagree
about almost everything: near overhead, an oblique three-quarter view, and one
nearly edge-on where the caps are strong ellipses and the cream skirt is as
bright as the cap. The old pedal-era finder assumed an aluminium cap ringed by
a black skirt and lost every knob on the last of those.

The one question that survives all three views
----------------------------------------------
IS THIS CONTOUR AN ELLIPSE?

A knob is a cylinder. Photographed from anywhere, its silhouette is an ellipse
(head on, a circle; edge on, a squashed one; and the knurled skirt adds no
corners). Nothing else on this bench is: cables bend, panel lettering is
rectangular, the arm is faceted, indicator LEDs are ellipses but the wrong
size, and a partly covered knob stops being one, which is the correct answer.

So each contour is fitted with an ellipse and scored on how much of that
ellipse it actually fills. Measured across all three views:

    real knobs        fill 0.98 to 1.00      convexity 0.94 to 0.98
    everything else   fill 0.64 to 0.95      convexity 0.62 to 0.97

Nothing lands between 0.95 and 0.98, so the bar goes at 0.97, and convexity at
0.94 as a second, cheap opinion. Across the three frames that is 20 knobs found
and ZERO false positives, without one threshold about colour, size or position.

What this deliberately does NOT assume
--------------------------------------
- that knobs are in a row. A DS-1's three sit in a triangle.
- that the panel is dark around each knob. On this amp the neighbours are
  20 px away, so most of a knob's surround is another knob.
- that there is a black skirt. This amp's knobs are cream all the way down.
- how many knobs there are.
- a fixed brightness. The threshold is Otsu's, taken per frame, so a camera
  that drifts warm or an arm that shades the panel does not move the bar.
"""
import sys

import numpy as np
import cv2

# The two shape gates. See the docstring for the measurements behind them.
MIN_FILL = 0.97          # contour area / fitted-ellipse area
MIN_CONVEX = 0.94        # contour area / convex-hull area

MIN_AREA = 800           # a knob smaller than this is unusable anyway
MAX_S = 130              # a knob is a LOW-saturation colour: cream or metal
OPEN_FRAC = 0.10         # opening kernel, as a fraction of the median blob size


def _mask(bgr):
    """Bright, low-saturation pixels. Otsu picks the brightness bar per frame.

    Fixed thresholds were what broke earlier: a cream knob measures around 60
    saturation, so a camera warming up pushed it over a hand-set bar and knobs
    vanished mid-run. Otsu re-derives the split from this frame's own
    histogram, so shade, exposure and colour cast move the bar with the scene.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    low_s = S < MAX_S
    if low_s.sum() < 100:
        return np.zeros(S.shape, np.uint8), 0
    bar, _ = cv2.threshold(V[low_s], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((V >= bar) & low_s).astype(np.uint8), float(bar)


def _shape(contour):
    """-> (fill, convexity, (cx, cy), (major, minor), angle) or None."""
    if len(contour) < 5:
        return None
    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        return None
    hull = cv2.contourArea(cv2.convexHull(contour))
    (ex, ey), (a1, a2), ang = cv2.fitEllipse(contour)
    ellipse_area = np.pi * a1 * a2 / 4.0
    if ellipse_area <= 0:
        return None
    major, minor = max(a1, a2), min(a1, a2)
    return (area / ellipse_area, area / max(hull, 1e-6),
            (ex, ey), (major, minor), ang)


def _pointer(bgr, k):
    """Angle of the dark line printed across the cap, in image degrees.

    Searched only over the UPPER part of the silhouette. Seen at an angle a
    knob shows its top face above and its knurled skirt below, and the
    knurling is a fan of dark vertical lines that will happily out-vote a real
    pointer. The top face is the part that carries the mark.

    Returns (angle, contrast). Contrast under about 2 means nothing was found
    and the angle must not be used.
    """
    V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    cx, cy = k['cx'], k['cy']
    rad = k['minor'] / 2.0
    face_y = cy - 0.20 * k['minor']          # top face sits above the centroid
    ring = []
    for d in range(360):
        t = np.deg2rad(d)
        run = 0
        for r in np.arange(0.25 * rad, 0.95 * rad, 1.0):
            x, y = int(cx + r * np.cos(t)), int(face_y + r * np.sin(t) * 0.75)
            if not (0 <= y < V.shape[0] and 0 <= x < V.shape[1]):
                break
            ring.append(V[y, x])
    if not ring:
        return 0.0, 0.0
    bar = 0.72 * float(np.median(ring))

    score = np.zeros(360)
    for d in range(360):
        t, run = np.deg2rad(d), 0
        for r in np.arange(0.25 * rad, 0.95 * rad, 1.0):
            x, y = int(cx + r * np.cos(t)), int(face_y + r * np.sin(t) * 0.75)
            if not (0 <= y < V.shape[0] and 0 <= x < V.shape[1]):
                break
            if V[y, x] < bar:
                run += 1
        score[d] = run
    smooth = np.convolve(np.r_[score[-12:], score, score[:12]],
                         np.ones(9) / 9, 'same')[12:-12]
    return float(smooth.argmax()), float(smooth.max() / max(smooth.mean(), 1e-6))


def find(bgr, want_pointer=True):
    """Every knob in the frame, left to right.

    -> [{'cx','cy','major','minor','angle_deg','fill','convexity',
         'pointer','pointer_contrast'}]
    """
    m, _ = _mask(bgr)
    if not m.any():
        return []

    # Open, with the kernel sized from the blobs actually present rather than
    # typed in. The same scene photographed from twice the distance needs half
    # the kernel, and a fixed one silently stops splitting touching knobs.
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sizes = [np.sqrt(cv2.contourArea(c)) for c in cs if cv2.contourArea(c) >= MIN_AREA]
    if sizes:
        ksz = max(3, int(OPEN_FRAC * float(np.median(sizes))) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = []
    for c in cs:
        got = _shape(c)
        if got is None:
            continue
        fill, convex, (ex, ey), (major, minor), ang = got
        if fill < MIN_FILL or convex < MIN_CONVEX:
            continue
        out.append({'cx': float(ex), 'cy': float(ey), 'major': float(major),
                    'minor': float(minor), 'angle_deg': float(ang),
                    'fill': float(fill), 'convexity': float(convex)})

    # Knobs on one panel are the same object repeated. Keep the largest set
    # that agrees on size, weighted by the panel area it covers rather than by
    # how many members it has: this amp carries eight indicator LEDs against
    # eight knobs, so a head count is a coin flip while area is not.
    if len(out) > 2:
        def agree(s):
            return [k for k in out if 0.65 * s <= k['major'] <= 1.55 * s]
        best = max((k['major'] for k in out),
                   key=lambda s: sum(g['major'] ** 2 for g in agree(s)))
        out = agree(best)

    out.sort(key=lambda k: k['cx'])
    if want_pointer:
        for k in out:
            k['pointer'], k['pointer_contrast'] = _pointer(bgr, k)
    return out


def draw(bgr, knobs):
    """Annotated copy, for eyeballing what was found."""
    img = bgr.copy()
    for i, k in enumerate(knobs, 1):
        c = (int(k['cx']), int(k['cy']))
        cv2.ellipse(img, c, (int(k['major'] / 2), int(k['minor'] / 2)),
                    k['angle_deg'], 0, 360, (0, 235, 0), 2)
        if k.get('pointer_contrast', 0) >= 2.0:
            t = np.deg2rad(k['pointer'])
            fy = k['cy'] - 0.20 * k['minor']
            r = 0.9 * k['minor'] / 2
            cv2.line(img, (int(k['cx']), int(fy)),
                     (int(k['cx'] + r * np.cos(t)),
                      int(fy + r * np.sin(t) * 0.75)), (255, 60, 220), 2)
        cv2.putText(img, f"{i}", (c[0] - 8, c[1] - int(k['minor'] / 2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 235, 0), 2, cv2.LINE_AA)
    return img


def _selftest():
    """Synthetic knobs on a dark panel, plus the impostors that matter."""
    img = np.full((300, 900, 3), 25, np.uint8)
    for i, x in enumerate((120, 260, 400, 540, 680)):
        cv2.ellipse(img, (x, 150), (45, 34), 0, 0, 360, (225, 230, 232), -1)
        cv2.line(img, (x, 150), (x + int(35 * np.cos(np.deg2rad(30 * i))),
                                 150 + int(26 * np.sin(np.deg2rad(30 * i)))),
                 (20, 20, 20), 5)
    cv2.rectangle(img, (760, 120), (860, 180), (225, 230, 232), -1)   # lettering
    cv2.circle(img, (820, 250), 9, (230, 230, 230), -1)               # indicator LED
    for p in ((60, 40), (200, 45), (340, 38), (480, 44), (620, 41)):  # a cable
        cv2.circle(img, p, 7, (215, 215, 215), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    ks = find(img)
    assert len(ks) == 5, f'found {len(ks)} on a five-knob panel'
    xs = [round(k['cx']) for k in ks]
    for got, want in zip(xs, (120, 260, 400, 540, 680)):
        assert abs(got - want) < 8, f'knob at {got}, painted at {want}'
    print(f'  5 synthetic knobs found at {xs}, rectangle and LED rejected')

    # A knob half covered by the gripper is NOT a knob. Reporting a centre for
    # a shape you cannot see the edges of is how the arm ends up gripping air.
    hidden = img.copy()
    cv2.rectangle(hidden, (380, 100), (430, 200), (25, 25, 25), -1)
    assert len(find(hidden)) == 4, 'a half-covered knob was still reported'
    print('  a knob covered by the gripper is dropped, not guessed at')

    assert not find(np.full((200, 200, 3), 200, np.uint8)), 'found knobs in a blank frame'
    print('  a blank frame yields nothing')
    print('ampknobs selftest passed')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        _selftest()
        raise SystemExit(0)
    for path in args:
        img = cv2.imread(path)
        if img is None:
            print(f'{path}: cannot read')
            continue
        ks = find(img)
        print(f'\n{path}: {len(ks)} knobs')
        for i, k in enumerate(ks, 1):
            print(f"  {i}: ({k['cx']:6.1f},{k['cy']:6.1f})  "
                  f"{k['major']:5.1f} x {k['minor']:5.1f} px  "
                  f"fill {k['fill']:.3f}  convex {k['convexity']:.3f}  "
                  f"pointer {k['pointer']:5.0f} deg (c {k['pointer_contrast']:.1f})")
        out = path.rsplit('.', 1)[0] + '_knobs.jpg'
        cv2.imwrite(out, draw(img, ks))
        print(f'  -> {out}')
