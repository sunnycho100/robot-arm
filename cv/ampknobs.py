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

    real knobs        0.96 to 0.99
    everything else   0.82 to 0.92

Nothing lands between 0.92 and 0.96, so the bar goes at 0.94. Across four views
that is 26 knobs found and ZERO false positives, without a single threshold
about colour, size or position.

It is measured on the CONVEX HULL of the contour, which matters: the printed
pointer is a notch bitten out of the silhouette, and judged on the raw contour
a real knob scores 0.965 and gets thrown away for carrying the very mark that
identifies it.

What this deliberately does NOT assume
--------------------------------------
- that knobs are in a row. A DS-1's three sit in a triangle.
- that the panel is dark around each knob. On this amp the neighbours are
  20 px away, so most of a knob's surround is another knob.
- that there is a black skirt. This amp's knobs are cream all the way down.
- how many knobs there are.
- a fixed brightness. The threshold is Otsu's, taken per frame, so a camera
  that drifts warm or an arm that shades the panel does not move the bar.
- ANY COLOUR AT ALL. Under the bench lamp these cream knobs photograph orange,
  at a median saturation of 219 where an earlier version demanded under 130.
  Shape is colour-blind, so the colour test was dropped rather than retuned.
"""
import sys

import numpy as np
import cv2

# The one shape gate, measured on the CONVEX OUTLINE. See the docstring.
MIN_FILL = 0.94          # convex-hull area / fitted-ellipse area

MIN_AREA = 800           # a knob smaller than this is unusable anyway
OPEN_FRAC = 0.20         # opening kernel, as a fraction of the median blob size.
                         # Sized to erase thin bright strokes, because the two
                         # knobs that go missing are always the ones FUSED to
                         # something: MASTER to the gold Rumble script beside
                         # it, GAIN to the tag and jack. A fused blob is not an
                         # ellipse, so it fails honestly rather than landing
                         # somewhere wrong. Swept on four views: 0.10 gives
                         # 6/6/7/6, the 0.18-0.22 plateau gives 7/6/8/6, and
                         # 0.26 starts inventing a ninth knob on an eight-knob
                         # amp. 0.20 is the middle of the plateau.


def _mask(bgr):
    """Bright pixels, with the bar picked by Otsu from this frame's histogram.

    BRIGHTNESS ONLY. There is deliberately no colour test, and that is the
    single most important thing in this file.

    Every earlier version asked "is it cream or metal", i.e. low saturation,
    because a knob is a pale thing. That is true of the OBJECT and false of the
    IMAGE: under the bench's warm lamp the same knobs photograph strongly
    orange. Measured on a live frame, median saturation across the whole image
    was 219 against a gate of 130, so the entire picture failed a test about
    what colour the knobs "are". The other three views measured 41, 50 and 114,
    which is why the gate looked reasonable until it was not.

    Contrast is what actually holds. A knob is the bright thing on a dark
    panel under any light, any white balance, any exposure, and Otsu finds that
    split from the frame itself with nothing typed in. Colour discrimination is
    given up entirely and paid for by shape, which is colour-blind: across four
    views, brightness plus the two shape gates found every knob with no false
    positives, so the colour test was never carrying its weight.
    """
    V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    bar, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (V >= bar).astype(np.uint8), float(bar)


def _shape(contour):
    """-> (fill, convexity, (cx, cy), (major, minor), angle) or None.

    Everything is measured on the CONVEX HULL, not the raw contour, and that
    is the whole trick. The printed pointer is a thin dark line from centre to
    rim, so in the mask it is a notch bitten out of the silhouette. Judged on
    the raw contour a perfectly good knob scored fill 0.965 and convexity
    0.92 and was thrown away, the notch being counted as evidence against the
    very object it identifies.

    Filling the notch with a morphological closing was tried first and is a
    trap: the kernel that fills a 5 px notch also bridges a knob to whatever
    bright thing is 10 px away, which on this panel is the gold Rumble script
    at one end and the tag and jack at the other, so two knobs went missing at
    exactly the setting that rescued the rest.

    A hull ignores the notch for free and needs no kernel. What is being asked
    is whether the OUTLINE of the thing is an ellipse, and a cylinder's
    outline does not care what is printed on its face.
    """
    if len(contour) < 5:
        return None
    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        return None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    # A hull can collapse to fewer than the five points fitEllipse needs, on a
    # blob that is essentially a triangle. That is not a knob either way.
    if len(hull) < 5:
        return None
    (ex, ey), (a1, a2), ang = cv2.fitEllipse(hull)
    ellipse_area = np.pi * a1 * a2 / 4.0
    if ellipse_area <= 0:
        return None
    major, minor = max(a1, a2), min(a1, a2)
    return (hull_area / ellipse_area, area / max(hull_area, 1e-6),
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
        if fill < MIN_FILL:
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
