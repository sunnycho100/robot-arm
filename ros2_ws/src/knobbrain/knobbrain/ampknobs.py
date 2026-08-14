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
            (ex, ey), (major, minor), ang, (a1, a2))


def _face_affine(k, pad=1.15):
    """-> (M, Minv, R) mapping this knob's face to a circle of radius R.

    Kept separate from the warp because the DRAWING has to invert exactly the
    transform the reading used. When it did not, the overlay was the only
    thing wrong and it looked like the angles were: pointers were measured on
    the normalised face and then drawn back with a hand-guessed
    foreshortening, so every line lay off its printed mark while the numbers
    behind them were correct to a degree.
    """
    R = int(round(pad * k['major'] / 2.0))
    if R < 6:
        return None, None, 0
    th = np.deg2rad(k['angle_deg'])
    c, s_ = np.cos(th), np.sin(th)
    A = (np.diag([k['major'] / max(k['_a1'], 1e-6),
                  k['major'] / max(k['_a2'], 1e-6)])
         @ np.array([[c, s_], [-s_, c]], float))
    t = np.array([R, R], float) - A @ np.array([k['cx'], k['cy']], float)
    M = np.hstack([A, t.reshape(2, 1)])
    return M, cv2.invertAffineTransform(M), R


def pointer_segment(k, frac=0.85):
    """The pointer as two image-space points, centre first.

    Built by placing the line on the NORMALISED face, where the angle was
    measured, and mapping it back. Anything else has to re-guess the
    foreshortening and will disagree with the number it is drawing.
    """
    M, Minv, R = _face_affine(k)
    if Minv is None:
        return None
    t = np.deg2rad(k['pointer'] - 90.0)      # undo the reporting offset
    pts = np.array([[R, R, 1.0],
                    [R + frac * R * np.cos(t), R + frac * R * np.sin(t), 1.0]]).T
    back = Minv @ pts
    return ((int(back[0, 0]), int(back[1, 0])), (int(back[0, 1]), int(back[1, 1])))


def _circle_crop(bgr, k, pad=1.15):
    """Warp one knob face into a circle. -> (crop, radius, inverse affine).

    An angled camera turns a circular face into an ellipse, so the same
    physical pointer direction lands at a different image angle depending only
    on where the camera happens to be. Undo it: rotate by the fitted ellipse's
    own angle and stretch its short axis up to match its long one, and what
    comes back is the face as it would look straight on.

    This replaces a hand-guessed 0.75 foreshortening factor that used to be
    multiplied into the y term of the radial sampling. That constant was a
    guess about one camera position, wrong everywhere else, and exactly the
    kind of magic number the rest of this file avoids. The ellipse already
    measured the foreshortening; use what it measured.
    """
    M, Minv, R = _face_affine(k, pad)
    if M is None:
        return None, 0, None
    return cv2.warpAffine(bgr, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR), R, Minv


def _pointer(bgr, k):
    """Angle of the dark line printed across the face, in NORMALISED degrees.

    Normalised meaning: measured on the face after it has been warped back to
    a circle, so the number is the knob's own rotation and does not change when
    the camera moves. Image-frame angles do change, which is why the loop kept
    comparing numbers that were not comparable.

    Scored against SIDE BANDS rather than on darkness alone. A ray is only a
    pointer if it is darker than the face immediately beside it: a shadow
    across the whole knob, or the dark gap where the skirt curves away, is
    just as dark as a printed line but its neighbours are dark too. The
    difference is what carries the signal.

        score(t) = brightness of the two flanking rays - brightness of this ray

    Returns (angle, contrast) with contrast the peak over the mean. Under
    about 2 nothing was found and the angle must not be used.
    """
    crop, R, _ = _circle_crop(bgr, k)
    if crop is None:
        return 0.0, 0.0
    V = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    radii = np.arange(0.20 * R, 0.90 * R, 1.0)
    if not len(radii):
        return 0.0, 0.0
    # flank far enough to be off the line, close enough to still be on the face
    dth = np.arctan2(3.0, 0.55 * R)

    def ray(t):
        xs = np.clip((R + radii * np.cos(t)).astype(int), 0, 2 * R - 1)
        ys = np.clip((R + radii * np.sin(t)).astype(int), 0, 2 * R - 1)
        return float(V[ys, xs].mean())

    score = np.zeros(360)
    for d in range(360):
        t = np.deg2rad(d)
        score[d] = 0.5 * (ray(t - dth) + ray(t + dth)) - ray(t)
    score = np.clip(score, 0, None)
    sm = np.convolve(np.r_[score[-12:], score, score[:12]],
                     np.ones(9) / 9, 'same')[12:-12]
    # +90 for cv2.fitEllipse's convention: it reports the angle of the box's
    # HEIGHT axis, so the warped face lands a quarter turn from where the
    # arithmetic suggests. Measured on synthetics with known painted angles,
    # the offset is a clean constant, and correcting it in the ROTATION
    # instead was tried and is wrong: that swaps which axis gets stretched
    # and the error stops being constant at all (163 to 179 degrees, all
    # over the place). Fix the number, not the warp.
    return float((sm.argmax() + 90) % 360), float(sm.max() / max(sm.mean(), 1e-6))


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
        fill, convex, (ex, ey), (major, minor), ang, (a1, a2) = got
        if fill < MIN_FILL:
            continue
        out.append({'cx': float(ex), 'cy': float(ey), 'major': float(major),
                    'minor': float(minor), 'angle_deg': float(ang),
                    'fill': float(fill), 'convexity': float(convex),
                    '_a1': float(a1), '_a2': float(a2)})

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

    out = _recover(bgr, out)
    out.sort(key=lambda k: k['cx'])
    if want_pointer:
        row = row_angle(out)
        for k in out:
            k['pointer'], k['pointer_contrast'] = _pointer(bgr, k)
            k['pointer_rel'] = float((k['pointer'] - row) % 360.0)
    return out


def row_angle(knobs):
    """Direction the knob row runs, in image degrees, by PCA on the centres.

    This is the panel's own reference direction, and pointer angles reported
    relative to it survive the camera being moved, which raw image angles do
    not.

    PCA over the centres rather than the ArUco tag's homography, deliberately.
    The tag is one small square and this bench has punished trusting it: it
    read as zero tags in twenty-five dictionaries when creased over the amp's
    corner, decoded as id 17 instead of 12 when half covered, which is a wrong
    answer rather than no answer, and it currently fuses with the GAIN knob
    and hides it. Seven centres spanning the whole panel are a far better
    conditioned line fit than one 112 px square, and they cannot be wrong
    about which object was measured. Use the tag to cross-check, not to anchor.
    """
    if len(knobs) < 2:
        return 0.0
    pts = np.array([[k['cx'], k['cy']] for k in knobs], float)
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    return float(np.degrees(np.arctan2(vt[0][1], vt[0][0])) % 180.0)


def _fit_local(bgr, cx, cy, want_major, span):
    """Look for one knob in a small box. -> knob dict or None.

    Otsu is run on the BOX, not the frame, and that is the point. Globally,
    the MASTER knob merges with the gold script beside it and GAIN merges with
    the tag and the jack, and a merged blob is not an ellipse so both are
    correctly but unhelpfully dropped. Inside a box barely bigger than one
    knob, the histogram is knob against panel and nothing else, so the split
    lands between them and the shapes come apart.
    """
    h, w = bgr.shape[:2]
    r = int(span / 2)
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(w, int(cx + r)), min(h, int(cy + r))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    box = bgr[y0:y1, x0:x1]
    V = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)[:, :, 2]
    bar, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (V >= bar).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best = None
    for c in cs:
        got = _shape(c)
        if got is None:
            continue
        fill, convex, (ex, ey), (major, minor), ang, (a1, a2) = got
        if fill < MIN_FILL:
            continue
        if not 0.7 * want_major <= major <= 1.4 * want_major:
            continue
        # nearest to where the layout said it would be, not merely present
        d = np.hypot(x0 + ex - cx, y0 + ey - cy)
        if d > 0.6 * want_major:
            continue
        if best is None or d < best[0]:
            best = (d, {'cx': float(x0 + ex), 'cy': float(y0 + ey),
                        'major': float(major), 'minor': float(minor),
                        'angle_deg': float(ang), 'fill': float(fill),
                        'convexity': float(convex), 'recovered': True,
                        '_a1': float(a1), '_a2': float(a2)})
    return best[1] if best else None


def _recover(bgr, knobs):
    """Look again where the layout says a knob should be but none was found.

    Only ever ADDS a knob that a local fit actually finds. Nothing is placed
    because the layout expects it: a knob hidden under the gripper must stay
    missing, since a centre invented for a shape whose edges cannot be seen is
    how the arm closes on air.
    """
    if len(knobs) < 3:
        return knobs
    ang = np.deg2rad(row_angle(knobs))
    u = np.array([np.cos(ang), np.sin(ang)])
    pts = np.array([[k['cx'], k['cy']] for k in knobs], float)
    along = pts @ u
    order = np.argsort(along)
    along, pts = along[order], pts[order]
    gaps = np.diff(along)
    if not len(gaps):
        return knobs
    g = float(np.median(gaps))
    major = float(np.median([k['major'] for k in knobs]))
    origin = pts[0] - along[0] * u

    wanted = []
    for i, gap in enumerate(gaps):                 # interior holes
        n = int(round(gap / g))
        for j in range(1, n):
            wanted.append(along[i] + j * g)
    for step in (1.0, 1.25, 1.5, 1.75, 2.0):       # just past either end
        wanted += [along[0] - step * g, along[-1] + step * g]

    found = list(knobs)
    for a in wanted:
        c = origin + a * u
        if any(np.hypot(k['cx'] - c[0], k['cy'] - c[1]) < 0.7 * major for k in found):
            continue
        got = _fit_local(bgr, c[0], c[1], major, 1.7 * major)
        if got is not None:
            found.append(got)
    found.sort(key=lambda k: k['cx'])
    return found


def circular_median(degrees):
    """Median direction of a set of angles, done on the unit circle.

    A plain median is wrong on angles: 359 and 1 average to 180, the exact
    opposite of the right answer.
    """
    if not len(degrees):
        return 0.0
    v = np.exp(1j * np.radians(np.asarray(degrees, float)))
    return float(np.degrees(np.angle(v.sum())) % 360.0)


def find_stable(frames, min_frac=0.6):
    """Knobs that hold still across frames, with angles taken as a circular
    median. -> the same dicts, plus 'seen' and 'of'.

    The arm throws a moving shadow across this panel and the camera reblurs on
    every reframe, so a single frame is a sample, not an answer. Position is
    voted on and the angle is a circular median of the frames that agreed.
    """
    votes = []
    for f in frames:
        for k in find(f):
            for v in votes:
                if np.hypot(v[0]['cx'] - k['cx'], v[0]['cy'] - k['cy']) < 0.5 * k['major']:
                    v.append(k)
                    break
            else:
                votes.append([k])
    need = max(2, int(round(min_frac * len(frames))))
    out = []
    for v in votes:
        if len(v) < need:
            continue
        k = dict(v[len(v) // 2])
        k['cx'] = float(np.median([x['cx'] for x in v]))
        k['cy'] = float(np.median([x['cy'] for x in v]))
        good = [x['pointer'] for x in v if x.get('pointer_contrast', 0) >= 2.0]
        k['pointer'] = circular_median(good) if good else 0.0
        k['pointer_contrast'] = float(np.median([x.get('pointer_contrast', 0) for x in v]))
        k['seen'], k['of'] = len(v), len(frames)
        out.append(k)
    out.sort(key=lambda k: k['cx'])
    return out


def draw(bgr, knobs):
    """Annotated copy, for eyeballing what was found."""
    img = bgr.copy()
    for i, k in enumerate(knobs, 1):
        c = (int(k['cx']), int(k['cy']))
        cv2.ellipse(img, c, (int(k['major'] / 2), int(k['minor'] / 2)),
                    k['angle_deg'], 0, 360, (0, 235, 0), 2)
        seg = pointer_segment(k) if k.get('pointer_contrast', 0) >= 2.0 else None
        if seg:
            cv2.line(img, seg[0], seg[1], (255, 60, 220), 2)
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
