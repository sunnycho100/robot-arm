#!/usr/bin/env python3
"""Find knobs on the pedal. Classical tier; ML tier lives in knobs_ml.py.

What the pixels actually look like, measured on the bench photos rather than
assumed:

  the orange pedal      hue 13ish, saturation 150+
  the black knob body   any hue, very low value
  the metal cap on top  low saturation (grey), value above the body
  the white pointer     a tab sticking out of the cap, so the cap blob is a
                        circle with a bump, not a clean circle

Two traps that cost real debugging time and are encoded in the thresholds:

1. Brightness alone does NOT find the cap. Under a hot exposure the orange
   pedal is brighter than the grey cap, so a "top N percent brightest" rule
   selects the pedal. Saturation is the honest discriminator: the cap is
   grey, the pedal is vivid. The bright/dark split is then made by Otsu on
   just the low-saturation pixels.
2. The white cells of an ArUco tag are also grey and bright, so they look
   exactly like a cap. What separates them is shape: a square fills its
   bounding box (ratio near 1.0) while a disc fills about 0.785. Hence the
   UPPER bound on bounding-box fill. A lower bound alone lets every tag cell
   through, which is what happened first.

The final say belongs to the dark-ring test: a real cap is ringed by the
black knob body, and nothing else in the scene is.

    knobs = find(frame)                       # pixel coordinates
    knobs = find(frame, view=plane_view)      # adds mm in the pedal frame
"""
import numpy as np
import cv2

# hue band of the DS-1 body; wide enough for blown highlights
ORANGE_H = (3, 32)
ORANGE_S = 55
GREY_S = 90          # above this a pixel is coloured, so not a metal cap
RING = 1.35          # sample the knob body at this multiple of the cap radius
DARK_FRAC = 0.45     # this much of that ring must be dark to accept the cap
AREA_FRAC = (0.0015, 0.08)    # cap area as a fraction of the frame
ASPECT = (0.65, 1.55)
BBFILL = (0.50, 0.88)         # upper bound is what rejects square tag cells
CIRCLE_FILL = 0.50            # area / area of min enclosing circle


def _cap_mask(hsv):
    """Pixels that could be a metal cap, plus the value cut used to find them."""
    H, S, V = cv2.split(hsv)
    orange = (H >= ORANGE_H[0]) & (H <= ORANGE_H[1]) & (S >= ORANGE_S)
    grey = (~orange) & (S < GREY_S)
    if grey.sum() < 500:
        return None, None
    cut, _ = cv2.threshold(V[grey].astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cap = (grey & (V > cut)).astype(np.uint8)
    cap = cv2.morphologyEx(cap, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cap = cv2.morphologyEx(cap, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cap, float(cut)


def _dark_ring(V, cx, cy, r, cut, ring=RING):
    """Fraction of a ring at r*ring that is darker than the cap/body cut."""
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rr = r * ring
    ys = np.clip((cy + rr * np.sin(th)).astype(int), 0, V.shape[0] - 1)
    xs = np.clip((cx + rr * np.cos(th)).astype(int), 0, V.shape[1] - 1)
    return float((V[ys, xs].astype(float) < cut).mean())


def find(frame, view=None, dark_frac=DARK_FRAC):
    """-> [{name, cx, cy, r_px, conf, source, cx_mm, cy_mm, r_mm}], best first.

    view: an aruco_locate.PlaneView. When given, millimetre fields are filled
    in and the list is sorted along the pedal's own x axis so knob names stay
    stable as the camera moves.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    cap, cut = _cap_mask(hsv)
    if cap is None:
        return []
    area = frame.shape[0] * frame.shape[1]
    cnts, _ = cv2.findContours(cap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (AREA_FRAC[0] * area < a < AREA_FRAC[1] * area):
            continue
        (cx, cy), r_min = cv2.minEnclosingCircle(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect, bbfill = w / max(h, 1), a / (w * h)
        circle_fill = a / (np.pi * r_min * r_min)
        if not (ASPECT[0] < aspect < ASPECT[1]):
            continue
        if not (BBFILL[0] < bbfill < BBFILL[1]):
            continue
        if circle_fill < CIRCLE_FILL:
            continue
        r = float(np.sqrt(a / np.pi))          # equivalent-circle radius
        df = _dark_ring(V, cx, cy, r, cut)
        if df < dark_frac:
            continue
        out.append(dict(cx=float(cx), cy=float(cy), r_px=r, conf=round(df, 3),
                        source='classical'))
    if view is not None:
        for k in out:
            k['cx_mm'], k['cy_mm'] = view.to_mm(k['cx'], k['cy'])
            edge = view.to_mm(k['cx'] + k['r_px'], k['cy'])
            k['r_mm'] = float(np.hypot(edge[0] - k['cx_mm'], edge[1] - k['cy_mm']))
        out.sort(key=lambda k: (k['cx_mm'], k['cy_mm']))
    else:
        out.sort(key=lambda k: -k['conf'])
    for i, k in enumerate(out):
        k['name'] = f'knob{i}'
    return out


def draw(frame, knobs, colour=(0, 255, 0)):
    vis = frame.copy()
    for k in knobs:
        cv2.circle(vis, (int(k['cx']), int(k['cy'])), int(k['r_px']), colour, 3)
        cv2.putText(vis, f"{k['name']} {k['conf']:.2f}",
                    (int(k['cx']) - 45, int(k['cy'] - k['r_px']) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return vis


if __name__ == '__main__':
    import json, pathlib, sys
    # The bench photos are private (docs/ is gitignored), so this self-check
    # runs against them when present and falls back to a synthetic scene.
    docs = pathlib.Path(__file__).resolve().parents[2] / 'docs' / 'bench_photos'
    man = docs / 'manifest.json'
    if man.exists():
        m = json.load(open(man))
        tests = sorted(n for n, v in m.items()
                       if v['orientation'] == 'topdown' and v['knobs'] >= 2
                       and not v['overlay'])
        hits = fails = 0
        for n in tests:
            ks = find(cv2.imread(str(docs / n)))
            print(f'  {n[:22]}: {len(ks)} knobs {[k["conf"] for k in ks]}')
            hits += len(ks) >= 2
            fails += len(ks) == 0
        print(f'{hits}/{len(tests)} frames with 2+ knobs, {fails} found nothing')
        assert hits >= 6, f'only {hits} frames reached 2 knobs'
    else:
        print('bench photos not present, synthetic check only')

    # synthetic: orange field, three dark discs with grey caps and a tab
    img = np.full((600, 800, 3), 0, np.uint8)
    img[:] = (40, 120, 240)                                    # BGR orange
    truth = [(200, 300), (400, 200), (600, 350)]
    for (cx, cy) in truth:
        cv2.circle(img, (cx, cy), 70, (25, 25, 30), -1)        # body
        cv2.circle(img, (cx, cy), 45, (190, 192, 195), -1)     # cap
        cv2.circle(img, (cx + 38, cy - 20), 9, (240, 240, 240), -1)  # pointer tab
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ks = find(img)
    assert len(ks) == 3, f'synthetic scene: expected 3 knobs, got {len(ks)}'
    for k in ks:
        near = min(np.hypot(k['cx'] - tx, k['cy'] - ty) for tx, ty in truth)
        assert near < 8, f'knob centre off by {near:.1f} px'
    print(f'synthetic: 3/3 knobs, centres within 8 px, radii '
          f'{[round(k["r_px"]) for k in ks]}')

    # a square grey patch (an ArUco cell) must NOT be called a knob
    tag = img.copy()
    cv2.rectangle(tag, (60, 60), (150, 150), (200, 200, 200), -1)
    assert len(find(tag)) == 3, 'a square patch was mistaken for a knob'
    print('square tag cell correctly rejected')
    print('knobs2 self-checks passed')
