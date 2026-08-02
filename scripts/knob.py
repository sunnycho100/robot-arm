#!/usr/bin/env python3
"""Find the pedal knobs by themselves, and read the white pointer on each.

  knob.py calibrate     look at the scene, grade the camera setup, save knobs.json
  knob.py               measure the pointer angles now
  knob.py mark          measure and store the result as the reference
  knob.py --shot=x.jpg  put the annotated frame somewhere else

Nothing about the knob positions is hardcoded. `calibrate` locates them in
whatever frame it is given, so a new bench setup only needs the camera pointed at
the pedal and one calibrate run. Numbers that would go stale, centres, radii,
scale, all get re-derived rather than remembered.


How a knob is recognised
------------------------
An aluminium cap is a compact, bright, UNSATURATED blob sitting in a dark
surround. That combination is rare in this scene, and each part of it is doing
work:

  compact + bright      finds the cap and rejects cable runs and table edges
  unsaturated           rejects the orange pedal body, which is bright in grey
  dark surround         rejects bright things not ringed by a black skirt

Its cap radius comes from the blob's area, not its bounding box, so a cap seen a
little off-axis does not read bigger than it is and push every derived radius out
past the knob. Every radius is per knob and per frame, so perspective making the
far knob smaller on screen costs nothing.

It is not magic, and it does fail on a bad viewpoint. On the earlier, much more
side-on camera position it returned three knobs, but one was a false positive on
the robot and one real knob was missed. That is what `calibrate` is for: it grades
the view and says what to change, rather than quietly handing back three numbers.


How the pointer is found, and why the obvious way fails
-------------------------------------------------------
Three bright things sit near the pointer and none of them is the pointer:

  - the brushed cap, whose starburst is fixed by the LIGHTING, not the knob, so
    it does not turn when the knob does
  - the orange pedal body, bright in greyscale, and only weakly saturated where
    it falls into shadow at the rim
  - the grey table beyond the pedal edge, bright and unsaturated, exactly like
    white paint

Sampling a ring and taking the brightest angle picks all three. Measured on a
real frame, that put the top knob 94 degrees out while reporting a contrast of
2.7, which reads as healthy.

Two things separate the pointer from all three, and it needs both.

**It is as bright as the cap.** The pointer is white paint and blows the sensor
out, the same as the aluminium beside it. The impostors do not. Measured on one
frame, along the true pointer V ran 249-255, while the grey background that fooled
the old version peaked at 208. So the threshold is taken from each knob's OWN cap
brightness rather than fixed, which is what lets it survive a lighting change
instead of needing a new magic number.

**It ends.** The pointer is painted on the skirt, so the skirt closes around it.
Everything else runs off the knob and keeps going. So an angle scores the length
of its contiguous bright run, and a run still going at the search limit scores
zero, because it left the knob.


What the angle means
--------------------
Image coordinates, y down, so it grows clockwise on screen. The camera looks at
the pedal a little off-axis, so this is a projection of the true knob angle. Good
for "did it move, and roughly how far", which is what the run needs. Not a
calibrated angle, do not quote it as one.
"""
import json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'knobs.json')
REF = os.path.join(HERE, 'knob_ref.json')

S_MAX, V_MIN = 90, 150      # the metal cap: unsaturated and clearly brighter than the skirt
V_DARK = 90                 # the black skirt
CAP_FRAC = 0.88             # pointer must be this fraction of its own cap's brightness
REACH = 1.8                 # search out to this multiple of the cap radius
MIN_CONTRAST = 2.0          # below this, treat the angle as not found
FX = 1441.0                 # C270 focal length in px at 1280x960, course calibration
CAP_MM = 10.0               # metal cap diameter. MEASURE YOURS, the distance scales with it

# A good setup, for the calibrate report to grade against.
WANT_CAP_PX, WANT_SHARP, WANT_ROUND = 24, 250, 0.75


def _masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    return ((S < S_MAX) & (V > V_MIN)), (V < V_DARK)


def _at(mask, cx, cy, r, deg):
    t = np.deg2rad(deg)
    y, x = int(cy + r * np.sin(t)), int(cx + r * np.cos(t))
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return bool(mask[y, x])
    return False


def _ring_vals(V, cx, cy, r, step=5):
    return [V[int(cy + r * np.sin(np.deg2rad(d))), int(cx + r * np.cos(np.deg2rad(d)))]
            for d in range(0, 360, step)
            if 0 <= int(cy + r * np.sin(np.deg2rad(d))) < V.shape[0]
            and 0 <= int(cx + r * np.cos(np.deg2rad(d))) < V.shape[1]]


def find_knobs(frame):
    """Locate every knob in the frame. Returns {name: {cx, cy, cap_r, skirt_r}}.

    Names are assigned by position, top to bottom, so they stay stable as long as
    the pedal does not get rearranged.
    """
    white, dark = _masks(frame)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        white.astype(np.uint8), connectivity=8)

    found = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (150 <= a <= 4000):
            continue
        if not (0.6 <= w / max(h, 1) <= 1.7):          # roughly circular
            continue
        if a / (w * h) < 0.55:                          # solid, not a ring or streak
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        # Equivalent-circle radius, not half the bounding box: a slightly elliptical
        # cap seen off-axis would otherwise read bigger than it is and push every
        # radius derived from it outward, off the knob.
        cap_r = int(round(np.sqrt(a / np.pi)))
        # The cap must be ringed by black skirt just outside it. Sampled close in,
        # because further out is the pedal, and on a knob near the edge, the table.
        if np.mean([_at(dark, cx, cy, cap_r + 3, d) for d in range(0, 360, 5)]) < 0.5:
            continue
        found.append({'cx': cx, 'cy': cy, 'cap_r': cap_r,
                      'skirt_r': int(round(cap_r * REACH)),
                      'roundness': min(w, h) / max(w, h)})

    found.sort(key=lambda k: k['cy'])
    return {f'knob{i+1}': f for i, f in enumerate(found)}


def _pointer(V, S, cx, cy, cap_r):
    """Score every angle by the length of its bounded bright run. Returns profile.

    The brightness bar is set by this knob's own cap, so a dimmer scene lowers the
    bar for the pointer by the same amount and the test still holds.
    """
    cap_v = np.median(_ring_vals(V, cx, cy, max(2, int(cap_r * 0.6)), step=3))
    thresh = max(160.0, CAP_FRAC * float(cap_v))
    limit = max(cap_r + 4, int(round(cap_r * REACH)))

    score = np.zeros(360)
    for d in range(360):
        t = np.deg2rad(d)
        run, gap, r = 0, 0, cap_r + 1
        while r <= limit:
            y, x = int(cy + r * np.sin(t)), int(cx + r * np.cos(t))
            if not (0 <= y < V.shape[0] and 0 <= x < V.shape[1]):
                break
            if V[y, x] >= thresh and S[y, x] < S_MAX:
                run, gap = run + 1, 0
            else:
                gap += 1
                if gap > 1:                 # tolerate single-pixel dropouts
                    break
            r += 1
        # Still bright at the limit means this ray left the knob: not a pointer.
        score[d] = 0 if r > limit else run
    return np.convolve(np.r_[score[-15:], score, score[:15]],
                       np.ones(11) / 11, 'same')[15:-15], thresh


def measure(frame, knobs=None):
    """{name: {'angle', 'contrast', ...}} for every knob.

    contrast is peak over mean of the score profile. Below MIN_CONTRAST the
    pointer was not distinguishable and the angle must not be trusted.
    """
    if knobs is None:
        knobs = json.load(open(CONFIG)) if os.path.exists(CONFIG) else find_knobs(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    out = {}
    for name, k in knobs.items():
        p, thresh = _pointer(V, S, k['cx'], k['cy'], k['cap_r'])
        out[name] = dict(k, angle=float(p.argmax()), thresh=round(thresh),
                         contrast=float(p.max() / max(p.mean(), 1e-6)))
    return out


def turned(before, after):
    """Signed change per knob, wrapped into -180..180. Missing knobs are skipped."""
    return {k: (after[k]['angle'] - before[k]['angle'] + 180) % 360 - 180
            for k in before if k in after}


def annotate(frame, found, path=None):
    out = frame.copy()
    for name, f in found.items():
        cx, cy, r = f['cx'], f['cy'], f['skirt_r']
        ok = f.get('contrast', 9) >= MIN_CONTRAST
        cv2.circle(out, (cx, cy), f['cap_r'], (255, 255, 0), 1)
        cv2.circle(out, (cx, cy), r, (255, 255, 0), 1)
        t = np.deg2rad(f['angle'])
        cv2.line(out, (cx, cy), (int(cx + (r + 8) * np.cos(t)),
                                 int(cy + (r + 8) * np.sin(t))),
                 (0, 0, 255) if ok else (0, 165, 255), 2)
        cv2.putText(out, f'{name} {f["angle"]:.0f}', (cx - 30, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if path:
        cv2.imwrite(path, out)
    return out


def grab():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    ok = False
    for _ in range(8):
        ok, frame = cam.read()
    cam.release()
    if not ok:
        sys.exit('no frame from the camera. Is camstream.py holding it? '
                 'Stop it with:  pkill -f "[c]amstream"')
    return frame


def check(frame):
    """Grade the camera setup and print what to change. Returns the knobs found."""
    knobs = find_knobs(frame)
    print(f'knobs found: {len(knobs)}')
    if not knobs:
        print('  NOTHING FOUND. Point the camera at the pedal so the knob tops are\n'
              '  visible, and make sure the pedal is lit well enough that the metal\n'
              '  caps read bright.')
        return knobs

    found = measure(frame, knobs)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok = True
    for name, f in sorted(found.items()):
        pad = f['skirt_r'] + 4
        patch = grey[max(0, f['cy']-pad):f['cy']+pad, max(0, f['cx']-pad):f['cx']+pad]
        sharp = cv2.Laplacian(patch, cv2.CV_64F).var()
        # From the CAP, the only radius actually measured. skirt_r is a derived
        # search bound (cap x REACH), so scaling off it would be circular.
        dist = FX * CAP_MM / max(2 * f['cap_r'], 1)
        flags = []
        if 2 * f['cap_r'] < WANT_CAP_PX:
            flags.append('SMALL, move the camera closer')
        if sharp < WANT_SHARP:
            flags.append('SOFT, past the fixed-focus limit, back off a little')
        if f['roundness'] < WANT_ROUND:
            flags.append('SQUASHED, too side-on, get more overhead')
        if f['contrast'] < MIN_CONTRAST:
            flags.append('POINTER NOT FOUND, check lighting and glare')
        ok &= not flags
        print(f'  {name}: at ({f["cx"]},{f["cy"]})  cap {2*f["cap_r"]}px  '
              f'round {f["roundness"]:.2f}  '
              f'sharp {sharp:.0f}  pointer {f["angle"]:.0f} deg '
              f'(contrast {f["contrast"]:.1f})  ~{dist:.0f}mm away')
        for fl in flags:
            print(f'      -> {fl}')

    print(f'\nscale assumes a {CAP_MM:.0f} mm metal cap. Measure yours and edit CAP_MM, '
          f'the distance scales straight off it.')
    print('SETUP OK' if ok else 'SETUP NEEDS WORK, see the arrows above')
    return knobs


def main():
    flag = lambda k, d: next((type(d)(a.split('=')[1])
                              for a in sys.argv[1:] if a.startswith(f'--{k}=')), d)
    shot = flag('shot', os.path.expanduser('~/images/knobs.jpg'))
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else ''
    frame = grab()

    if cmd == 'calibrate':
        knobs = check(frame)
        if knobs:
            json.dump(knobs, open(CONFIG, 'w'), indent=1)
            print(f'saved {len(knobs)} knobs to {CONFIG}')
            annotate(frame, measure(frame, knobs), shot)
            print(f'annotated frame -> {shot}\nLOOK AT IT. The circles must sit on '
                  f'the knobs and each red line on its white pointer.')
        return

    found = measure(frame)
    annotate(frame, found, shot)
    ref = json.load(open(REF)) if os.path.exists(REF) else None
    for name, f in sorted(found.items()):
        warn = '' if f['contrast'] >= MIN_CONTRAST else '   <- NOT FOUND, do not trust'
        line = (f'  {name:8s} {f["angle"]:6.1f} deg   '
                f'contrast {f["contrast"]:5.2f}{warn}')
        if ref and name in ref:
            line += f'   moved {(f["angle"] - ref[name] + 180) % 360 - 180:+6.1f}'
        print(line)

    if cmd == 'mark':
        json.dump({k: v['angle'] for k, v in found.items()}, open(REF, 'w'), indent=1)
        print(f'reference saved to {REF}')
    print(f'annotated frame -> {shot}')


if __name__ == '__main__':
    main()
