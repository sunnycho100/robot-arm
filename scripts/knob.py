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
#
# WANT_ROUND is the found blob's bounding-box aspect ratio. It is TEMPTING to
# read that as cos(camera tilt), since a circular cap seen off-axis projects to
# an ellipse, and this was briefly raised to 0.90 on exactly that reasoning.
# It is wrong. The blob is the cap PLUS the pointer, which is just as bright
# and merges with it, so the box is stretched along the pointer no matter where
# the camera is. Measured straight down on the model, zero tilt: 0.72 with the
# pointer visible, 1.00 with it darkened. The real bench photos read 0.63.
#
# So this cannot measure tilt, and a threshold set as though it could would
# call a perfect overhead view "too side-on" and send someone off to fix a
# camera that was already right. It is kept LOW on purpose, catching only a
# genuinely bad viewpoint. Judge tilt by looking at the pedal's top face
# instead, and see docs/RUNBOOK.md for what tilt actually costs.
WANT_CAP_PX, WANT_SHARP, WANT_ROUND = 24, 250, 0.60


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


# The cap's allowed size on screen, in pixels of blob area. This is a FRAMING
# constraint, not a tuning knob, and it is the one that will bite first if the
# camera gets remounted: a cap outside this window is silently not a knob.
#
#     cap diameter on screen = FX * CAP_MM / distance
#
# so with FX = 1441 and a 10 mm cap the window below accepts roughly 205 mm to
# 1000 mm, and with a 14 mm cap roughly 285 mm to 1400 mm. Anything CLOSER than
# that is rejected for being too big. Measure CAP_MM before trusting either
# number, because the whole window slides with it. `knob.py calibrate` prints
# the distance it infers, which is the check that matters.
CAP_AREA = (150, 4000)


def find_knobs(frame, rejects=None):
    """Locate every knob in the frame. Returns {name: {cx, cy, cap_r, skirt_r}}.

    Names are assigned by position, top to bottom, so they stay stable as long as
    the pedal does not get rearranged.

    Pass a list as `rejects` to find out what was thrown away and why. Finding
    nothing is the failure mode that costs the most bench time, because the
    frame looks fine to a human and the only message is that there are no
    knobs; every plausible blob and the gate that killed it is the difference
    between a two-minute fix and an afternoon.
    """
    white, dark = _masks(frame)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        white.astype(np.uint8), connectivity=8)

    def toss(a, why):
        if rejects is not None and a >= CAP_AREA[0] // 2:
            rejects.append((int(a), why))

    found = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (CAP_AREA[0] <= a <= CAP_AREA[1]):
            toss(a, f'{"too big" if a > CAP_AREA[1] else "too small"}: '
                    f'{a} px, want {CAP_AREA[0]}-{CAP_AREA[1]}'
                 + (', so the camera is TOO CLOSE' if a > CAP_AREA[1] else ''))
            continue
        if not (0.6 <= w / max(h, 1) <= 1.7):          # roughly circular
            toss(a, f'not round enough: {w}x{h}')
            continue
        if a / (w * h) < 0.55:                          # solid, not a ring or streak
            toss(a, f'not solid: fills {a/(w*h):.0%} of its box')
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        # Equivalent-circle radius, not half the bounding box: a slightly elliptical
        # cap seen off-axis would otherwise read bigger than it is and push every
        # radius derived from it outward, off the knob.
        cap_r = int(round(np.sqrt(a / np.pi)))
        # The cap must be ringed by black skirt just outside it. Sampled close in,
        # because further out is the pedal, and on a knob near the edge, the table.
        if np.mean([_at(dark, cx, cy, cap_r + 3, d) for d in range(0, 360, 5)]) < 0.5:
            toss(a, 'no black skirt around it, so it is not a knob cap')
            continue
        found.append({'cx': cx, 'cy': cy, 'cap_r': cap_r,
                      'skirt_r': int(round(cap_r * REACH)),
                      'roundness': min(w, h) / max(w, h)})

    found.sort(key=lambda k: k['cy'])
    return {f'knob{i+1}': f for i, f in enumerate(found)}


def _saved():
    """The calibrated knob positions, or None if there are none to use.

    Checked rather than trusted. A knobs.json written by an older version of
    this file stores {name: [x, y]}, which reaches the pointer reader and dies
    on `k['cx']` with "list indices must be integers" at the first camera read
    of a run, several layers away from the actual problem. Anything that is not
    the current shape is treated as no calibration at all, which falls back to
    finding the knobs live in this frame, and says so.
    """
    if not os.path.exists(CONFIG):
        return None
    try:
        saved = json.load(open(CONFIG))
        ok = (isinstance(saved, dict) and saved and
              all(isinstance(v, dict) and {'cx', 'cy', 'cap_r'} <= set(v)
                  for v in saved.values()))
    except Exception as e:
        saved, ok = None, False
        print(f'{CONFIG} will not parse ({e})', file=sys.stderr)
    if ok:
        return saved
    print(f'ignoring {CONFIG}: it is not in the current format, so the knobs '
          f'are being\nfound live instead. Run  python3 knob.py calibrate  to '
          f'replace it.', file=sys.stderr)
    return None


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
        knobs = _saved() or find_knobs(frame)
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
    rejects = []
    knobs = find_knobs(frame, rejects)
    print(f'knobs found: {len(knobs)}')
    if not knobs:
        print('  NOTHING FOUND. Point the camera at the pedal so the knob tops are\n'
              '  visible, and make sure the pedal is lit well enough that the metal\n'
              '  caps read bright.')
        if rejects:
            # Say what was nearly a knob. "Nothing found" on a frame that looks
            # perfectly good to a human is the most expensive message this tool
            # can print, and the reason is almost always in this list.
            print('\n  the biggest things it looked at and threw away:')
            for a, why in sorted(rejects, reverse=True)[:6]:
                print(f'    {a:7d} px  {why}')
            big = sum(1 for a, w in rejects if 'TOO CLOSE' in w)
            if big:
                print(f'\n  {big} blobs were rejected for being too big, which '
                      f'means the camera is\n  closer than this finder allows. '
                      f'Caps must land in {CAP_AREA[0]}-{CAP_AREA[1]} px of '
                      f'area\n  ({2*np.sqrt(CAP_AREA[0]/np.pi):.0f} to '
                      f'{2*np.sqrt(CAP_AREA[1]/np.pi):.0f} px across). Back the '
                      f'camera off and run this again.')
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
            flags.append('SQUASHED. Either very side-on, or the pointer has '
                         'merged into the cap; look at the frame before '
                         'moving the camera')
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


def selftest():
    """Check the finder and the pointer reader without a camera.

    This file is what the demo actually runs, and until now it had no
    automated check at all: `pointer.py` in cv/ is a separate, later
    implementation and passing there says nothing about this one. The bench
    photos cannot fill the gap either, since they are hand-held close-ups
    whose caps are five times too big for this finder's window, so the scene
    is synthetic and borrowed from pointer.py, starburst and all.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'cv'))
    from pointer import _synth

    def pedal(angles, r=25, scale=1, seed0=0):
        """Three knobs stacked, the way the finder names them: top to bottom."""
        return np.vstack([_synth(a, r=int(r * scale), size=int(200 * scale),
                                 seed=seed0 + i)[0]
                          for i, a in enumerate(angles)])

    def decoy(size=200, r=25):
        """A bright round blob of exactly knob size, with NO black ring.

        The right size, the right shape, the right brightness: everything the
        finder looks for except the skirt. Without one of these in the frame
        the dark-surround test can be deleted and every check here still
        passes, which makes them a description of the code rather than a test
        of it. Physically this is a highlight on the pedal body.
        """
        img = np.full((size, size, 3), 140, np.uint8)
        cv2.circle(img, (size // 2, size // 2), r, (188, 190, 192), -1)
        return cv2.GaussianBlur(img, (7, 7), 0)

    want = [0.0, 90.0, -140.0]
    frame = np.vstack([pedal(want), decoy()])
    found = find_knobs(frame)
    assert len(found) == 3, (f'found {len(found)} knobs in a scene with three '
                             f'knobs and one skirtless impostor')
    read = measure(frame, found)
    for (name, f), w in zip(sorted(read.items()), want):
        err = abs((f['angle'] - w + 180) % 360 - 180)
        assert err < 5.0, f'{name}: read {f["angle"]:.0f}, painted {w:.0f}'
        assert f['contrast'] >= MIN_CONTRAST, \
            f'{name}: contrast {f["contrast"]:.1f} on a clean synthetic pointer'
    print(f'  3 knobs found, pointers within 5 deg of where they were painted')

    # The sign convention is load-bearing: turn_knob.py subtracts these angles
    # to decide which way the next bite goes, so a flipped sign sends the wrist
    # the wrong way and the run walks away from the target.
    f0, f1 = pedal([0.0] * 3), pedal([30.0] * 3)
    a0 = measure(f0, find_knobs(f0))['knob1']['angle']
    a1 = measure(f1, find_knobs(f1))['knob1']['angle']
    assert 20 < ((a1 - a0 + 180) % 360 - 180) < 40, \
        f'a +30 deg turn measured as {(a1 - a0 + 180) % 360 - 180:+.0f}: ' \
        f'angles must grow CLOCKWISE, as knob.py documents'
    print(f'  a +30 deg turn reads as {(a1 - a0 + 180) % 360 - 180:+.0f} deg, '
          f'clockwise as documented')

    # Too close is the failure the runbook can walk you into, so the finder has
    # to both fail and SAY SO. A silent zero is what costs bench time.
    rejects = []
    assert not find_knobs(pedal(want, scale=4), rejects), \
        'caps four times too big were accepted as knobs'
    close = [w for _, w in rejects if 'TOO CLOSE' in w]
    assert close, f'nothing was blamed on the camera being close: {rejects[:3]}'
    print(f'  caps 4x too big are rejected, and {len(close)} of them say why')

    # Pin the fact that cost an hour: roundness is NOT cos(tilt). The pointer
    # is as bright as the cap and merges with it, so the bounding box is
    # stretched even head-on. Anyone tempted to raise WANT_ROUND to grade the
    # camera angle should fail here first.
    head_on = pedal([0.0, 0.0, 0.0])
    worst_round = min(v['roundness'] for v in find_knobs(head_on).values())
    assert worst_round < 0.95, (f'a head-on knob measured {worst_round:.2f} '
                                f'round, so the pointer no longer merges with '
                                f'the cap and this metric may have become '
                                f'usable as a tilt gauge. Re-derive before '
                                f'trusting it.')
    assert WANT_ROUND < worst_round, (
        f'WANT_ROUND is {WANT_ROUND}, but a knob seen straight on only reads '
        f'{worst_round:.2f}. calibrate would call a perfect view "squashed".')

    # An empty scene must stay empty. Without this the checks above pass just
    # as well on a finder that calls everything a knob.
    blank = np.full((600, 200, 3), 200, np.uint8)
    assert not find_knobs(blank), 'found knobs in a blank frame'
    print('  a blank frame yields nothing')

    # A stale knobs.json must not reach the pointer reader. turn_knob.py calls
    # measure() with no knobs, so whatever is on disk is what the first camera
    # read of the demo uses, and the old {name: [x, y]} shape died there with
    # "list indices must be integers" three layers from the cause.
    global CONFIG
    keep, CONFIG = CONFIG, os.path.join(HERE, '_selftest_stale.json')
    try:
        json.dump({'top': [835, 442], 'left': [787, 483]}, open(CONFIG, 'w'))
        assert _saved() is None, 'the old knobs.json format was accepted'
        got = measure(frame)              # the exact call turn_knob.py makes
        assert len(got) == 3, f'fell back to live finding but got {len(got)}'
        json.dump('not even a dict', open(CONFIG, 'w'))
        assert _saved() is None, 'a garbage knobs.json was accepted'
    finally:
        if os.path.exists(CONFIG):
            os.remove(CONFIG)
        CONFIG = keep
    print('  a stale knobs.json is ignored, not followed into a crash')
    print('knob self-checks passed')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
    else:
        main()
