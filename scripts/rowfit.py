#!/usr/bin/env python3
"""Teach the two ENDS of a knob row, let the camera fill in the middle.

    rowfit.py first last                    what it would save, no motion
    rowfit.py first last --save             write the in-between poses
    rowfit.py first last --verify=NAME      held-out accuracy test, see below

Why the ends and not two neighbours
-----------------------------------
The spacings are known ratios, so two taught poses fix the whole row. WHICH two
decides the error. Teaching neighbours 29 mm apart and extrapolating to a knob
225 mm away multiplies the teaching error by eight: 2 mm of hand placement
becomes 16 mm at the far end, and a knob is 25 mm across. Teaching the ends and
interpolating INSIDE them cannot amplify anything, so the worst case stays at
the 2 mm you started with. Same two teaches, eight times better.

Why the camera measures the spacings and not a ruler
----------------------------------------------------
The gaps on a Fender Rumble are not equal: four at 29 mm, two at 32, and 48
before GAIN. A teammate's table of those numbers is a table for HIS amp. The
camera already reports every knob's centre and cap radius, and cap radius is a
depth gauge, so the physical gaps come off our own panel:

    distance   d  = FX * CAP_MM / (2 * cap_r)
    gap        dx = du * d / FX = du * CAP_MM / (2 * cap_r)

FX cancels. So does CAP_MM, and that one matters: the fit works in the
FRACTION of the way along the row, and both the numerator and the span carry
the same CAP_MM, so it divides out entirely. Checked against the real numbers:
with CAP_MM left at the pedal's 17 mm the gaps come back 22 to 25 mm on knobs
that are really 29 to 32, a uniform 25 percent short, and the fitted positions
are unaffected because every gap is short by the same factor.

That is the useful property here. The camera supplies the SHAPE of the row,
which is what it measures well, and the two taught poses supply the SIZE,
which is what the arm knows. Neither has to be independently calibrated, and
no unverified camera-to-arm transform appears anywhere.

Held-out check
--------------
--verify=NAME takes a THIRD taught pose, leaves it out of the fit, and reports
how far the prediction lands from it in millimetres. That is the honest number:
everything else here is the model agreeing with itself.
"""
import json
import sys

import numpy as np

import arm as A
import ik
import knob

FLAGS = [a for a in sys.argv[1:] if a.startswith('--')]
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
SAVE = '--save' in FLAGS
VERIFY = next((f.split('=', 1)[1] for f in FLAGS if f.startswith('--verify=')), None)


def row_positions(frames=9):
    """-> [(name, cx, cy, cap_r, along_mm)] left to right, physical mm along the row.

    along_mm is cumulative, starting at 0 on the first knob, and each gap is
    corrected for how far away that part of the row is. Perspective is not
    small here: the near end of the Fender read 285 mm and the far end 322, so
    a gap in pixels is worth 13 percent less at one end than the other.
    """
    kept, flicker = knob.find_knobs_stable(knob.grab_frames(frames))
    if len(kept) < 3:
        raise SystemExit(f'only {len(kept)} knobs held still over {frames} '
                         f'frames ({len(flicker)} flickering). Fix the view '
                         f'first: knob.py calibrate')
    ks = sorted(kept.items(), key=lambda kv: kv[1]['cx'])
    out, along = [], 0.0
    for i, (name, k) in enumerate(ks):
        if i:
            prev = ks[i - 1][1]
            du = float(np.hypot(k['cx'] - prev['cx'], k['cy'] - prev['cy']))
            r = 0.5 * (k['cap_r'] + prev['cap_r'])
            along += du * knob.CAP_MM / (2.0 * r)
        out.append((name, k['cx'], k['cy'], k['cap_r'], along))
    return out


def fit(first, last, row):
    """-> {name: counts} for every knob between the two taught ends."""
    poses = A.poses()
    for n in (first, last):
        if n not in poses:
            raise SystemExit(f'no taught pose "{n}". Teach the ends first:\n'
                             f'  arm.py teach {n}   then   arm.py settle {n}')
    c0, c1 = list(A.counts_of(poses[first])), list(A.counts_of(poses[last]))
    p0, p1 = A.endpoint(c0) * 1000.0, A.endpoint(c1) * 1000.0
    span_mm = float(np.linalg.norm(p1 - p0))
    span_px = row[-1][4] - row[0][4]
    print(f'{first} -> {last}: {span_mm:.1f} mm apart by the arm, '
          f'{span_px:.1f} mm by the camera '
          f'({100*abs(span_mm-span_px)/max(span_mm,1):.0f}% apart)')

    out = {}
    for name, cx, cy, cap_r, along in row:
        t = (along - row[0][4]) / max(span_px, 1e-6)
        d = (p1 - p0) * t + p0 - A.endpoint(c0) * 1000.0
        try:
            counts, got = ik.nudged(c0, float(d[0]), float(d[1]), float(d[2]),
                                    max_mm=span_mm + 50.0)
        except ValueError as e:
            print(f'  {name} at t={t:.2f}: unreachable ({e})')
            continue
        out[name] = (counts, t, cx, cy)
    return out, c0, c1, p0, p1


def main():
    if len(ARGS) < 2:
        raise SystemExit(__doc__)
    first, last = ARGS[0], ARGS[1]
    row = row_positions()
    print(f'camera sees {len(row)} knobs; gaps in mm along the row:')
    print('   ' + '  '.join(f'{row[i+1][4]-row[i][4]:.1f}'
                            for i in range(len(row) - 1)))

    fitted, c0, c1, p0, p1 = fit(first, last, row)

    if VERIFY:
        held = A.poses().get(VERIFY)
        if held is None:
            raise SystemExit(f'no taught pose "{VERIFY}" to verify against')
        truth = A.endpoint(A.counts_of(held)) * 1000.0
        best = min(fitted.items(),
                   key=lambda kv: np.linalg.norm(A.endpoint(kv[1][0]) * 1000.0 - truth))
        err = A.endpoint(best[1][0]) * 1000.0 - truth
        print(f'\nHELD OUT: {VERIFY} was taught but not fitted.')
        print(f'  nearest prediction is {best[0]}, off by '
              f'({err[0]:+.1f}, {err[1]:+.1f}, {err[2]:+.1f}) mm, '
              f'|{np.linalg.norm(err):.1f}| mm')
        print(f'  a knob is {knob.CAP_MM:.0f} mm across, so under about '
              f'{knob.CAP_MM/4:.0f} mm is a grip and over it is not')

    print('\npredicted poses:')
    floor = A.z_floor()
    for name, (counts, t, cx, cy) in sorted(fitted.items(), key=lambda kv: kv[1][1]):
        z = float(A.endpoint(counts)[2]) * 1000
        flag = '' if floor is None or z >= floor * 1000 else '   BELOW THE FLOOR'
        print(f'  {name:8s} t={t:4.2f}  px=({cx},{cy})  z={z:5.1f} mm  '
              f'{counts}{flag}')

    if not SAVE:
        print('\ndry run. Add --save to write these into poses.json')
        return
    poses = A.poses()
    for name, (counts, t, cx, cy) in fitted.items():
        poses[name] = {'counts': [int(c) for c in counts], 'from': 'rowfit',
                       'between': [first, last], 't': round(float(t), 4)}
    json.dump(poses, open(A.POSES, 'w'), indent=1)
    print(f'\nsaved {len(fitted)} poses. Each one still has to be verified '
          f'before it is trusted:  arm.py settle <name>')


if __name__ == '__main__':
    main()
