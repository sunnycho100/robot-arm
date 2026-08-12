#!/usr/bin/env python3
"""Where can the arm sit without blocking its own camera?

    python3 parkscan.py [base counts, comma separated] [--save=park]

The camera looks down at the pedal and the arm reaches in under it, so the arm
is in its own picture. Measured on this bench: with the arm forward, the
consensus finder saw 2 of 3 knobs and the base tag vanished entirely, and the
one frame where the base tag did decode came back as id 17 instead of 12,
because a partly covered marker still decodes, just wrongly. A wrong id is
worse than none.

So the park pose is not a matter of taste. This swings the base through a few
angles and reports, at each one, how many knobs hold still over nine frames and
which tags decode. Keep the angle where everything is visible, and look from
there before every move.
"""
import sys

import numpy as np

import arm as A
import knob

sys.path.insert(0, '/home/pi/cv')
import compat

ANGLES = [int(v) for v in
          (next((a for a in sys.argv[1:] if not a.startswith('--')),
                '300,380,460,540,620,700')).split(',')]
SAVE = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--save=')), None)


def path_is_clear(now, target, floor):
    """Same check as safemove: every interpolation step, not just the target."""
    zs = [float(A.endpoint([round(a + (b - a) * (k / 60))
                            for a, b in zip(now, target)])[2]) * 1000
          for k in range(61)]
    return min(zs) >= floor * 1000, min(zs)


def look():
    """-> (stable knob names, sorted tag ids). Nine frames, consensus only."""
    frames = knob.grab_frames(9)
    if not frames:
        return [], []
    kept, _flickering = knob.find_knobs_stable(frames)
    ks = sorted(kept)
    ids = set()
    for f in frames:
        _, found, _ = compat.detect_any(f)
        ids.update(int(i) for i in found)
    return ks, sorted(ids)


def main():
    floor = A.z_floor()
    start = [int(np.clip(c, 0, 1000)) for c in A.read()]
    results = []
    for base in ANGLES:
        target = list(start)
        target[0] = base
        clear, low = path_is_clear([int(np.clip(c, 0, 1000)) for c in A.read()],
                                   target, floor)
        if not clear:
            print(f'base {base}: SKIPPED, path drops to {low:.1f} mm '
                  f'(floor {floor*1000:.1f})')
            continue
        A.move(target, speed=35)
        ks, ids = look()
        results.append((len(ks), len(ids), base, ks, ids))
        print(f'base {base}: {len(ks)} knobs {ks}   tags {ids}')

    if not results:
        raise SystemExit('nothing was reachable, so nothing was measured')
    best = max(results)
    print(f'\nbest: base {best[2]} with {best[0]} knobs and {best[1]} tags')
    target = list(start)
    target[0] = best[2]
    A.move(target, speed=35)
    if SAVE:
        A.save(SAVE, A.look_safe())
        print(f'saved as pose "{SAVE}"')


if __name__ == '__main__':
    main()
