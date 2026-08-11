#!/usr/bin/env python3
"""What breaks the cycle, and at which stage.

The nominal setup works. The question that matters for a demo is what happens
when it is not nominal: the pedal further away, rotated, sitting on something,
a different pedal with wider knobs. The assumption throughout is that ArUco
gives us the pedal's pose and scale, so the knob positions are KNOWN in each
case; what is being tested is whether the arm can then physically do the job.

That distinction is the point. A failure here is not a perception failure. It
is the arm running out of reach, or the gripper running out of room between
neighbouring knobs. Those are the limits worth knowing before demo day,
because they decide where the pedal gets taped down.

    python3 edge_cases.py           # the sweep, as a table
    python3 edge_cases.py -v        # plus the stage-by-stage log for failures
"""
import sys
import numpy as np
import scene
import cycle

VERBOSE = '-v' in sys.argv

# (label, pedal overrides). Distances in metres, angles in radians.
CASES = [
    ('nominal',                 {}),
    ('20 mm nearer',            dict(x=0.180)),
    ('20 mm further',           dict(x=0.220)),
    ('40 mm further',           dict(x=0.240)),
    ('60 mm further',           dict(x=0.260)),
    ('80 mm further',           dict(x=0.280)),
    ('30 mm left',              dict(y=0.030)),
    ('30 mm right',             dict(y=-0.030)),
    ('60 mm left',              dict(y=0.060)),
    ('rotated 15 deg',          dict(yaw=np.radians(15))),
    ('rotated 30 deg',          dict(yaw=np.radians(30))),
    ('rotated 45 deg',          dict(yaw=np.radians(45))),
    ('raised 20 mm',            dict(h=0.075)),
    ('raised 40 mm',            dict(h=0.095)),
    ('lowered 20 mm',           dict(h=0.035)),
    ('tall knobs (+6 mm)',      dict(knob_h=0.020)),
    ('short knobs (-6 mm)',     dict(knob_h=0.008)),
    ('knobs 6 mm closer',       dict(knob_dy=0.018)),
    ('knobs 8 mm further',      dict(knob_dy=0.032)),
    ('fat knobs (r 13 mm)',     dict(knob_r=0.013, cap_r=0.007)),
    ('slim knobs (r 6 mm)',     dict(knob_r=0.006, cap_r=0.004)),
    ('far and rotated',         dict(x=0.245, yaw=np.radians(20))),
    ('near, raised, rotated',   dict(x=0.185, h=0.075, yaw=np.radians(15))),
]


def run_case(label, overrides, knobs=('knob0', 'knob1', 'knob2')):
    scene.reset()
    if overrides:
        scene.configure(**overrides)
    out = {}
    for k in knobs:
        try:
            out[k] = cycle.run_once(k, 90.0, verbose=False)
        except Exception as e:
            out[k] = dict(ok=False, stages=[dict(name='crash', ok=False,
                                                 note=f'{e.__class__.__name__}: {e}')])
    scene.reset()
    return out


def summarise(res):
    done = sum(1 for r in res.values() if r['ok'])
    if done == len(res):
        return 'all 3', ''
    stopped = {}
    for k, r in res.items():
        if not r['ok']:
            last = r['stages'][-1]
            stopped.setdefault(last['name'], []).append(k)
    why = '; '.join(f'{s}: {",".join(ks)}' for s, ks in stopped.items())
    return f'{done}/3', why


def main():
    print(f'{"case":26s} {"turned":>7}  why not')
    print('-' * 92)
    rows = []
    for label, ov in CASES:
        res = run_case(label, ov)
        got, why = summarise(res)
        rows.append((label, ov, res, got, why))
        print(f'{label:26s} {got:>7}  {why}')
        if VERBOSE and why:
            for k, r in res.items():
                if not r['ok']:
                    for st in r['stages']:
                        print(f'      {k} {"ok  " if st["ok"] else "STOP"} '
                              f'{st["name"]:9s} {st["note"]}')
    print()

    full = [l for l, _, _, g, _ in rows if g == 'all 3']
    partial = [l for l, _, _, g, _ in rows if g not in ('all 3', '0/3')]
    none = [l for l, _, _, g, _ in rows if g == '0/3']
    print(f'{len(full)} of {len(rows)} cases turn all three knobs')
    if partial:
        print(f'partial: {", ".join(partial)}')
    if none:
        print(f'none:    {", ".join(none)}')

    # The nominal case must work, or the sweep is measuring a broken pipeline
    # rather than the arm's limits.
    assert rows[0][3] == 'all 3', 'the nominal case failed; fix that first'
    return rows


if __name__ == '__main__':
    main()
