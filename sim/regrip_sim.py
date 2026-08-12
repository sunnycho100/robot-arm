#!/usr/bin/env python3
"""Does the regrip search recover a mis-taught pose against real geometry?

regrip.py's own self-check scores attempts with a toy falloff. This runs the
same policy against the model: real inverse kinematics, real collision
detection, the real gripper with its mimic joints applied, and the arm's own
1.5 mm of landing scatter. Nothing here is allowed to know the true error; the
search sees only what the bench would hand back, which is one force number and
whether the jaws closed on something.

    python3 regrip_sim.py            # the recovery envelope
    python3 regrip_sim.py -v         # every attempt of every trial
"""
import pathlib
import sys

import numpy as np
import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'scripts'))
import regrip
import scene
import cycle

VERBOSE = '-v' in sys.argv

# What the bench can actually tell us about an attempt. The jaws are 28 mm
# apart pre-closed and the knob is 19 mm across, so the fingers stop straddling
# it a few mm off centre; past EJECT_MM the knob squirts out under torque,
# which is the number cycle.py already uses.
HOLD_MM = cycle.EJECT_MM        # 4.0: lateral error a grip survives
FALLOFF_MM = 9.0                # force fades to nothing this far off the axis


def attempt(knob, offset_mm, rng, scatter_mm=1.5):
    """Fly to the taught pose plus this offset and report what the bench would.

    Returns (score, holding, note). The search never sees the true error, only
    these, exactly as on the Pi.
    """
    target = np.array(scene.knob_targets()[knob], float)
    want = target + np.r_[np.asarray(offset_mm, float) / 1000.0, 0.0]
    landed = want.copy()
    landed[:2] += rng.normal(0, scatter_mm / 1000.0, 2)

    pose = landed + cycle.GRIP_LIFT * scene.knob_normal()
    roll = cycle.choose_wrist_roll(pose, knob, cycle.PRECLOSE, 90.0)
    g = scene.grade(pose, gripper_angle=None, target_knob=knob,
                    jaw=cycle.PRECLOSE, graze_mm=cycle.GRAZE_MM,
                    wrist_roll=roll)
    if not g.get('reachable', False):
        return 0.0, False, 'out of reach'
    # Only a NEIGHBOURING KNOB counts as a collision here, filtered the same way
    # cycle.py filters it. Contact with the pedal itself is the model's known
    # artefact: the URDF's fingers are too long for a 14 mm knob standing on a
    # pedal, so there is no height where the jaws are on the knob and clear of
    # the deck, and the real gripper manages both. Treating that contact as a
    # failure scored EVERY pose zero, including a perfectly taught one, and the
    # whole envelope came out flat at nothing.
    hits = [h for h in g.get('collisions', [])
            if 'knob' in h and not (cycle.PEDAL_CONTACT_IS_MODEL_ARTEFACT
                                    and 'pedal' in h)]
    if hits:
        # Fouling a neighbour reads as force on the bench, and it is the
        # dangerous kind: plenty of squeeze, nothing held. Scoring it high
        # would walk the search into the neighbouring knob.
        return 0.0, False, f'fouls {hits[0]}'
    # How far the jaws actually ended up from the knob's axis.
    d = float(np.linalg.norm(np.asarray(g['achieved'])[:2] - target[:2])) * 1000
    score = max(0.0, 1.0 - d / FALLOFF_MM)
    return score, d <= HOLD_MM, f'{d:.1f} mm off the axis'


def trial(knob, true_err_mm, seed, max_tries=regrip.MAX_TRIES):
    """One mis-taught pose, searched. Returns (recovered, attempts, final mm)."""
    rng = np.random.default_rng(seed)
    s = regrip.Search(max_tries=max_tries)
    # The taught pose is wrong by true_err: an offset of -true_err is perfect.
    while True:
        off = s.next_offset()
        if off is None:
            break
        score, holding, note = attempt(knob, off + np.asarray(true_err_mm),
                                       rng)
        s.record(off, score, holding)
        if VERBOSE:
            print(f'      try ({off[0]:+5.1f},{off[1]:+5.1f}) -> '
                  f'score {score:.2f} {"HOLD" if holding else "    "}  {note}')
    landed = np.asarray(s.best) + np.asarray(true_err_mm)
    return s.holding, s.attempts, float(np.linalg.norm(landed))


def main():
    scene.reset()
    knob = 'knob1'
    print('recovering a mis-taught pose against the model: real IK, real '
          'collisions,\nreal gripper, 1.5 mm arm scatter. The search sees only '
          'force and hold.\n')
    print(f'{"taught pose off by":>19} {"one shot":>9} {"with search":>12} '
          f'{"attempts":>9} {"ends up":>9}')
    envelope, solo = {}, {}
    for err in (0, 2, 4, 6, 8, 10, 14):
        got = won = 0
        tries, finals = [], []
        for k in range(16):
            ang = 2 * np.pi * k / 16
            e = (err * np.cos(ang), err * np.sin(ang))
            ok, n, final = trial(knob, e, seed=k)
            won += ok
            if ok:
                tries.append(n)
                finals.append(final)
            got += trial(knob, e, seed=k, max_tries=1)[0]
        envelope[err], solo[err] = won / 16, got / 16
        med = f'{int(np.median(tries))}' if tries else '-'
        fin = f'{np.mean(finals):.1f} mm' if finals else '-'
        print(f'{err:16.0f} mm {got:6d}/16 {won:9d}/16 {med:>9} {fin:>9}')

    print()
    for err in (0, 2, 4):
        assert envelope[err] >= 0.95, \
            f'recovered only {envelope[err]:.0%} of {err} mm errors in the model'
    assert envelope[6] >= 0.80, \
        f'recovered only {envelope[6]:.0%} of 6 mm errors in the model'
    # The whole point: it has to beat giving up after one attempt, and by a lot
    # where the real errors live.
    assert envelope[4] - solo[4] >= 0.4, \
        (f'the search only bought {envelope[4]-solo[4]:.0%} at 4 mm, which does '
         f'not pay for a dozen squeezes')
    # And it must NOT claim to fix a pose that is simply wrong.
    assert envelope[14] <= 0.35, \
        (f'recovered {envelope[14]:.0%} of 14 mm errors, past the '
         f'{regrip.MAX_REACH:.0f} mm leash')
    print(f'the search is worth {envelope[4]-solo[4]:+.0%} at 4 mm and '
          f'{envelope[6]-solo[6]:+.0%} at 6 mm')
    print('regrip_sim self-checks passed')


if __name__ == '__main__':
    main()
