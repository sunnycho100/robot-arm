#!/usr/bin/env python3
"""Feel for the top of the knob and the panel beside it, and grip halfway.

    probez.py POSE                 measure only, move nothing permanently
    probez.py POSE --save          re-save POSE at the measured mid-height

Height is the axis that broke every grip this week, and it is the one axis the
overhead camera cannot see: a top-down view has no z. Guessing it by eye put
the fingers on top of the cap, where a tapered knob squeezes the jaws upward
and off. So do not guess it and do not triangulate it either.

Triangulating with the ArUco tag was the obvious alternative and it is worse
here. The tag gives the panel plane in CAMERA coordinates, and turning that
into a servo command needs a camera-to-arm transform nobody has verified, on
top of a forward-kinematics model whose ABSOLUTE accuracy this project has
never checked (arm.py says so itself). A contact probe skips all of it: the
answer comes back in the same servo counts the grip command is written in, so
no frame conversion can be wrong.

    top of the knob     descend on the knob until something stops the arm
    the panel           step aside by one cap width, descend again
    grip height         halfway between, which is the middle of the skirt

The measurement is only as good as the stall detection, so it is taken twice
and disagreement is reported rather than averaged away.
"""
import sys
import time

import numpy as np

import arm as A
import ik
import knob

FLAGS = [a for a in sys.argv[1:] if a.startswith('--')]
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
SAVE = '--save' in FLAGS

STEP_MM = 1.0            # how far each probe step descends
MAX_STEPS = 25           # 25 mm of travel is more than any knob is tall
CLEAR_MM = 18.0          # start this far above the taught pose
TOL = 25                 # counts of lag that mean the arm was stopped


def _z():
    return float(A.endpoint(A.read())[2]) * 1000.0


def descend(label):
    """Step down until something stops the arm. -> z in mm where it stopped."""
    floor = A.z_floor()
    for i in range(MAX_STEPS):
        here = A.read()
        try:
            target, _ = ik.nudged(here, 0.0, 0.0, -STEP_MM)
        except ValueError as e:
            print(f'   {label}: cannot step lower ({e})')
            return _z()
        if floor is not None and float(A.endpoint(target)[2]) < floor:
            print(f'   {label}: reached the height floor without touching '
                  f'anything. Either the floor is set too high or this pose '
                  f'is not over the knob.')
            return None
        if not A.move(target, speed=25, tol=TOL, check_every=1):
            z = _z()
            print(f'   {label}: contact after {i} steps, z = {z:.1f} mm')
            return z
    print(f'   {label}: {MAX_STEPS} steps with no contact. Nothing under the '
          f'fingers, so this pose is not over what you think it is.')
    return None


def lift(mm):
    target, _ = ik.nudged(A.read(), 0.0, 0.0, float(mm), max_mm=abs(mm) + 5)
    A.move(target, speed=60)


def main():
    if not ARGS:
        raise SystemExit(__doc__)
    name = ARGS[0]
    poses = A.poses()
    if name not in poses:
        raise SystemExit(f'no taught pose "{name}"')
    taught = list(A.counts_of(poses[name]))

    A.release()
    A.approach(taught, speed=120)
    time.sleep(0.3)
    base_z = _z()
    print(f'{name}: taught at z = {base_z:.1f} mm')

    tops = []
    for attempt in (1, 2):
        lift(CLEAR_MM)
        print(f'probe {attempt}: knob top')
        z = descend('knob')
        if z is None:
            raise SystemExit('probe failed on the knob, nothing measured')
        tops.append(z)
    if abs(tops[0] - tops[1]) > 1.5:
        print(f'\nthe two knob probes disagree by {abs(tops[0]-tops[1]):.1f} mm '
              f'({tops[0]:.1f} and {tops[1]:.1f}). The stall threshold is not '
              f'reading contact cleanly; do not trust the number below.')
    top = float(np.mean(tops))

    # Step aside by a cap width so the fingers are beside the knob, not on it,
    # then find the panel. Sideways, not along the row, so the neighbouring
    # knob is not what gets touched.
    lift(CLEAR_MM)
    aside = knob.CAP_MM * 1.2
    print(f'stepping {aside:.0f} mm aside to find the panel')
    try:
        target, _ = ik.nudged(A.read(), 0.0, aside, 0.0, max_mm=aside + 5)
        A.move(target, speed=60)
    except ValueError as e:
        raise SystemExit(f'cannot step aside: {e}')
    panel = descend('panel')

    print()
    if panel is None:
        print(f'knob top {top:.1f} mm. No panel reading, so no height to '
              f'halve: grip height stays a judgement call.')
        return
    height = panel - top
    mid = 0.5 * (top + panel)
    print(f'knob top   {top:6.1f} mm')
    print(f'panel      {panel:6.1f} mm')
    print(f'knob is    {abs(height):6.1f} mm tall, grip at {mid:6.1f} mm '
          f'({mid - base_z:+.1f} mm from where it was taught)')

    if not SAVE:
        print('\nmeasure only. Add --save to move there, test the grip and '
              'keep it if it holds.')
        return

    lift(CLEAR_MM)
    A.approach(taught, speed=120)
    try:
        target, _ = ik.nudged(A.read(), 0.0, 0.0, mid - base_z,
                              max_mm=abs(mid - base_z) + 5)
    except ValueError as e:
        raise SystemExit(f'cannot reach the mid height: {e}')
    A.move(target, speed=40, tol=A.TOL if hasattr(A, 'TOL') else 35)
    _, force, holding = A.squeeze(A.GRIP_FORCE)
    A.release()
    if not holding:
        raise SystemExit(f'the jaws closed on air at the measured mid height '
                         f'(force {force}). Not saved. The probe found '
                         f'something, but it was not this knob.')
    A.save(name, A.look_safe())
    print(f'{name} re-saved at the measured grip height, holding at {force} counts')


if __name__ == '__main__':
    main()
