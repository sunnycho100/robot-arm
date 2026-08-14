#!/usr/bin/env python3
"""Knob angles, bite planning, and the session zero. No ROS, no camera.

Everything here is arithmetic on numbers the detector already produced, which
is why it runs and tests on a laptop with nothing plugged in.

The one idea worth stating: an angle is only meaningful against a reference,
and the reference we use is the ROW, not the image. ampknobs already reports
`pointer_rel`, the pointer angle minus the direction the knob row runs in. Tilt
the camera, roll it, move it to the other side of the bench, and `pointer_rel`
does not change. Subtract the angle recorded at calibration and what is left is
how far that knob has turned this session.

    python3 dial.py        # self-test
"""
import json
import math
import os

BITE_DEG = 25.0      # one bite is his macro: grip, twist, release. Unchanged.
TOL_DEG = 8.0        # close enough. Below this we stop rather than fuss.
MAX_BITES = 6        # converge, but never grind. See next_bite().
FULL_TRAVEL = 300.0  # ASSUMED pot travel. Measure stop to stop and fix this.

# His macro's numbers, from preset_controller.build_macro_sequence().
WRIST_HOME = 1.57    # rad. Wrist square to the panel, the pose he grips from.
GRIP_OPEN = 0.0
GRIP_CLOSED = 1.57

# Whether a positive wrist command turns the knob clockwise as the knob camera
# sees it. UNVERIFIED: it depends on which side the camera sits, and no run has
# checked it. The first bite of a session proves it, and check_bite() below
# reports 'backwards' rather than letting the loop chase its own tail.
WRIST_SIGN = 1.0

SESSION = os.path.expanduser('~/.knobbrain.json')


def dial(pointer_rel, zero_rel, sign=1.0):
    """How far this knob has turned since calibration, in degrees.

    Positive is clockwise. The result is absolute, not modular: a pot has hard
    stops, so 350 and -10 are different claims about the world and only one of
    them is reachable. Anything landing in the 30 degrees of dead zone beyond
    full travel is reported as a small negative, because a knob nudged slightly
    below its zero is common and a knob turned 350 degrees is impossible.
    """
    d = ((pointer_rel - zero_rel) * sign) % 360.0
    if d > (FULL_TRAVEL + 360.0) / 2.0:
        d -= 360.0
    return d


def in_range(deg):
    return -TOL_DEG <= deg <= FULL_TRAVEL + TOL_DEG


def next_bite(now, target, taken=0):
    """The next twist to command, signed degrees. 0 means stop.

    Converging, not counting: the size comes from what the camera measured, not
    from a plan made before anything moved. That matters because commanded and
    delivered are different numbers on this rig, measured at one point as +90
    commanded and +19 delivered. A fixed count of bites would simply stop short
    and call it done.

    The cap is what keeps converging from becoming grinding.
    """
    if taken >= MAX_BITES:
        return 0.0
    err = target - now
    if abs(err) <= TOL_DEG:
        return 0.0
    return math.copysign(min(abs(err), BITE_DEG), err)


def bites_planned(now, target):
    """How many bites this would take if every one landed perfectly.

    Shown before the run so you know roughly what you are agreeing to. The loop
    does not use it.
    """
    err = abs(target - now)
    return 0 if err <= TOL_DEG else int(math.ceil(err / BITE_DEG))


def wrist_target(bite_deg):
    """The absolute /wrist_roll_desired value for a bite of this size."""
    return WRIST_HOME + math.radians(bite_deg) * WRIST_SIGN


def check_bite(commanded, delivered):
    """What the camera says about the bite that just ran.

    'backwards' is the one worth having. If the wrist sign is wrong, every bite
    drives the knob away from the target and a converging loop would happily
    command bite after bite in the direction that is making things worse.
    """
    if abs(delivered) < TOL_DEG / 2.0:
        return 'slipping'
    if commanded * delivered < 0:
        return 'backwards'
    return 'ok'


def bar(now, target, cells=12):
    """The row's progress bar. One cell is one bite, which is the whole point:
    `[##-.........]` reads as two done and one to go without any arithmetic."""
    per = FULL_TRAVEL / cells
    done = max(0, min(cells, int(round(now / per))))
    want = done if target is None else max(0, min(cells, int(round(target / per))))
    owed = max(0, want - done)
    return '[' + '#' * done + '-' * owed + '.' * max(0, cells - done - owed) + ']'


def load_session(path=SESSION):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_session(obj, path=SESSION):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def _selftest():
    # dial: plain subtraction, and the wrap that a pot makes unambiguous
    assert abs(dial(100.0, 100.0)) < 1e-9
    assert abs(dial(150.0, 100.0) - 50.0) < 1e-9
    assert abs(dial(95.0, 100.0) + 5.0) < 1e-9, 'just below zero stays negative'
    assert abs(dial(40.0, 100.0) - 300.0) < 1e-9, 'full travel reads as 300'
    assert abs(dial(100.0, 150.0, sign=-1.0) - 50.0) < 1e-9, 'mirrored camera'
    assert in_range(0.0) and in_range(300.0)
    assert not in_range(320.0)

    # next_bite: full bites while far, the remainder when close, stop inside tol
    assert next_bite(0, 75) == 25.0
    assert next_bite(50, 75) == 25.0
    assert abs(next_bite(60, 75) - 15.0) < 1e-9, 'partial correction bite'
    assert next_bite(72, 75) == 0.0, 'within tolerance, do not fuss'
    assert next_bite(75, 25) == -25.0, 'counterclockwise is just a sign'
    assert next_bite(0, 300, taken=MAX_BITES) == 0.0, 'the cap stops grinding'

    # a slipping grip still converges, and never overshoots past tolerance
    now, taken = 0.0, 0
    while (b := next_bite(now, 75, taken)):
        now += b * 0.55          # deliver barely half of every bite
        taken += 1
    assert taken <= MAX_BITES
    assert now <= 75 + TOL_DEG, f'overshot to {now}'

    # and a healthy grip arrives in the number of bites we promised
    now, taken = 0.0, 0
    while (b := next_bite(now, 75, taken)):
        now += b
        taken += 1
    assert taken == bites_planned(0, 75) == 3, taken
    assert abs(now - 75.0) < 1e-9

    assert bites_planned(0, 0) == 0
    assert bites_planned(0, 100) == 4

    # the wrist command his macro would have sent
    assert abs(wrist_target(25.0) - 2.006) < 0.01, 'his 1.57 -> 2.00 twist'
    assert wrist_target(-25.0) < WRIST_HOME, 'counterclockwise goes below home'

    assert check_bite(25, 23) == 'ok'
    assert check_bite(25, -22) == 'backwards'
    assert check_bite(25, 1) == 'slipping'

    assert bar(0, None) == '[............]'
    assert bar(50, 75) == '[##-.........]', bar(50, 75)
    assert bar(300, 300) == '[############]'
    assert bar(100, 25) == '[####........]', 'overshoot owes nothing'

    print('dial: 26 assertions pass')


if __name__ == '__main__':
    _selftest()
