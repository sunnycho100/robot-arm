#!/usr/bin/env python3
"""First thing to run when the arm is back on the bench.

Ten minutes of checks, in the order that failures cascade, so a problem is
found where it happens rather than three stages later disguised as something
else. Everything is read-only or gently reversible; nothing here squeezes hard
or drives toward the pedal.

    python3 bench_check.py              # the whole sequence
    python3 bench_check.py --no-motion  # electrical and camera only
    python3 bench_check.py --skip 3     # resume at step 3

Each step prints PASS, WARN or FAIL and says what to do about it. Stop at the
first FAIL: everything after it is measuring a broken setup.
"""
import argparse
import os
import sys
import time

import numpy as np

SKIP_MOTION = False
RESULTS = []


def step(n, title):
    def deco(fn):
        fn._step = (n, title)
        return fn
    return deco


def report(state, msg, fix=''):
    RESULTS.append((state, msg))
    tag = {'PASS': 'PASS', 'WARN': 'WARN', 'FAIL': 'FAIL'}[state]
    print(f'  [{tag}] {msg}')
    if fix and state != 'PASS':
        print(f'         -> {fix}')
    return state != 'FAIL'


# ---------------------------------------------------------------- 1. the bus
@step(1, 'USB and servo bus')
def check_bus(A):
    try:
        counts = A.read()
    except Exception as e:
        return report('FAIL', f'cannot read servo positions ({e})',
                      'check the USB cable is a DATA cable, not power-only, and '
                      'that the arm has its own power switched on')
    if counts is None or len(counts) != len(A.IDS):
        return report('FAIL', f'got {counts!r} instead of {len(A.IDS)} positions',
                      'power-cycle the arm: pull Pi power, switch the arm off, '
                      'wait 5 s, back on')
    dead = [i for i, c in zip(A.IDS, counts) if c in (0, None)]
    if dead:
        return report('FAIL', f'servos {dead} report nothing',
                      'a servo has dropped off the bus; check its cable')
    report('PASS', f'all {len(counts)} servos answer: {list(counts)}')
    return True


# ------------------------------------------------------------- 2. the battery
@step(2, 'battery under load')
def check_battery(A):
    try:
        v = float(A.arm.getBatteryVoltage())
    except Exception as e:
        return report('WARN', f'could not read the battery ({e})',
                      'not fatal, but you lose the brownout warning')
    if v < 7.2:
        return report('FAIL', f'battery {v:.2f} V',
                      'below 7.2 V the controller browns out under grip load and '
                      'drops off USB mid-squeeze. Charge or swap before running')
    if v < 7.6:
        report('WARN', f'battery {v:.2f} V, getting low',
               'fine for now, expect trouble after a dozen grips')
        return True
    report('PASS', f'battery {v:.2f} V')
    return True


# -------------------------------------------------------- 3. does it move at all
@step(3, 'every joint moves, and the sign is right')
def check_joints(A):
    if SKIP_MOTION:
        return report('WARN', 'skipped (--no-motion)')
    ok = True
    base = A.read()
    for idx, name in enumerate(['base', 'shoulder', 'elbow', 'forearm',
                                'wrist pitch', 'wrist roll']):
        target = list(base)
        # Move AWAY from whichever limit is nearer. Always adding +60 clamps to
        # a 2-count nudge for a joint already parked near 880, which is inside
        # the servo deadband, and the joint then fails a test it never got: the
        # wrist pitch sits at 878 in the taught pose and read as "stalled".
        step_by = 60 if base[idx] < 500 else -60
        target[idx] = int(np.clip(base[idx] + step_by, 120, 880))
        if abs(target[idx] - base[idx]) < 20:
            report('WARN', f'{name}: no room to test (at {base[idx]}, '
                           f'limits 120-880)')
            continue
        A.move(target, speed=120)
        time.sleep(0.2)
        after = A.read()
        moved = after[idx] - base[idx]
        want = target[idx] - base[idx]
        # A joint that does not move at all passes any "equal and opposite"
        # test, so demand a plausible MAGNITUDE, not just consistency.
        if abs(moved) < abs(want) * 0.4:
            ok = report('FAIL', f'{name}: commanded {want:+d}, moved {moved:+d}',
                        'the servo is stalled, unpowered or mechanically jammed') and ok
        elif moved * want < 0:
            ok = report('FAIL', f'{name}: commanded {want:+d}, moved {moved:+d} '
                                f'(wrong direction)',
                        'the count-to-angle map for this joint has its sign '
                        'flipped; fix _MAP in arm.py before anything else') and ok
        else:
            report('PASS', f'{name}: commanded {want:+d}, moved {moved:+d}')
        A.move(base, speed=120)
        time.sleep(0.2)
    return ok


# -------------------------------------------------------------- 4. the gripper
@step(4, 'gripper opens, closes, and reports force')
def check_gripper(A):
    if SKIP_MOTION:
        return report('WARN', 'skipped (--no-motion)')
    A.release()
    time.sleep(0.4)
    opened = A.read()[A.GRIP]
    # squeeze on air: force should stay low, because there is nothing to hold
    cmd, lag, holding = A.squeeze(A.GRIP_FORCE)
    time.sleep(0.2)
    A.release()
    if holding:
        return report('FAIL', f'the gripper reports HOLDING on empty air '
                              f'(force {lag})',
                      'the force threshold is too low or a jaw is fouling; '
                      'nothing can be trusted downstream until this is right')
    report('PASS', f'open at {opened}, squeezed to {cmd} on air, force {lag}, '
                   f'correctly not holding')
    return True


# ---------------------------------------------------------------- 5. the camera
@step(5, 'camera, and whether it can see what it needs to')
def check_camera(A):
    try:
        import cv2
    except ImportError:
        return report('FAIL', 'no OpenCV on this machine', 'pip install opencv-contrib-python')
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    ok_read, frame = cap.read()
    cap.release()
    if not ok_read or frame is None:
        return report('FAIL', 'the camera returned no frame',
                      'check it is plugged in and not held by another process '
                      '(a live viewer will hold it)')
    h, w = frame.shape[:2]
    sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var()
    report('PASS', f'frame {w}x{h}, sharpness {sharp:.0f}')
    if sharp < 120:
        report('WARN', f'sharpness {sharp:.0f} is low',
               'focus the lens, or add light; the knob finder needs the cap edge')

    # ../cv in the repo, ./cv on the Pi, where the scripts are deployed flat
    # into the home directory rather than as a checkout.
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ('../cv', 'cv'):
        sys.path.insert(0, os.path.join(here, rel))
    try:
        import knobs2
        import compat
    except Exception as e:
        return report('WARN', f'perception modules did not import ({e})',
                      'the arm can still be driven from taught poses')

    ks = knobs2.find(frame)
    report('PASS' if len(ks) >= 2 else 'WARN',
           f'old single-frame finder sees {len(ks)} '
           f'{[round(k["conf"], 2) for k in ks]}')

    # The single-frame count above is what the demo used to trust, and on this
    # bench it reports five knobs on a three-knob pedal. Report the consensus
    # finder next to it: same frames, but only the detections that hold still.
    import knob as knob_new
    frames = knob_new.grab_frames(9)
    stable, flickering = (knob_new.find_knobs_stable(frames) if frames
                          else ({}, []))
    if len(stable) == 3:
        report('PASS', f'consensus over {len(frames)} frames: {len(stable)} knobs '
                       f'{sorted(stable)}')
    else:
        report('WARN', f'consensus over {len(frames)} frames: {len(stable)} knobs, '
                       f'expected 3 ({len(flickering)} flickering)',
               'usually lighting or framing: run aimlive.py and move the camera '
               'until the banner stays green')

    seen, which = set(), None
    for f in (frames or [frame]):
        _, ids, w = compat.detect_any(f)
        seen.update(int(i) for i in ids)
        which = w or which
    ids = sorted(seen)
    if len(ids):
        report('PASS', f'ArUco: {which} ids {ids}')
    else:
        report('WARN', 'no ArUco tag detected',
               'the old tag is torn; print fresh ones. Without tags the pedal '
               'pose has to be taught by hand and will not survive a bump')
    return True


# ------------------------------------------------- 6. repeatability, the real one
@step(6, 'how repeatable is this arm, actually')
def check_repeatability(A):
    if SKIP_MOTION:
        return report('WARN', 'skipped (--no-motion)')
    poses = A.poses()
    if 'grip0' not in poses:
        return report('WARN', 'no taught pose "grip0" to test against',
                      'teach one first; this measures the error that visual '
                      'servoing exists to remove')
    target = A.counts_of(poses['grip0'])
    seen = []
    for _ in range(5):
        A.move(A.NEUTRAL, speed=150)
        time.sleep(0.2)
        A.move(target, speed=120)
        time.sleep(0.4)
        seen.append(A.endpoint(A.read()))
    seen = np.array(seen)
    spread = (seen.max(axis=0) - seen.min(axis=0)) * 1000
    worst = float(spread.max())
    msg = (f'returning to the same taught pose 5 times spread '
           f'{spread[0]:.1f}/{spread[1]:.1f}/{spread[2]:.1f} mm in x/y/z')
    if worst > 4.0:
        report('WARN', msg + f' (worst {worst:.1f} mm)',
               'larger than the 4 mm that ejects the knob, which is exactly '
               'why the run must not trust a taught pose alone')
    else:
        report('PASS', msg)
    print(f'         this number is the case FOR visual servoing; it does not '
          f'have to be small')
    return True


STEPS = [check_bus, check_battery, check_joints, check_gripper,
         check_camera, check_repeatability]


def main():
    global SKIP_MOTION
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-motion', action='store_true')
    ap.add_argument('--skip', type=int, default=0, help='resume at this step')
    args = ap.parse_args()
    SKIP_MOTION = args.no_motion

    try:
        import arm as A
    except Exception as e:
        print(f'cannot import the arm driver: {e}')
        print('this script has to run on the Pi, with the arm connected')
        return 2

    print('bench check: stop at the first FAIL, everything after it is '
          'measuring a broken setup\n')
    for fn in STEPS:
        n, title = fn._step
        if n < args.skip:
            continue
        print(f'{n}. {title}')
        try:
            ok = fn(A)
        except Exception as e:
            ok = report('FAIL', f'the check itself blew up: '
                                f'{e.__class__.__name__}: {e}')
        print()
        if not ok:
            print(f'stopping at step {n}. Fix it, then: '
                  f'python3 bench_check.py --skip {n}')
            return 1

    fails = sum(1 for s, _ in RESULTS if s == 'FAIL')
    warns = sum(1 for s, _ in RESULTS if s == 'WARN')
    print(f'{len(RESULTS)} checks, {fails} failed, {warns} warnings')
    if not fails:
        print('the arm is fit to work with. Next: docs/RUNBOOK.md step 3')
    try:
        A.move(A.NEUTRAL, speed=150)
    except Exception:
        pass
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
