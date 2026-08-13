#!/usr/bin/env python3
"""ME439 xArm pose tool.

  arm.py read              show current servo counts
  arm.py neutral           interpolate to the neutral pose
  arm.py teach NAME        cut torque, pose by hand, save on Enter
  arm.py save NAME         record the current pose under NAME
  arm.py goto NAME         interpolate to a saved pose
  arm.py list              list saved poses
  arm.py grip N            close the gripper by N counts (negative opens)
  arm.py turn DEG          roll the gripper about its own axis by DEG degrees
  arm.py nudge --dz=-4     shift the CURRENT pose by millimetres, add --save=NAME

Motion flags:  --linear      constant velocity instead of minimum jerk
               --speed N     counts per second of the fastest joint (default 120)
               --roll DEG    on goto, override the wrist roll in the target pose,
                             to wind up before a big turn
               --check       re-observe the ArUco tag and report how far the world
                             has drifted since the pose was taught, then move anyway

Vision is strictly optional. Every command works with no camera attached; --check
is the only thing that touches it. Blind replay is the baseline we compare against.
"""
import sys, os, json, time
import numpy as np
import xarm

IDS = [6, 5, 4, 7, 3, 2, 1]                    # base -> tip; last is the gripper
POSES = os.path.expanduser('~/poses.json')
YAML = '/home/pi/ros2_ws/src/xarmrob/config/robot_xarm_info.yaml'


def _neutral():
    """Read neutral from the ROS config, so that yaml is the single source of truth.

    Manual calibration with xarm_cmd_sliders edits that file, and the sliders
    initialise from it. Hard-coding the values here would silently drift from
    whatever was last calibrated.
    """
    try:
        for line in open(YAML):
            s = line.strip()
            if s.startswith('bus_servo_neutral_cmds_base_to_tip'):
                return [int(v) for v in s.split('[')[1].split(']')[0].split(',')]
    except Exception as e:
        print(f'could not read neutral from {YAML} ({e}), using fallback',
              file=sys.stderr)
    return [500, 510, 510, 512, 840, 515, 430]


NEUTRAL = _neutral()
DT = 0.04                                      # command period, s

FLOOR = os.path.expanduser('~/z_floor.json')

# Counts-to-angle tables and link offsets, straight out of robot_xarm_info.yaml.
# Same numbers the course kinematics node uses, lifted out so height can be
# checked without standing up ROS.
_MAP = {                       # joint: (angles deg, servo counts)
    '01': ([-90, 0, 90], [120, 500, 880]),
    '12': ([-180, -90, 0], [870, 500, 120]),
    '23': ([0, 90, 180], [140, 500, 880]),
    '34': ([-112, -90, 0, 90, 112], [1000, 890, 505, 140, 0]),
    '45': ([-112, -90, 0, 90, 112], [0, 120, 490, 880, 1000]),
    '56': ([-112, -90, 0, 90, 112], [0, 120, 500, 880, 1000]),
}
_JOINTS = ['01', '12', '23', '34', '45', '56']
_LINKS = [[0., 0., 0.074], [0.010, 0., 0.], [0.101, 0., 0.],
          [0.0627, 0., 0.0758], [0., 0., 0.], [0., 0., 0.]]
_END = [0.133, 0., -0.003]     # wrist to gripper centre, gripper open
_YSIGN = 1


def angles_of(counts):
    """Servo counts -> joint angles in radians, base to tip (6 joints, no gripper)."""
    out = []
    for j, c in zip(_JOINTS, counts[:6]):
        deg, cmd = _MAP[j]
        if cmd[0] > cmd[-1]:               # np.interp needs an increasing x
            deg, cmd = deg[::-1], cmd[::-1]
        out.append(np.deg2rad(float(np.interp(c, cmd, deg))))
    return out


def _rot(axis, t):
    c, s = np.cos(t), np.sin(t)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def endpoint(counts):
    """Gripper centre in metres, in the base frame. +z is up.

    This is the course's own forward kinematics. Our angle calibration has never
    been independently verified, so treat the absolute number with suspicion. The
    height check does not need it to be right, only CONSISTENT: the floor is
    measured with this same function, so both sides carry the same error and it
    cancels.
    """
    a = angles_of(counts)
    axes = ['z', 'y', 'y', 'x', 'y', 'x']
    signs = [1, _YSIGN, _YSIGN, 1, _YSIGN, 1]
    T = np.eye(4)
    for ax, sg, ang, r in zip(axes, signs, a, _LINKS):
        step = np.eye(4)
        step[:3, :3] = _rot(ax, ang * sg)
        step[:3, 3] = r
        T = T @ step
    return (T @ np.array(_END + [1.0]))[:3]


def z_floor():
    """Lowest gripper height allowed, or None if none has been taught."""
    try:
        return json.load(open(FLOOR))['z']
    except Exception:
        return None


def too_low(counts):
    """(True, z, floor) when this pose would put the gripper under the floor."""
    f = z_floor()
    if f is None:
        return False, None, None
    z = float(endpoint(counts)[2])
    return z < f, z, f

ROLL = 5            # index of servo ID 2, joint_56: the gripper's own roll axis
GRIP = 6            # index of servo ID 1, the gripper
# joint_56 angle-to-command table from robot_xarm_info.yaml, increasing in both
ROLL_DEG = [-112, -90, 0, 90, 112]
ROLL_CMD = [0, 120, 500, 880, 1000]

arm = xarm.Controller('USB')


def read(tries=4):
    """Servo counts base to tip. The bus reports unsigned 16-bit.

    Retries, because the HID link drops a report now and then when reads are
    interleaved with a command stream and the servos are drawing current. A lost
    report is a hiccup, not a fault, and it must not abort a motion in progress.
    """
    for attempt in range(tries):
        try:
            return [p - 65536 if p > 32767 else p
                    for p in (arm.getPosition(i) for i in IDS)]
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(0.05)


def minimum_jerk(tau):
    """Adamczyk's 5th-order profile. Zero velocity AND zero acceleration at both
    ends, which is what stops the arm snapping into motion and slamming to a halt."""
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def move(target, speed=120, linear=False, tol=80, check_every=0):
    """Interpolate to target, aborting on a stalled joint.

    A joint that cannot reach its commanded count is jammed against something.
    Holding position there stalls the servo, which browns out the USB bus.
    """
    assert len(target) == len(IDS)
    for v in target:
        assert 0 <= v <= 1000, f'command {v} outside 0-1000'

    # Height floor. Driving the gripper into the pedal shoves it, and once the
    # pedal has moved every taught pose is wrong and the camera calibration with
    # it. Cheaper to refuse the move than to notice afterwards.
    low, z, f = too_low(target)
    if low:
        print(f'REFUSED: that pose puts the gripper at z={z*1000:.0f} mm, below '
              f'the {f*1000:.0f} mm floor. Re-teach the floor if the bench '
              f'changed:  arm.py floor', file=sys.stderr)
        return False

    start = read()
    span = max(abs(t - s) for s, t in zip(start, target))
    if span == 0:
        print('already there')
        return True

    duration = max(0.5, span / speed)
    n = max(2, round(duration / DT))
    ease = (lambda tau: tau) if linear else minimum_jerk
    print(f'from {start}\n  to {target}'
          f'\n     {"constant velocity" if linear else "minimum jerk"}, '
          f'{span} counts in {duration:.1f}s over {n} steps')

    for k in range(1, n + 1):
        s = ease(k / n)
        # Clamp every STEP, not just the target. The arm can be sitting
        # outside the commandable range: measured on the bench, the base read
        # 1041 against a 0-1000 limit, having been pushed past its stop. The
        # target is checked above, but the interpolation starts from where the
        # arm actually IS, so the first steps are still out of range and the
        # xarm library refuses them ("position must be between 0 and 1000").
        # A move that is perfectly safe then dies on step one, and the arm is
        # stuck out of range with no way back through this function.
        step = [int(np.clip(round(a + (b - a) * s), 0, 1000))
                for a, b in zip(start, target)]
        # duration outruns the loop period on purpose: the servo is still moving
        # when the next command lands, so the steps blend instead of stair-step.
        # wait=False because wait is just a sleep, and we sleep below anyway.
        arm.setPosition(list(zip(IDS, step)), duration=int(DT * 1500), wait=False)
        time.sleep(DT)
        due = (k % check_every == 0) if check_every else (k == n // 2)
        if due:
            try:
                actual = read()
            except Exception as e:
                # Never abandon a loaded arm mid-motion over a comms hiccup.
                print(f'  (skipped stall check, link hiccup: {e})', file=sys.stderr)
                continue
            stalled = [(i, c, a) for i, c, a in zip(IDS, step, actual)
                       if abs(c - a) > tol]
            if stalled:
                print(f'STALL, stopping here. (servo, commanded, actual): {stalled}',
                      file=sys.stderr)
                return False

    arm.setPosition(list(zip(IDS, target)), duration=300, wait=True)
    print(f'  at {read()}')
    return True


# ---- poses. Entries are {'counts': [...], 'tag': {...} or None}. Bare lists
# ---- from before vision existed still load.

CAP = 605           # fully closed calibrates to 610; stay just under it. Was
                    # 590, and every real squeeze ended pinned there at ~73
                    # counts of force with the knob still slipping: the limit
                    # was choosing the grip force, not the caller.
HOLD_MIN = 30       # counts of force that constitute a real grip, not air
SLIP_DROP = 12      # force falling this far below its own peak means it escaped
GRIP_FORCE = 85     # the force the real run uses, so teach must test at it too


def squeeze(want_lag=35, step=10):
    """Close until the servo is visibly pushing. Returns (command, lag, holding).

    The gap between commanded and actual count IS the grip force: the servo is
    fighting the object by that much. Ramping and watching it beats naming a
    count, because a stalled servo is what browns out the USB bus.

    holding is False when the fingers ran all the way to the close limit without
    the force ever building, which means they shut on air. That is the exact
    failure that let an earlier run try to turn a knob it was not holding.

    A force that climbs and then COLLAPSES is a third outcome, and a different
    diagnosis: the knob was held and then squirted out of the jaws. Squeezing on
    past that point is what ejects it, so stop at the peak instead of the cap.
    """
    c, lag, peak, peak_c = read()[GRIP], 0, 0, None
    slipped = False
    for _ in range(40):
        c = min(CAP, c + step)
        arm.setPosition([(IDS[GRIP], c)], duration=150, wait=True)
        lag = c - read()[GRIP]
        print(f'  commanded {c}  actual {c-lag}  squeeze {lag}')
        if lag > peak:
            peak, peak_c = lag, c
        # A tapered knob wedges out under load. The force dropping well below its
        # own peak while the command is still rising can only mean the object left.
        if peak >= HOLD_MIN and lag < peak - SLIP_DROP:
            slipped = True
            break
        if lag >= want_lag or c >= CAP:
            break
    # What distinguishes a real grip from closing on air is the SIZE of the lag,
    # not whether the requested target was met. An object smaller than the finger
    # span caps out below any target you ask for, and that is still a firm grip.
    holding = lag >= HOLD_MIN and not slipped
    if slipped:
        print(f'SLIPPED: force peaked at {peak} counts (command {peak_c}) then fell '
              f'to {lag}. The fingers had it and it wedged out. This is placement, '
              f'not force: grip lower on the knob where the barrel is fattest.',
              file=sys.stderr)
    elif not holding:
        print(f'NOT HOLDING: reached the close limit at {c} with only {lag} counts '
              f'of force. The fingers shut on nothing.', file=sys.stderr)
    elif lag < want_lag:
        print(f'holding at {c}, squeeze {lag} counts. That is all this object '
              f'allows, you asked for {want_lag}: the close limit binds first.')
    else:
        print(f'holding at {c}, squeeze {lag} counts')
    return c, lag, holding


def turn_by(deg, speed=200, linear=False):
    """Roll the gripper about its own axis. Returns False if out of joint range."""
    target = read()
    was = float(np.interp(target[ROLL], ROLL_CMD, ROLL_DEG))
    want = was + deg
    if not ROLL_DEG[0] <= want <= ROLL_DEG[-1]:
        print(f'roll would reach {want:.0f} deg, outside the joint range '
              f'{ROLL_DEG[0]} to {ROLL_DEG[-1]}. Re-grip at a different angle '
              f'and turn in two bites.', file=sys.stderr)
        return False
    target[ROLL] = round(float(np.interp(want, ROLL_DEG, ROLL_CMD)))
    print(f'roll {was:+.0f} -> {want:+.0f} deg (servo 2 -> {target[ROLL]})')
    return move(target, speed=speed, linear=linear)


# How narrow to make the jaws before lowering them between two knobs. This is
# a COUNT, and it has to be found on the bench: simulation could only settle
# the direction, not the number.
#
# What simulation established: with all four finger joints driven, the tips
# splay from 28 mm apart at the closed end to 88 mm at the open end, while the
# DS-1 knobs are 24 mm apart. Approaching with the jaws open therefore drives
# the fingers 5 to 6 mm into BOTH neighbours. Descending nearly shut is the
# only way through.
#
# What simulation could NOT establish: the actual count. Its fingers are too
# long for a 14 mm knob standing on a pedal (there is no height where the jaws
# are on the knob and clear of the pedal, which the real gripper manages), so
# its millimetre clearances say nothing about this hardware.
#
# To calibrate: put the arm over the CENTRE knob at grip height, run
#   python3 arm.py preclose <count>
# from wide to narrow, and take the first count where the fingers pass between
# the neighbours without touching them. Write it here.
PRECLOSE = 470      # UNVERIFIED starting guess, between release (410) and CAP


def preclose(to=PRECLOSE, speed=200):
    """Narrow the jaws before descending between knobs, without gripping.

    Not a grip: it moves to a fixed opening and returns, so nothing is being
    held and no force is measured. squeeze() is what grips, afterwards.
    """
    t = read()
    t[GRIP] = int(np.clip(to, 90, CAP))
    print(f'pre-close jaws to {t[GRIP]} (was {read()[GRIP]})')
    return move(t, speed=speed)


def release(to=410, speed=200):
    """Open the gripper back to its resting width."""
    t = read()
    t[GRIP] = to
    return move(t, speed=speed)


def approach(target, via=0.75, speed=120, creep=50, tol=35):
    """Two-stage approach: normal speed most of the way, then creep with contact
    detection on every step.

    Driving straight onto the knob has pushed the pedal across the desk. A single
    move is a straight line in JOINT space, so the gripper arrives diagonally, and
    any bearing error puts it into the deck rather than onto the knob, with the
    full travel of the move behind it. Creeping the last leg with a tight stall
    tolerance turns a shove into a detected contact: the arm stops instead of
    winning the argument with the pedal.
    """
    start = read()
    via_pt = [round(a + (b - a) * via) for a, b in zip(start, target)]
    print(f'-- approach stage 1: {via*100:.0f}% of the way')
    if not move(via_pt, speed=speed):
        return False
    print(f'-- approach stage 2: creeping the last {(1-via)*100:.0f}%, '
          f'contact detection every step (tol {tol})')
    if not move(target, speed=creep, tol=tol, check_every=2):
        return False

    # Say so when the pose was not actually reached, because everything
    # downstream misreads it as a sideways aiming error.
    #
    # A pose taught with the arm limp records UNLOADED counts. Replayed under
    # power the servo has to hold the arm's weight and settles short, and it
    # stays short: measured here, the elbow sat 11 counts under on six
    # consecutive commands to the same target, putting the fingertips 6.8 mm
    # above a knob 17 mm across. Biasing the command by the residual is the
    # obvious fix and it does not work on this arm: elbow+11 computes to
    # 68.9 mm, under the 70.4 mm floor, so move() refuses it. The pose as
    # taught is not reachable under load, and no amount of searching finds
    # that out, because find_grip only searches x and y. Teach under power
    # instead, so the saved counts are ones the arm can hold.
    err = [t - c for t, c in zip(target, read())][:GRIP]
    if max(abs(e) for e in err) > 8:
        z = float(endpoint(read())[2]) * 1000
        want = float(endpoint(target)[2]) * 1000
        print(f'   NOT AT THE POSE: off by {err}, {z - want:+.1f} mm in z. '
              f'The jaws will close where the knob is not. Re-teach this pose '
              f'with the arm powered rather than limp.', file=sys.stderr)
    return True


def poses():
    return json.load(open(POSES)) if os.path.exists(POSES) else {}


def counts_of(entry):
    return entry['counts'] if isinstance(entry, dict) else entry


def save(name, tag=None):
    p = poses()
    p[name] = {'counts': read(), 'tag': tag}
    json.dump(p, open(POSES, 'w'), indent=1)
    print(f"saved {name} = {p[name]['counts']}"
          + (f"  with tag {list(tag)}" if tag else '  (no tag seen)'))


def look_safe(shot=None):
    """Tag poses, or {} if the camera or OpenCV is missing. Never fatal."""
    try:
        sys.path.insert(0, os.path.expanduser('~'))
        import see
        return see.look(frames=20, shot=shot)
    except Exception as e:
        print(f'vision unavailable ({e.__class__.__name__}: {e})', file=sys.stderr)
        return {}


def drift(taught, now):
    """How far each tag moved between teach time and now, in mm."""
    for i, was in sorted(taught.items()):
        i = int(i)
        if i not in now:
            print(f'  tag {i}: TAUGHT but NOT VISIBLE now')
            continue
        d = (np.array(now[i]['xyz']) - np.array(was['xyz'])) * 1000
        print(f'  tag {i}: moved dx={d[0]:+6.1f} dy={d[1]:+6.1f} dz={d[2]:+6.1f} mm'
              f'   |d|={np.linalg.norm(d):5.1f} mm'
              f"   (seen {now[i]['seen']}/{now[i]['of']})")


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    linear, check = '--linear' in flags, '--check' in flags
    speed = next((int(f.split('=')[1]) for f in flags if f.startswith('--speed=')), 120)
    roll = next((float(f.split('=')[1]) for f in flags if f.startswith('--roll=')), None)

    cmd = argv[0] if argv else 'read'
    name = argv[1] if len(argv) > 1 else None

    if cmd == 'read':
        print(dict(zip(IDS, read())))

    elif cmd == 'neutral':
        move(NEUTRAL, speed=speed, linear=linear)

    elif cmd == 'goto':
        entry = poses()[name]
        target = list(counts_of(entry))
        if roll is not None:
            # Wind the wrist up before gripping, so a big turn has range to spend.
            # Rolling about the gripper's own axis shifts the fingertip ~3 mm, so
            # the grip point is effectively unchanged.
            target[ROLL] = round(float(np.interp(roll, ROLL_DEG, ROLL_CMD)))
            print(f'roll overridden to {roll:+.0f} deg (servo 2 -> {target[ROLL]})')
        if check:
            taught = entry.get('tag') if isinstance(entry, dict) else None
            if not taught:
                print('no tag was recorded when this pose was taught, '
                      'so there is nothing to compare against')
            else:
                print('re-observing the world before replaying:')
                drift(taught, look_safe())
        approach(target, speed=speed)

    elif cmd == 'grip':
        target = read()
        target[GRIP] += int(name)
        print(f'gripper {read()[GRIP]} -> {target[GRIP]}')
        move(target, speed=speed, linear=linear)

    elif cmd == 'squeeze':
        squeeze(int(name) if name else 35)

    elif cmd == 'preclose':
        # Calibration aid: run this over the CENTRE knob at grip height,
        # sweeping from wide to narrow, and keep the first count whose fingers
        # pass between the neighbouring knobs without touching them.
        preclose(int(name) if name else PRECLOSE)

    elif cmd == 'floor':
        # Teach the lowest allowed height by putting the arm there, rather than
        # naming a number. The FK's absolute accuracy is unverified, so a number
        # typed in millimetres would mean nothing; a pose measured with the same
        # function the check uses means exactly the right thing.
        z = float(endpoint(read())[2])
        margin = float(name) / 1000 if name else 0.003
        json.dump({'z': z + margin, 'taught_at': z, 'margin_mm': margin * 1000},
                  open(FLOOR, 'w'), indent=1)
        print(f'floor set to {(z+margin)*1000:.1f} mm '
              f'(here {z*1000:.1f} mm, plus {margin*1000:.0f} mm margin)')
        print('moves that would put the gripper below this are now refused.')

    elif cmd == 'height':
        z = float(endpoint(read())[2])
        f = z_floor()
        print(f'gripper at z = {z*1000:.1f} mm' +
              (f'   floor {f*1000:.1f} mm   {(z-f)*1000:+.1f} mm of clearance'
               if f is not None else '   (no floor taught yet)'))

    elif cmd == 'nudge':
        # The bench loop this exists for: goto a taught pose, look at where the
        # fingers actually sit against the knob, correct by millimetres, look
        # again, save. Before this the only remedy for "a few mm low" was
        # re-teaching the whole pose by hand and hoping it did not sag.
        #
        # Deliberately slow and with contact detection on every step: a nudge is
        # usually aimed at the pedal, and the point is to stop on a touch rather
        # than shove. The height floor still applies, so a nudge cannot dig in.
        import ik
        d = [next((float(f.split('=')[1]) for f in flags
                   if f.startswith(f'--d{ax}=')), 0.0) for ax in 'xyz']
        here = read()
        try:
            target, got = ik.nudged(here, *d)
        except ValueError as e:
            sys.exit(f'nudge refused: {e}')
        print(f'asked ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) mm, '
              f'this pose moves ({got[0]:+.1f}, {got[1]:+.1f}, {got[2]:+.1f})')
        print(f'  counts change by {[t - h for t, h in zip(target, here)]}')
        if not move(target, speed=50, tol=35, check_every=2):
            sys.exit('the nudge did not complete, so nothing was saved')
        z, fl = float(endpoint(read())[2]), z_floor()
        print(f'  now at z = {z*1000:.1f} mm' +
              (f', {(z-fl)*1000:+.1f} mm above the floor' if fl is not None
               else ' (no floor taught yet)'))
        into = next((f.split('=')[1] for f in flags
                     if f.startswith('--save=')), None)
        if into:
            save(into, look_safe())

    elif cmd == 'turn':
        turn_by(float(name), speed=speed, linear=linear)

    elif cmd == 'teach':
        # Relax and save from one process. The board drops the unload command if
        # the process exits on top of it, so resend it while waiting, and read
        # the pose here rather than handing the USB device to a second process.
        import threading
        stop = threading.Event()

        def keep_off():
            while not stop.wait(0.4):
                arm.servoOff(IDS)

        t = threading.Thread(target=keep_off, daemon=True)
        arm.servoOff(IDS)
        t.start()
        input('torque off. Pose the arm by hand, then press Enter to save. ')
        stop.set()
        t.join()
        save(name, look_safe())

        # Verify the fingers actually straddle something, right now, while the
        # user's hands are still on it. Saving a pose that closes on air is what
        # wasted a whole session: the miss only showed up on replay.
        # The pose is already saved with the open gripper count, so squeezing
        # here is only a test and does not change what was stored.
        taught_grip = read()[GRIP]
        print('\nchecking the grip has hold of something...')
        # Test at the force the real run applies. Testing softer passes placements
        # that hold at 35 counts and then wedge out at 70, which is exactly the
        # false GOOD that sent a good-looking pose into a failed run.
        _, lag, holding = squeeze(GRIP_FORCE)
        if holding:
            print('GOOD, the fingers are on the object.')
        else:
            print('WARNING: this pose does not survive the run force (see above).\n'
                  '  Re-pose and teach again, or the replay will fail the same way.',
                  file=sys.stderr)
        t2 = read()
        t2[GRIP] = taught_grip
        move(t2, speed=200)

    elif cmd == 'settle':
        # Teach the pose the arm can actually HOLD, not the one it was posed
        # at. `teach` records counts while the arm is limp in your hand, and
        # under power the elbow settles ~11 counts short of them, which put
        # the fingertips 6.8 mm above the knob and closed the jaws on air. The
        # command that would compensate is below the height floor, so the
        # taught pose is simply unreachable under load. Fly to it, read where
        # the arm truly came to rest, and save THAT.
        entry = poses()[name]
        target = list(counts_of(entry))
        preclose()
        approach(target, speed=speed)
        time.sleep(0.5)
        landed = read()
        landed[GRIP] = target[GRIP]
        off = [l - t for l, t in zip(landed, target)][:GRIP]
        print(f'taught {target}')
        print(f'landed {landed}   off by {off}')
        print(f'z: taught {float(endpoint(target)[2])*1000:.1f} mm, '
              f'actually {float(endpoint(landed)[2])*1000:.1f} mm')
        _, lag, holding = squeeze(GRIP_FORCE)
        release()
        if not holding:
            sys.exit(f'the jaws closed on air here (force {lag}), so this is '
                     f'not a grip worth saving. Re-teach with arm.py teach, '
                     f'and hold the fingers LOWER on the knob than feels right.')
        save(name, look_safe())
        print(f'{name} re-saved as the pose the arm actually reaches, '
              f'holding at {lag} counts')

    elif cmd == 'save':
        save(name, look_safe())

    elif cmd == 'list':
        for k, v in poses().items():
            tag = (v.get('tag') if isinstance(v, dict) else None) or {}
            print(f'{k:12s} {counts_of(v)}   tags: {sorted(tag) or "none"}')

    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
