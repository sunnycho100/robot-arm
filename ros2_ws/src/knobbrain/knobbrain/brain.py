#!/usr/bin/env python3
"""The orchestrator: reads the knobs, decides, and drives his stack.

It subscribes to one topic, /aruco, and publishes three, /endpoint_desired,
/gripper_desired and /wrist_roll_desired. That is the entire interface. His
kinematics and servo nodes are reactive, so nothing moves until we publish.

His PresetController is subclassed rather than copied, so the tag chain and the
coordinate math stay in exactly one place: his. What we do not inherit is his
sequencer, which is cancelled in __init__ before rclpy ever spins it.

    ros2 run knobbrain brain      # needs the backend already running
    ros2 run knobbrain go         # starts the backend too
"""
import sys
import threading
import time

import numpy as np
import rclpy
from std_msgs.msg import Float32
from xarmrob_interfaces.msg import ME439PointXYZ
from xarmrob.preset_controller import PresetController

try:
    from knobbrain import cams, dial, eyes, screen
except ImportError:
    import cams
    import dial
    import eyes
    import screen

# Our display names against the keys in his offsets table. Only one differs.
OFFSET_KEY = {'level': 'dist lev'}

TAG_MOVED_M = 0.010     # 10 mm of tag drift invalidates every stored position
PLUNGE_Z = 0.6          # inches, straight from his macro


class Brain(PresetController):
    def __init__(self):
        super().__init__()
        # His constructor built an eight-knob queue and a 1 s timer. Kill both.
        # This is not a race: a rclpy timer cannot fire until rclpy.spin() runs,
        # and spin happens after __init__ has returned.
        self.timer.cancel()
        self.action_queue = []

        self.first_lock = None      # the two transforms as first seen
        self.tag_drift = 0.0
        self.zero = {}              # name -> pointer_rel at calibration
        self.now = {}               # name -> degrees turned this session
        self.target = {}
        self.last = {}
        self.arm = 'PARKED'

        saved = dial.load_session()
        self.sign = float(saved.get('sign', 1.0))

        # Which camera his aruco node grabbed is plug and probe order, not
        # something either of us chooses, so check it rather than assume it.
        warn = cams.swapped(cams.scan(), '/dev/video0', saved.get('tag_serial'))
        if warn:
            raise SystemExit(warn)
        cam = saved.get('camera')
        if not cam:
            cam, why = cams.knob_camera()
            if cam is None:
                raise SystemExit('knob camera: ' + why)
        self.eyes = eyes.Eyes(cam)
        self.get_logger().info(f'knob camera {cam}, tags on /dev/video0')

    # ---------------------------------------------------------------- tags
    def aruco_callback(self, msg):
        """Lock once for aiming, then keep comparing.

        His version returns early forever once locked, which is right for
        aiming: positions that jump mid-turn would be worse than useless. But
        the later readings are still worth having as an answer to "has the amp
        been nudged", so they are compared and never used to move anything.
        """
        was_locked = self.locked_T_cam_to_big is not None
        super().aruco_callback(msg)
        if not was_locked and self.locked_T_cam_to_big is not None:
            self.first_lock = self.locked_T_cam_to_big[0:3, 3].copy()
            return
        if self.first_lock is None:
            return
        for tag in msg.data.strip().split(' '):
            parts = tag.split(',')
            try:
                if int(parts[0].split(':')[1]) != self.AMP_ID:
                    continue
                here = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                self.tag_drift = float(np.linalg.norm(here - self.first_lock))
            except (IndexError, ValueError):
                pass

    def locked(self):
        return (self.locked_T_cam_to_little is not None
                and self.locked_T_cam_to_big is not None)

    # --------------------------------------------------------------- motion
    def _endpoint(self, name, z=None):
        key = OFFSET_KEY.get(name, name)
        c = self.calculate_robot_coords(key, z_override=z)
        m = ME439PointXYZ()
        m.xyz = [float(c[0]), float(c[1]), float(c[2])]
        self.pub_endpoint.publish(m)

    def _grip(self, v):
        m = Float32()
        m.data = float(v)
        self.pub_gripper.publish(m)

    def _wrist(self, v):
        m = Float32()
        m.data = float(v)
        self.pub_wrist.publish(m)

    def bite(self, name, deg):
        """One grip-twist-release. His macro, with only the twist size changed.

        The waits are his too. They exist because his stack has no motion-done
        feedback of any kind: read_bus_servos() is stubbed out in the provided
        driver, so there is nothing to wait ON. Sleeping is not laziness here,
        it is the only synchronisation available.
        """
        self.arm = 'MOVING'
        for step in (
                lambda: self._endpoint('initial'),      3.0,
                lambda: self._grip(dial.GRIP_OPEN),     0.5,
                lambda: self._wrist(dial.WRIST_HOME),   0.5,
                lambda: self._endpoint(name),           3.0,
                lambda: self._endpoint(name, PLUNGE_Z), 2.0,
                lambda: self._grip(dial.GRIP_CLOSED),   2.0,
                lambda: self._wrist(dial.wrist_target(deg)), 2.0,
                lambda: self._grip(dial.GRIP_OPEN),     2.0,
                lambda: self._wrist(dial.WRIST_HOME),   2.0,
                lambda: self._endpoint(name),           2.0,
                lambda: self._endpoint('initial'),      3.0):
            if callable(step):
                step()
            else:
                time.sleep(step)
        self.arm = 'PARKED'

    # --------------------------------------------------------------- reading
    def calibrate(self):
        seen = self.eyes.read()
        if not seen:
            return False
        self.zero = dict(seen)
        self.now = {n: 0.0 for n in seen}
        self.last = {}
        dial.save_session({'sign': self.sign, 'zero': self.zero,
                           'stamp': time.strftime('%H:%M')})
        return True

    def measure(self):
        seen = self.eyes.read()
        if not seen:
            return False
        for n, rel in seen.items():
            if n in self.zero:
                self.now[n] = dial.dial(rel, self.zero[n], self.sign)
        return True


def turn(b, name, target, say):
    """Converge on the target, and stop for anything that is not converging."""
    if b.tag_drift > TAG_MOVED_M:
        say(screen.box('AMP MOVED', [
            f'tag has shifted {b.tag_drift * 1000:.0f} mm since calibration.',
            'every stored position is wrong now.',
            're-seat the amp, then run  zero']))
        return
    b.target[name] = target
    taken = 0
    while True:
        step = dial.next_bite(b.now[name], target, taken)
        if step == 0.0:
            break
        say(f'    bite {taken + 1}   twist {step:+.0f} ...')
        before = b.now[name]
        b.bite(name, step)
        if not b.measure():
            say('    cannot see the row after that bite. stopping.')
            return
        got = b.now[name] - before
        b.last[name] = got
        taken += 1
        verdict = dial.check_bite(step, got)
        say(f'    read back {b.now[name]:.0f}   {got:+.0f}   '
            f'{target - b.now[name]:+.0f} to go')
        if verdict == 'backwards':
            say(screen.box('WRONG WAY', [
                f'commanded {step:+.0f}, the knob went {got:+.0f}.',
                'the wrist sign is inverted for this camera position.',
                'stopping. flip WRIST_SIGN in dial.py and re-run.']))
            return
        if verdict == 'slipping':
            say(screen.box('SLIPPING', [
                f'commanded {step:+.0f}, delivered {got:+.0f}.',
                'the knob is turning inside the jaws, not with them.',
                'stopping rather than hunting.']))
            return
    if taken >= dial.MAX_BITES and abs(target - b.now[name]) > dial.TOL_DEG:
        say(f'    gave up after {dial.MAX_BITES} bites at {b.now[name]:.0f}.')
    else:
        say(f'    {b.now[name]:.0f} of {target:.0f}, within tolerance.  DONE')


def tui(b):
    say = print
    while True:
        rows = [{'name': n, 'now': b.now.get(n), 'target': b.target.get(n),
                 'last': b.last.get(n)} for n in eyes.NAMES]
        print('\033[2J\033[H' + screen.render(rows, {
            'zero': dial.load_session().get('stamp', '--:--'),
            'amp': 'LOCKED' if b.locked() else 'WAITING',
            'camK': b.eyes.note, 'camT': f'drift {b.tag_drift * 1000:.0f}mm',
            'arm': b.arm}))
        if not b.zero:
            print('  no session zero yet. press  c  to calibrate.')
        try:
            cmd = screen.parse(input(' > '), len(eyes.NAMES))
        except (EOFError, KeyboardInterrupt):
            return
        if cmd[0] == 'quit':
            return
        if cmd[0] == 'error':
            if cmd[1]:
                say('  ' + cmd[1])
                input('  [enter]')
            continue
        if cmd[0] == 'zero':
            say('  reading the row ...')
            say('  zero set.' if b.calibrate() else
                '  cannot see all eight knobs. nothing stored.')
            input('  [enter]')
            continue
        if cmd[0] == 'state':
            b.measure()
            continue

        _, i, deg, multiple = cmd
        name = eyes.NAMES[i - 1]
        if not b.locked():
            say('  the amp is not located yet: both tags must be in view.')
            input('  [enter]')
            continue
        if not b.zero:
            say('  calibrate first, with  c')
            input('  [enter]')
            continue
        if not multiple:
            lo, hi = screen.nearest_multiples(deg)
            say(f'  {deg:.0f} is not a multiple of {dial.BITE_DEG:.0f}. '
                f'nearest: {lo:.0f} or {hi:.0f}.')
            if input('  [enter] use it anyway, or type one of those: ').strip():
                continue
        n = dial.bites_planned(b.now[name], deg)
        way = 'clockwise' if deg >= b.now[name] else 'counterclockwise'
        say(screen.box('CONFIRM', [
            f'{name}   {b.now[name]:.0f}  ->  {deg:.0f}      {way}, '
            f'{n} bite{"s" if n != 1 else ""} of {dial.BITE_DEG:.0f}', '',
            'each bite is his macro, unchanged:',
            '  hover -> plunge -> grip -> twist -> release -> untwist -> raise']))
        if input('    proceed?  [y/n] ').strip().lower() != 'y':
            continue
        turn(b, name, deg, say)
        input('  [enter]')


def main(args=None):
    rclpy.init(args=args)
    b = Brain()
    threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start()
    try:
        tui(b)
    finally:
        b.eyes.stop = True
        b.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
