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
import os
import sys
import threading
import time

import numpy as np
import rclpy
from std_msgs.msg import Float32
from xarmrob_interfaces.msg import ME439PointXYZ
from xarmrob.preset_controller import PresetController

try:
    from knobbrain import cams, dial, eyes, macro, screen
except ImportError:
    import cams
    import dial
    import eyes
    import macro
    import screen

TAG_MOVED_M = 0.010     # 10 mm of tag drift invalidates every stored position


class Brain(PresetController):
    def __init__(self):
        super().__init__()
        # His constructor built an eight-knob queue and a 1 s timer. Kill both.
        # This is not a race: a rclpy timer cannot fire until rclpy.spin() runs,
        # and spin happens after __init__ has returned.
        self.timer.cancel()

        # Read his macro before discarding it. Everything we would otherwise
        # have copied out of his file lives in here: the knob names and their
        # order, the wrist angle each is approached at (he flips four of them),
        # how far his twist goes, the plunge depth, the gripper values, and his
        # waits. Copying any of it would go stale the next time he retunes.
        self.park, self.blocks = macro.blocks(self.action_queue)
        self.knobs = list(self.blocks)
        dial.adopt(macro.bite_deg(next(iter(self.blocks.values()))))
        self.action_queue = []

        self.first_lock = None      # the two transforms as first seen
        self.tag_drift = 0.0
        self.zero = {}              # name -> pointer_rel at calibration
        self.now = {}               # name -> degrees turned this session
        self.at = {}                # name -> (cx, cy, major) at calibration
        self.target = {}
        self.last = {}
        self.arm = 'PARKED'
        # Assume every bite delivers what it was asked for, instead of checking.
        # Off by default and it should stay off: the measured slip on this rig
        # was +90 commanded against +19 delivered, with the gripper reporting a
        # clean hold throughout, so the assumed numbers are not a slightly worse
        # estimate, they are a different story from what the knob did. It exists
        # because a demo that runs beats a demo waiting on camera framing.
        self.open_loop = False

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
        self.eyes = eyes.Eyes(cam, self.knobs)
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
        c = self.calculate_robot_coords(name, z_override=z)
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

    def _do(self, action, value=None):
        t = action['type']
        if t == 'hover':
            self._endpoint(action['target'])
        elif t == 'plunge':
            self._endpoint(action['target'], action['z'])
        elif t == 'gripper':
            self._grip(action['value'])
        elif t == 'twist':
            self._wrist(action['value'] if value is None else value)
        else:
            raise ValueError(f'his macro has a step we do not know: {t}')

    def bite(self, name, deg):
        """One grip-twist-release: HIS block for this knob, replayed.

        Not a re-implementation of his sequence but a replay of it, straight
        off build_macro_sequence(). The only substitution is the one number we
        are actually deciding, the size of the twist. Everything else, his
        order, his plunge depth, his gripper values and the wrist angle he
        approaches this particular knob at, comes across untouched, so him
        retuning the demo retunes ours.

        The waits are his too, and they are not laziness: his stack has no
        motion-done feedback of any kind, since read_bus_servos() is stubbed
        out in the provided driver. There is nothing to wait ON.
        """
        block = self.blocks[name]
        i = macro.twist_at(block)
        home = macro.home_of(block, i)
        self.arm = 'MOVING'
        for j, action in enumerate(block):
            self._do(action, dial.wrist_target(deg, home) if j == i else None)
            time.sleep(action['wait'])
        # Back to his safe pose, which is where the next block would have
        # started anyway. A bite that ends hovering over the knob leaves the
        # arm in front of the camera we are about to read with.
        self._endpoint(self.park)
        time.sleep(block[0]['wait'])
        self.arm = 'PARKED'

    # --------------------------------------------------------------- reading
    def calibrate(self):
        seen = self.eyes.read()
        if not seen:
            return False
        self.zero = {n: k['rel'] for n, k in seen.items()}
        # Where each knob sits in the image. This is what lets a later read
        # name a PARTIAL row: match by position instead of by counting along
        # it. Only valid while the camera holds still, which is also why moving
        # the camera means calibrating again.
        self.at = {n: (k['cx'], k['cy'], k['major']) for n, k in seen.items()}
        self.now = {n: 0.0 for n in seen}
        self.last = {}
        dial.save_session({'sign': self.sign, 'zero': self.zero,
                           'stamp': time.strftime('%H:%M')})
        return True

    def measure(self, need=None):
        """Update every knob the camera can currently see.

        `need` is the one knob that has to be among them. The rest being hidden
        by the arm, or off the edge of the frame, is not a reason to refuse: the
        only knob whose angle matters right now is the one being turned.
        """
        if self.open_loop:
            return True
        ks = self.eyes.read_raw()
        if len(ks) < 2:
            return False
        got = {}
        for n, (cx, cy, major) in self.at.items():
            near = min(ks, key=lambda k: np.hypot(k['cx'] - cx, k['cy'] - cy))
            if np.hypot(near['cx'] - cx, near['cy'] - cy) < 0.6 * major:
                got[n] = near['rel']
        for n, rel in got.items():
            self.now[n] = dial.dial(rel, self.zero[n], self.sign)
        self.eyes.note = f'{len(ks)} seen, {len(got)} of {len(self.at)} matched'
        return (need in got) if need else bool(got)


def turn(b, name, target, say):
    """Converge on the target, and stop for anything that is not converging."""
    # Read BEFORE committing to a move. Without this a row the camera cannot
    # currently see still gets one bite, and only then fails the read, so the
    # arm has gone and gripped something on the strength of a stale number.
    if not b.measure(need=name):
        say(f'    cannot see {name} right now, so nothing was moved.')
        say('    re-aim until it is in frame, then try again.')
        return
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
        if b.open_loop:
            # Assumed, not measured. Every check below is about comparing what
            # was asked with what happened, and open loop has nothing to
            # compare, so it credits the full bite and carries on.
            b.now[name] = before + step
            b.last[name] = step
            taken += 1
            say(f'    assumed {b.now[name]:.0f}   (not measured)')
            continue
        if not b.measure(need=name):
            say(f'    cannot see {name} after that bite. stopping.')
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
                 'last': b.last.get(n)} for n in b.knobs]
        print('\033[2J\033[H' + screen.render(rows, {
            'zero': dial.load_session().get('stamp', '--:--'),
            'amp': 'LOCKED' if b.locked() else 'WAITING',
            'camK': b.eyes.note, 'camT': f'drift {b.tag_drift * 1000:.0f}mm',
            'arm': b.arm + ('   OPEN LOOP, numbers ASSUMED' if b.open_loop
                            else '')}))
        if not b.zero:
            print('  no session zero yet. press  c  to calibrate.')
        try:
            cmd = screen.parse(input(' > '), len(b.knobs))
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
        if cmd[0] == 'openloop':
            b.open_loop = not b.open_loop
            say('  OPEN LOOP: bites are now assumed to land, not measured.'
                if b.open_loop else '  back to measuring every bite.')
            input('  [enter]')
            continue

        _, i, deg, multiple = cmd
        name = b.knobs[i - 1]
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
        # An explicit y, because this is the one gate before the arm moves. But
        # say so when it is declined: a silent return to the table looks exactly
        # like the program having ignored you.
        if input('    proceed?  type y then enter:  ').strip().lower() \
                not in ('y', 'yes'):
            say('  cancelled, nothing moved.')
            input('  [enter]')
            continue
        turn(b, name, deg, say)
        input('  [enter]')


STDERR_LOG = '/tmp/knobbrain-brain.log'


def _quiet():
    """Point file descriptor 2 at a file so the screen stays readable.

    libjpeg prints "Corrupt JPEG data: N extraneous bytes" for almost every
    MJPG frame this camera sends. The frames decode fine and the detector is
    unbothered, but the messages come from C, not Python, so sys.stderr cannot
    catch them: the descriptor itself has to move. rcutils writes the ROS INFO
    lines the same way, and those trample the screen too.

    Done AFTER the node is built, so a failure to find a camera is still
    printed where you can see it.
    """
    f = open(STDERR_LOG, 'w')
    os.dup2(f.fileno(), 2)
    return f                    # kept alive: closing it would close fd 2


def main(args=None):
    rclpy.init(args=args)
    b = Brain()
    spin = threading.Thread(target=rclpy.spin, args=(b,), daemon=True)
    spin.start()
    keep = _quiet()
    try:
        tui(b)
    finally:
        # Order matters. Destroying the node while the other thread is inside
        # spin() on it aborts the process from C++, which is what "terminate
        # called without an active exception" is. Shut down first so spin
        # returns, wait for it, and only then take the node apart.
        b.eyes.stop = True
        b.eyes.srv.shutdown()          # free 8080 for the next run
        rclpy.shutdown()
        spin.join(timeout=2.0)
        try:
            b.destroy_node()
        except Exception:
            pass                       # already gone with the context
    return 0


if __name__ == '__main__':
    sys.exit(main())
