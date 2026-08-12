#!/usr/bin/env python3
"""Turn a knob, over ROS 2, using the course's own arm controller.

    ros2 run xarmrob knob_turner --ros-args -p degrees:=115.0

Where this sits in the graph. `command_xarm` owns the USB and nothing else may:
one device, one controller, so `arm.py` must NOT be running at the same time.

    knob_turner ──/bus_servo_commands──▶ command_xarm ──▶ servos
         ▲                                     │
         └────────────/joint_states────────────┘

The feedback edge is the one that matters. `command_xarm.set_joint_state()`
calls `read_bus_servos()` and publishes the ACTUAL servo positions, while the
commands we send are what we asked for. Grip force is the gap between the two,
which is the same quantity `arm.squeeze()` measures over USB, so the whole
control loop survives the hop onto topics.

The sequence itself is not here. It lives in `scripts/turn_core.py` and is
shared with the direct-USB path, so there is one grip-and-turn implementation
and two transports rather than two implementations that drift apart. This file
is the transport, and the joint-angle bookkeeping the transport needs.

UNTESTED AGAINST HARDWARE at the time of writing: developed without a ROS
install to hand. The direct path (`scripts/turn_knob.py`) remains the fallback
and is unaffected by anything here.
"""
import pathlib
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xarmrob_interfaces.msg import ME439JointCommand

# The brains are plain numpy and live with the bench scripts. Import them from
# there rather than copying: a rule tuned in one place has to be the rule that
# runs in the other.
for _p in (pathlib.Path.home() / 'me439' / 'pi' / 'scripts',
           pathlib.Path(__file__).resolve().parents[4] / 'scripts',
           pathlib.Path.home()):
    if (_p / 'turn_core.py').exists():
        sys.path.insert(0, str(_p))
        break
import ik            # noqa: E402
import strategy      # noqa: E402
import turn_core     # noqa: E402

JOINT_NAMES = ['cmd00', 'cmd01', 'cmd02', 'cmd03', 'cmd04', 'cmd05', 'cmd06']

# command_xarm converts counts to radians with interp1d over these tables and
# will RAISE outside them, which kills the /joint_states publish and with it
# our only feedback. arm.py opens the jaws to 410, which is outside. So the
# gripper is clamped here rather than discovering that at the bench.
GRIP_CMD_RANGE = (440, 610)
GRIP_ANGLE_RANGE = np.radians([0.0, 90.0])

STALE_S = 1.0        # a joint reading older than this is not evidence
SETTLE_S = 0.45      # command_frequency is 5 Hz, so allow two publishes


class RosBackend:
    """turn_core's eight calls, spoken as topics instead of USB."""

    def __init__(self, node, taught_counts, speed_steps=14):
        self.node = node
        self._taught = list(taught_counts)
        self.speed_steps = speed_steps
        self.commanded = list(taught_counts)
        self._actual = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self.pub = node.create_publisher(ME439JointCommand,
                                         '/bus_servo_commands', 1)
        node.create_subscription(JointState, '/joint_states', self._on_state, 1)

    # ---- feedback ------------------------------------------------------
    def _on_state(self, msg):
        try:
            counts = self.angles_to_counts(list(msg.position))
        except Exception:
            return                       # a malformed frame is not a reading
        with self._lock:
            self._actual, self._stamp = counts, time.time()

    def actual(self, timeout=2.0):
        """The servos' real positions, or None if nobody is publishing them."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._actual and time.time() - self._stamp < STALE_S:
                    return list(self._actual)
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return None

    # ---- unit conversion ----------------------------------------------
    @staticmethod
    def angles_to_counts(angles_rad):
        """/joint_states radians -> servo counts, using arm.py's own maps."""
        A = ik._arm()
        out = []
        for j, a in zip(A._JOINTS, angles_rad[:6]):
            deg, cmd = A._MAP[j]
            out.append(float(np.interp(np.degrees(a), *(
                (deg, cmd) if deg[0] < deg[-1] else (deg[::-1], cmd[::-1])))))
        out.append(float(np.interp(angles_rad[6], GRIP_ANGLE_RANGE,
                                   GRIP_CMD_RANGE)))
        return [int(round(c)) for c in out]

    # ---- motion --------------------------------------------------------
    def _send(self, counts):
        counts = [int(np.clip(c, 0, 1000)) for c in counts]
        counts[6] = int(np.clip(counts[6], *GRIP_CMD_RANGE))
        msg = ME439JointCommand()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.command = counts
        msg.enable = True
        self.pub.publish(msg)
        self.commanded = counts
        return counts

    def taught(self):
        return list(self._taught)

    def counts(self):
        return self.actual() or list(self.commanded)

    def approach(self, target):
        """Interpolate there, checking for a stall on the way.

        Same two-stage idea as arm.approach: driving straight at a knob has
        pushed the pedal across the desk before. There is no per-step blocking
        read here, so it leans on command_xarm's own rate limiting and checks
        the arrival at the end.
        """
        start = self.counts()
        for k in range(1, self.speed_steps + 1):
            f = k / self.speed_steps
            self._send([round(a + (b - a) * f) for a, b in zip(start, target)])
            time.sleep(SETTLE_S / 2)
        time.sleep(SETTLE_S)
        got = self.actual()
        if got is None:
            self.node.get_logger().error(
                'no /joint_states. Is command_xarm running, and is arm.py '
                'stopped? They cannot share the USB.')
            return False
        off = max(abs(g - t) for g, t in zip(got[:6], target[:6]))
        if off > 60:
            self.node.get_logger().warn(
                f'stalled {off} counts short: something is in the way')
            return False
        return True

    def squeeze(self, want_lag, step=10, cap=590):
        """Close until the servo is visibly pushing. -> (command, force, holding).

        The gap between commanded and actual IS the force: the servo is
        fighting the object by that much. Ramping and watching it beats naming
        a count, because a stalled servo is what browns out the bus.
        """
        c = self.counts()[6]
        lag = peak = 0
        slipped = False
        for _ in range(30):
            c = min(cap, c + step)
            self._send(self.commanded[:6] + [c])
            time.sleep(SETTLE_S)
            got = self.actual()
            if got is None:
                return c, 0, False
            lag = c - got[6]
            peak = max(peak, lag)
            if peak >= 30 and lag < peak - 12:
                slipped = True           # it was held, then squirted out
                break
            if lag >= want_lag or c >= cap:
                break
        return c, lag, (lag >= 30 and not slipped)

    def preclose(self, to=470):
        self._send(self.commanded[:6] + [to])
        time.sleep(SETTLE_S)
        return True

    def release(self, to=GRIP_CMD_RANGE[0]):
        self._send(self.commanded[:6] + [to])
        time.sleep(SETTLE_S)
        return True

    def roll_by(self, deg, roll_index=5):
        A = ik._arm()
        now = self.counts()
        was = float(np.interp(now[roll_index], A.ROLL_CMD, A.ROLL_DEG))
        want = was + deg
        if not A.ROLL_DEG[0] <= want <= A.ROLL_DEG[-1]:
            self.node.get_logger().warn(
                f'roll would reach {want:.0f} deg, outside the joint range. '
                f'Re-grip at a different angle and turn in two bites.')
            return False
        target = list(now)
        target[roll_index] = round(float(np.interp(want, A.ROLL_DEG, A.ROLL_CMD)))
        return self.approach(target)

    def park(self):
        A = ik._arm()
        return self.approach(list(A.NEUTRAL))


class KnobTurner(Node):
    """One knob, to a commanded angle, with the camera as the authority."""

    def __init__(self):
        super().__init__('knob_turner')
        self.degrees = self.declare_parameter('degrees', 115.0).value
        self.tol = self.declare_parameter('tolerance', 8.0).value
        self.tries = self.declare_parameter('bites', 4).value
        self.force = self.declare_parameter('squeeze', 70).value
        pose = self.declare_parameter('pose', 'grip0').value

        A = ik._arm()
        poses = A.poses()
        if pose not in poses:
            raise SystemExit(f'no taught pose {pose!r}. Teach one first: '
                             f'python3 arm.py teach {pose}')
        self.backend = RosBackend(self, A.counts_of(poses[pose]))
        self.offset = np.zeros(2)

    def run(self):
        log = self.get_logger()
        if self.backend.actual(timeout=5.0) is None:
            raise SystemExit(
                'no /joint_states after 5 s. Start the arm controller first:\n'
                '  ros2 run xarmrob command_xarm\n'
                'and make sure scripts/arm.py is NOT running: one USB device, '
                'one controller.')
        # The camera is imported late and only if it is needed, so a run that
        # is only exercising motion still starts on a machine without OpenCV.
        import knob
        import cv2
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        for _ in range(8):
            cam.read()

        def angles():
            ok, frame = cam.read()
            return knob.measure(frame) if ok else {}

        try:
            start = angles()
            if not start:
                raise SystemExit('no knobs seen. Run: python3 knob.py calibrate')
            plan = strategy.Plan(self.degrees, tol=self.tol,
                                 max_bites=self.tries)
            target_knob, done = None, 0.0
            for i in range(1, self.tries + 1):
                now = angles()
                if target_knob:
                    done = strategy.wrap(now[target_knob]['angle']
                                         - start[target_knob]['angle'])
                plan.target, plan.done_deg, plan.why = self.degrees, done, ''
                bite = plan.next_bite()
                if bite is None:
                    log.info(f'stopping: {plan.why}')
                    break
                log.info(f'bite {i}: {done:+.0f} of {self.degrees:+.0f} done')
                before = now
                rolled, self.offset = turn_core.one_bite(
                    self.backend, bite, self.force, offset=self.offset,
                    log=lambda m: log.info(str(m)))
                after = angles()
                if rolled is None:
                    log.warn('the grip failed, so nothing was turned')
                    continue
                if target_knob is None:
                    moved = {k: abs(strategy.wrap(after[k]['angle']
                                                 - before[k]['angle']))
                             for k in before}
                    target_knob = max(moved, key=moved.get)
                    if moved[target_knob] < 5.0:
                        raise SystemExit(f'no knob moved measurably: {moved}')
                    log.info(f'the gripper is on {target_knob}')
                got = strategy.wrap(after[target_knob]['angle']
                                    - before[target_knob]['angle'])
                if plan.implausible(bite, got):
                    log.warn(f'ignoring an impossible reading: commanded '
                             f'{bite:+.0f}, pointer claims {got:+.0f}')
                    continue
                plan.record(bite, got)
                log.info(f'commanded {bite:+.0f}, knob followed {got:+.0f} '
                         f'({got / bite * 100:.0f}% tracking)')
            end = angles()
            if target_knob:
                final = strategy.wrap(end[target_knob]['angle']
                                      - start[target_knob]['angle'])
                log.info(f'{target_knob}: asked {self.degrees:+.0f}, got '
                         f'{final:+.0f}, error {self.degrees - final:+.0f} deg')
        finally:
            cam.release()
            self.backend.release()
            self.backend.park()


def main(args=None):
    rclpy.init(args=args)
    node = KnobTurner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
