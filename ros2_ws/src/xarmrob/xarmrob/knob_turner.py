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

Tested against a REAL ROS 2 graph, not just written: built with colcon in the
CAE ros2-humble container and run against the course's own `command_xarm` in
its no-hardware echo mode. Poses round-trip at zero counts and the whole
regrip search executes over live topics. NOT yet tested against the arm
itself, so the direct path (`scripts/turn_knob.py`) remains the fallback and
is unaffected by anything here.

    apptainer exec /cae/apps/data/ros2-2024/images/ros2-humble.sif bash -c \
      "source /opt/ros/humble/setup.bash && colcon build"
"""
import pathlib
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
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
import runlog        # noqa: E402
import strategy      # noqa: E402
import turn_core     # noqa: E402

JOINT_NAMES = ['cmd00', 'cmd01', 'cmd02', 'cmd03', 'cmd04', 'cmd05', 'cmd06']

# The joint maps are NOT hardcoded here, and that is the whole point. They are
# read from this node's own ROS parameters, which the launch file fills from
# the same robot_xarm_info.yaml that command_xarm is given. Hardcoding them
# means guessing, and guessing was wrong twice in one afternoon: command_xarm's
# built-in defaults disagree with the yaml on joint_01 ([880,500,120] against
# [120,500,880], a mirrored base) and on the gripper ([440,610] against
# [90,610], which is 74 counts of error and made arm.py's open width look
# out of range when it is not). Whatever command_xarm was launched with is
# what we must convert with, so we ask for it rather than assume it.
JOINTS = ('01', '12', '23', '34', '45', '56')

STALE_S = 1.0        # a joint reading older than this is not evidence
SETTLE_S = 0.45      # command_frequency is 5 Hz, so allow two publishes


class RosBackend:
    """turn_core's eight calls, spoken as topics instead of USB."""

    def __init__(self, node, taught_counts, maps, grip_map, speed_steps=14):
        self.node = node
        self._taught = list(taught_counts)
        self.maps = maps                 # joint -> (degrees, counts)
        self.grip_map = grip_map         # (radians, counts)
        self.speed_steps = speed_steps
        self.commanded = list(taught_counts)
        self._actual = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._sent_at = 0.0
        self.pub = node.create_publisher(ME439JointCommand,
                                         '/bus_servo_commands', 1)
        node.create_subscription(JointState, '/joint_states', self._on_state, 1)
        # Own the spinning. Callers of this class run a blocking sequence of
        # moves and squeezes, so nobody is left to pump callbacks; and a
        # caller that spins as well would be racing spin against spin_once.
        self._stop = threading.Event()
        self._spinner = threading.Thread(target=self._spin_forever, daemon=True)
        self._spinner.start()

    def _spin_forever(self):
        # spin_once in a loop rather than spin(), so stop() can end it. Calling
        # destroy_node() underneath a live spin() aborts the process on the way
        # out ("terminate called without an active exception"), which at the
        # bench looks like a crash and skips whatever cleanup came after it.
        while not self._stop.is_set():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except Exception:
                return

    def stop(self):
        self._stop.set()
        self._spinner.join(timeout=1.0)

    # ---- feedback ------------------------------------------------------
    def _on_state(self, msg):
        try:
            counts = self.angles_to_counts(list(msg.position))
        except Exception:
            return                       # a malformed frame is not a reading
        with self._lock:
            self._actual, self._stamp = counts, time.time()

    def actual(self, timeout=2.0, fresh=True):
        """The servos' real positions, or None if nobody is publishing them.

        `fresh` demands a reading that arrived AFTER the last command went out,
        and it is not a nicety. /joint_states runs at 5 Hz while a squeeze step
        settles in 0.45 s, so without this the reader can hand back a frame
        recorded BEFORE the command it is supposed to be measuring, and the
        grip force is then computed against the previous pose. Caught by
        sending six poses through a live graph and finding they came back up to
        318 counts out; with this, zero.
        """
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                new_enough = (self._stamp > self._sent_at if fresh
                              else time.time() - self._stamp < STALE_S)
                if self._actual and new_enough:
                    return list(self._actual)
            time.sleep(0.02)
        return None

    # ---- unit conversion ----------------------------------------------
    @staticmethod
    def _interp(x, xs, ys):
        """np.interp needs an increasing x, and half these tables descend."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs[0] > xs[-1]:
            xs, ys = xs[::-1], ys[::-1]
        return float(np.interp(x, xs, ys))

    def angles_to_counts(self, angles_rad):
        """/joint_states radians -> servo counts, by command_xarm's own maps."""
        out = [self._interp(np.degrees(a), self.maps[j][0], self.maps[j][1])
               for j, a in zip(JOINTS, angles_rad[:6])]
        out.append(self._interp(angles_rad[6], self.grip_map[0],
                                self.grip_map[1]))
        return [int(round(c)) for c in out]

    def counts_to_angles(self, counts):
        """The inverse, which is what command_xarm publishes."""
        out = [np.radians(self._interp(c, self.maps[j][1], self.maps[j][0]))
               for j, c in zip(JOINTS, counts[:6])]
        out.append(self._interp(counts[6], self.grip_map[1], self.grip_map[0]))
        return out

    @property
    def grip_limits(self):
        lo, hi = min(self.grip_map[1]), max(self.grip_map[1])
        return int(lo), int(hi)

    # ---- motion --------------------------------------------------------
    def _send(self, counts):
        counts = [int(np.clip(c, 0, 1000)) for c in counts]
        # Outside the gripper table command_xarm's interp1d raises, which kills
        # the /joint_states publish and with it our only feedback. The bound is
        # whatever the table says, not a constant typed here.
        counts[6] = int(np.clip(counts[6], *self.grip_limits))
        msg = ME439JointCommand()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.command = counts
        msg.enable = True
        self.pub.publish(msg)
        self.commanded = counts
        self._sent_at = time.time()
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

        # Wait for the reading to SETTLE rather than sampling once. A single
        # sample taken SETTLE_S after the last step catches the link mid-ramp:
        # /joint_states runs at 5 Hz and the ramp publishes faster than that,
        # so the first frame back can still describe an earlier step. That read
        # as a 428-count stall on a move that was going perfectly well.
        #
        # Polling until it stops improving keeps the stall check meaningful:
        # a real obstruction never converges, and reports the gap it stuck at.
        off = None
        for _ in range(8):
            self._send(target)
            got = self.actual()
            if got is None:
                self.node.get_logger().error(
                    'no /joint_states. Is command_xarm running, and is arm.py '
                    'stopped? They cannot share the USB.')
                return False
            off = max(abs(g - t) for g, t in zip(got[:6], target[:6]))
            if off <= 60:
                return True
        self.node.get_logger().warn(
            f'stalled {off} counts short of the target: something is in the way')
        return False

    def squeeze(self, want_lag, step=10, cap=590):
        """Close until the servo is visibly pushing. -> (command, force, holding).

        The gap between commanded and actual IS the force: the servo is
        fighting the object by that much. Ramping and watching it beats naming
        a count, because a stalled servo is what browns out the bus.
        """
        c = self.counts()[6]
        cap = min(cap, self.grip_limits[1])
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

    def release(self, to=410):
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
        # A camera index for the real thing, or a path to an image or a folder
        # of them to replay. Replay is not test scaffolding: it is how a run
        # gets rehearsed without the arm in front of you, and how a recorded
        # failure gets re-examined afterwards.
        self.camera = self.declare_parameter('camera', '0').value

        A = ik._arm()
        poses = A.poses()
        if pose not in poses:
            raise SystemExit(f'no taught pose {pose!r}. Teach one first: '
                             f'python3 arm.py teach {pose}')
        self.backend = RosBackend(self, A.counts_of(poses[pose]),
                                  *self.calibration())
        self.offset = np.zeros(2)

    def calibration(self):
        """The joint maps, from OUR parameters, filled by the same yaml.

        Declared with command_xarm's own defaults so a bare `ros2 run` still
        starts. It will be WRONG in the same way command_xarm is wrong in that
        case, which is the point: both nodes agree, right or wrong, instead of
        disagreeing silently. Launch with the params file and both are right.
        """
        defaults = {
            '01': ([-90, 0, 90], [880, 500, 120]),
            '12': ([-180, -90, 0], [870, 500, 120]),
            '23': ([0, 90, 180], [140, 500, 880]),
            '34': ([-112, -90, 0, 90, 112], [1000, 890, 505, 140, 0]),
            '45': ([-112, -90, 0, 90, 112], [0, 120, 490, 880, 1000]),
            '56': ([-112, -90, 0, 90, 112], [0, 120, 500, 880, 1000]),
        }
        # dynamic_typing, because the yaml writes these as bare integers
        # ([-90,0,90]) and a float default declares the parameter DOUBLE_ARRAY,
        # which rclpy then refuses to override with an INTEGER_ARRAY. The node
        # died on startup with exactly that. The values are interpolation
        # tables; whether they arrive as ints or floats is not our business.
        loose = ParameterDescriptor(dynamic_typing=True)

        def table(name, fallback):
            v = self.declare_parameter(name, fallback, loose).value
            return [float(x) for x in v]

        maps, defaulted = {}, []
        for j in JOINTS:
            deg = table(f'rotational_angles_for_mapping_joint_{j}', defaults[j][0])
            cmd = table(f'bus_servo_cmd_for_mapping_joint_{j}', defaults[j][1])
            maps[j] = (deg, cmd)
            if cmd == [float(x) for x in defaults[j][1]]:
                defaulted.append(j)
        gdeg = table('rotational_angles_for_mapping_gripper', [0, 90])
        gcmd = table('bus_servo_cmd_for_mapping_gripper', [90, 610])
        grip = (list(np.radians(gdeg)), list(gcmd))
        if len(defaulted) == len(JOINTS):
            self.get_logger().warn(
                'every joint map came from the built-in defaults, so no params '
                'file was passed. command_xarm is then almost certainly running '
                'on ITS defaults too, where joint_01 is mirrored against the '
                'yaml. Launch with knob_turner.launch.py instead.')
        return maps, grip

    def run(self):
        log = self.get_logger()
        if self.backend.actual(timeout=5.0) is None:
            raise SystemExit(
                'no /joint_states after 5 s. Start the arm controller first:\n'
                '  ros2 run xarmrob command_xarm\n'
                'and make sure scripts/arm.py is NOT running: one USB device, '
                'one controller.')
        # Imported late and only when needed, so a run that is only exercising
        # motion still starts on a machine without OpenCV.
        import knob
        import cv2
        import glob as _glob

        cam, replay = None, []
        if str(self.camera).lstrip('-').isdigit():
            cam = cv2.VideoCapture(int(self.camera))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            for _ in range(8):
                cam.read()
        else:
            path = pathlib.Path(self.camera)
            replay = ([str(path)] if path.is_file()
                      else sorted(_glob.glob(str(path / '*'))))
            if not replay:
                raise SystemExit(f'no frames to replay at {self.camera!r}')
            log.info(f'replaying {len(replay)} frame(s) from {self.camera}')

        def angles():
            if cam is not None:
                ok, frame = cam.read()
                return knob.measure(frame) if ok else {}
            # Walk the folder, holding on the last frame once it runs out, so a
            # loop that reads more often than there are frames still finishes.
            frame = cv2.imread(replay[min(angles.n, len(replay) - 1)])
            angles.n += 1
            return knob.measure(frame) if frame is not None else {}
        angles.n = 0

        # Record it, in the same format and the same place turn_knob.py uses,
        # so runs.py compares a run over topics against one over USB without
        # caring which is which. A run that leaves no evidence cannot be used
        # to improve the setup, which is the entire point of running it.
        rec = runlog.Run(self.degrees, tol=self.tol, squeeze=self.force,
                         transport='ros')
        log.info(f'recording into {rec.dir}')
        try:
            start = angles()
            if not start:
                raise SystemExit('no knobs seen. Run: python3 knob.py calibrate')
            rec.start(start)
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
                    rec.bite(i, bite, None, note='grip failed')
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
                    rec.bite(i, bite, got, note='rejected as impossible')
                    continue
                plan.record(bite, got)
                others = {k: strategy.wrap(after[k]['angle'] - before[k]['angle'])
                          for k in after if k != target_knob}
                rec.bite(i, bite, got, tracking=got / bite if bite else 0.0,
                         others=others)
                log.info(f'commanded {bite:+.0f}, knob followed {got:+.0f} '
                         f'({got / bite * 100:.0f}% tracking)')
            end = angles()
            out = rec.finish(end, knob=target_knob)
            if target_knob:
                log.info(f'{target_knob}: asked {self.degrees:+.0f}, got '
                         f'{out["final"]:+.0f}, error {out["error"]:+.0f} deg  '
                         f'{"WITHIN TOLERANCE" if out["ok"] else "MISSED"}')
            log.info(f'run recorded in {rec.dir}')
        finally:
            if cam is not None:
                cam.release()
            self.backend.release()
            self.backend.park()
            self.backend.stop()
            if 'end' not in rec.log:
                rec.finish()          # a run that died still leaves its trace


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
