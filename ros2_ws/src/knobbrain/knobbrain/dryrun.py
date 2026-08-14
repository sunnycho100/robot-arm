#!/usr/bin/env python3
"""Run the real decision loop against a simulated arm, with no ROS installed.

brain.turn() is the function that decides how many times to grip a knob and
when to give up, and it is the one piece that cannot be exercised on the bench
without an audience watching. It only ever touches its argument through a
handful of attributes, so a stand-in that turns a number is enough to drive it.

The ROS imports at the top of brain.py are stubbed rather than avoided, so what
runs here is the same turn() that will run on the Pi, not a copy of it.

    python3 dryrun.py
"""
import sys
import types


def _stub():
    """Enough of rclpy and his package for `import brain` to succeed."""
    for name in ('rclpy', 'std_msgs', 'std_msgs.msg', 'xarmrob',
                 'xarmrob.preset_controller', 'xarmrob_interfaces',
                 'xarmrob_interfaces.msg'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['std_msgs.msg'].Float32 = type('Float32', (), {})
    sys.modules['xarmrob_interfaces.msg'].ME439PointXYZ = type(
        'ME439PointXYZ', (), {})
    sys.modules['xarmrob.preset_controller'].PresetController = type(
        'PresetController', (), {})


_stub()
import brain    # noqa: E402
import dial     # noqa: E402


class FakeArm:
    """A knob that turns by `keep` of whatever it is asked for."""

    def __init__(self, keep=1.0, blind_after=None, drift=0.0):
        self.now = {'treble': 0.0}
        self.target, self.last = {}, {}
        self.tag_drift = drift
        self.keep = keep
        self.blind_after = blind_after
        self.bites = []

    def bite(self, name, deg):
        self.bites.append(deg)
        self.now[name] += deg * self.keep

    def measure(self):
        return not (self.blind_after is not None
                    and len(self.bites) > self.blind_after)


def run(arm, target=75.0):
    out = []
    brain.turn(arm, 'treble', target, out.append)
    return '\n'.join(str(s) for s in out)


def main():
    # a grip that works: arrives, and in the number of bites we promised
    a = FakeArm(keep=1.0)
    assert 'DONE' in run(a)
    assert len(a.bites) == 3, a.bites
    assert abs(a.now['treble'] - 75.0) < dial.TOL_DEG

    # a grip that slips badly but consistently: still arrives, never overshoots
    a = FakeArm(keep=0.55)
    log = run(a)
    assert 'DONE' in log, log
    assert a.now['treble'] <= 75.0 + dial.TOL_DEG, a.now
    assert len(a.bites) <= dial.MAX_BITES

    # a grip on air: caught on the FIRST bite, not hunted
    a = FakeArm(keep=0.0)
    log = run(a)
    assert 'SLIPPING' in log and len(a.bites) == 1, (len(a.bites), log)

    # the wrist sign inverted: caught on the first bite, and named
    a = FakeArm(keep=-1.0)
    log = run(a)
    assert 'WRONG WAY' in log and 'WRIST_SIGN' in log
    assert len(a.bites) == 1, 'it must not keep driving the wrong way'

    # the amp was nudged: nothing moves at all
    a = FakeArm(keep=1.0, drift=0.020)
    log = run(a)
    assert 'AMP MOVED' in log and a.bites == [], a.bites

    # the camera loses the row mid-run: stop, do not carry on blind
    a = FakeArm(keep=1.0, blind_after=1)
    log = run(a)
    assert 'cannot see' in log and len(a.bites) == 2, (len(a.bites), log)

    # counterclockwise is the same loop with a sign
    a = FakeArm(keep=1.0)
    a.now['treble'] = 100.0
    assert 'DONE' in run(a, 25.0)
    assert all(b < 0 for b in a.bites) and len(a.bites) == 3, a.bites

    # a target already satisfied costs nothing
    a = FakeArm(keep=1.0)
    a.now['treble'] = 73.0
    assert 'DONE' in run(a) and a.bites == []

    # and the cap really does stop a knob that never gets there. 0.3 is chosen
    # to deliver 7.5 degrees a bite: poor, but above the half-tolerance floor
    # that check_bite calls slipping, so this exercises the cap and not the
    # slip abort.
    a = FakeArm(keep=0.3)
    log = run(a, 300.0)
    assert len(a.bites) == dial.MAX_BITES, a.bites
    assert 'gave up' in log, log

    print(f'dryrun: 9 scenarios pass, {dial.MAX_BITES} bite cap holds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
