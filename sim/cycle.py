#!/usr/bin/env python3
"""The whole turn, end to end, in simulation.

    locate      where is the pedal (on the bench: ArUco; here: the scene truth)
    plan        knob positions follow from the pedal pose, not from a taught pose
    approach    stand off above the knob, jaws pre-closed
    centre      visual servoing until the gripper is on the knob in the image
    descend     down to grip height, which is only safe once pre-closed
    turn        roll the wrist
    verify      read the pointer angle before and after and compare

Every stage can refuse, and the refusal is the useful part: a stage that
cannot be done safely says so and the cycle stops with a reason rather than
driving the arm into the pedal.

    python3 cycle.py            # one full turn on the centre knob
    python3 cycle.py 0 90       # knob0, 90 degrees
"""
import numpy as np
import mujoco
import ik
import scene
import simcam
import servo_center
from servo_center import Servo

STANDOFF = 0.030     # metres above the grip target to approach at
PRECLOSE = 1.15      # jaw angle (rad) before descending; below this the open
                     # jaws foul the neighbouring knobs. 0.90 is the bare
                     # minimum for the centre knob and leaves the outer ones
                     # 0.7 mm inside a neighbour, so this carries real margin.
TURN_TOL = 8.0       # degrees, matches turn_knob.py
JAW_OPEN, JAW_SHUT = 0.0, 1.83


# The course invkin chooses the gripper's pitch from HEIGHT alone
# (gripper_angle = z/0.4334 * -pi + pi/2), which at knob height is 61 degrees
# below horizontal. That steep wrist is what limits reach: a pedal 40 mm
# further out than nominal is "unreachable" with it and comfortably reachable
# at 20 to 40 degrees. So rather than accepting the default's verdict, try
# flatter angles before declaring a target out of range.
GRIPPER_ANGLES = [None, np.radians(40), np.radians(30), np.radians(20),
                  np.radians(50), np.radians(10)]


def choose_gripper_angle(poses, knob, jaw):
    """First gripper pitch that works for EVERY pose in `poses`.

    Every pose, not just the grip target: the arm has to fly to the standoff
    before it descends, and 30 mm higher is a different reach problem. Checking
    only the destination let the planner approve a target whose standoff was
    out of range, and the run then died inside a move instead of refusing.

    -> (found, angle, grade). `found` is a separate flag rather than testing
    the angle against None, because None is a legitimate answer meaning "the
    height-based default works". Returning None for both "use the default"
    and "nothing worked" made every nominal target look unreachable.
    """
    last = None
    for ga in GRIPPER_ANGLES:
        ok = True
        for p in poses:
            g = scene.grade(p, gripper_angle=ga, target_knob=knob, jaw=jaw)
            last = g
            if not (g.get('reachable') and g.get('above_floor')
                    and not g.get('collisions')):
                ok = False
                break
        if ok:
            return True, ga, last
    return False, None, last


class Cycle:
    """Runs one knob. Holds the sim state so stages can be inspected."""

    def __init__(self, cam=None, noise_px=0.5, seed=0):
        self.model = scene.MODEL
        self.data = mujoco.MjData(self.model)
        self.cam = cam or simcam.SimCam(model=self.model)
        self.rng = np.random.default_rng(seed)
        self.noise_px = noise_px
        self.jaw = JAW_OPEN
        self.gripper_angle = None      # set by the plan stage
        self.steps = []

    # ---- primitives the stages are built from ---------------------------
    def move(self, xyz):
        scene._pose(self.data, xyz, self.gripper_angle)
        self._apply_jaw()

    def _apply_jaw(self):
        for jn in ('arm1', 'arm1_left'):
            try:
                self.data.qpos[self.model.joint(jn).qposadr[0]] = self.jaw
            except KeyError:
                pass
        mujoco.mj_forward(self.model, self.data)

    def set_jaw(self, value):
        self.jaw = float(np.clip(value, JAW_OPEN, JAW_SHUT))
        self._apply_jaw()

    def see(self):
        uv = self.cam.project(self.data.geom('gripper_marker').xpos)
        if uv is None:
            return None
        return (uv[0] + self.rng.normal(0, self.noise_px),
                uv[1] + self.rng.normal(0, self.noise_px))

    def knob_angle(self, name):
        """Pointer angle in degrees, read from the tab's actual orientation."""
        q = self.data.qpos[self.model.joint(f'{name}_turn').qposadr[0]]
        return float(np.degrees(q) % 360.0)

    def turn_knob(self, name, deg):
        j = self.model.joint(f'{name}_turn').qposadr[0]
        self.data.qpos[j] += np.radians(deg)
        mujoco.mj_forward(self.model, self.data)

    def collisions(self, xyz, target_knob):
        # pass the CURRENT jaw angle: the same pose is a collision with the
        # jaws open and clear once pre-closed, which is the whole point
        g = scene.grade(xyz, gripper_angle=self.gripper_angle,
                        target_knob=target_knob, jaw=self.jaw)
        return g.get('collisions', []), g

    # ---- the cycle ------------------------------------------------------
    def run(self, knob='knob1', degrees=90.0, verbose=True):
        log = dict(knob=knob, want=degrees, stages=[], ok=False)

        def stage(name, ok, note=''):
            log['stages'].append(dict(name=name, ok=bool(ok), note=note))
            if verbose:
                print(f'  {"ok  " if ok else "STOP"} {name:10s} {note}')
            return ok

        targets = scene.knob_targets()
        if knob not in targets:
            stage('locate', False, f'no such knob: {knob}')
            return log
        target = targets[knob]
        stage('locate', True,
              f'{knob} at ({target[0]*1000:.0f}, {target[1]*1000:.0f}, '
              f'{target[2]*1000:.0f}) mm')

        # Can the arm get there, with SOME gripper pitch? The default is
        # chosen from height alone and is the binding constraint far out.
        self.set_jaw(PRECLOSE)
        stand = np.array(target, float)
        stand[2] += STANDOFF
        found, ga, g = choose_gripper_angle([stand, target], knob, PRECLOSE)
        if not found:
            why = (g.get('why', 'out of reach') if not g.get('reachable') else
                   'below the z floor' if not g.get('above_floor') else
                   f'always fouls {g.get("collisions", ["something"])[0]}')
            stage('plan', False, f'no workable gripper angle: {why}')
            return log
        self.gripper_angle = ga
        stage('plan', True,
              'reachable, above the floor, gripper at '
              + ('the height-based default' if ga is None
                 else f'{np.degrees(ga):.0f} deg'))
        hits, _ = self.collisions(stand, knob)
        if hits:
            stage('approach', False, f'standoff fouls {hits[0]}')
            return log
        self.move(stand)
        stage('approach', True,
              f'{STANDOFF*1000:.0f} mm above, jaws pre-closed to '
              f'{np.degrees(PRECLOSE):.0f} deg')

        # centre in the image, at the standoff height
        servo = Servo(self.see, self.move)
        try:
            servo.calibrate(stand)
        except RuntimeError as e:
            stage('centre', False, str(e))
            return log
        self.move(stand)
        want_px = self.cam.project(target + np.array([0, 0, STANDOFF]))
        start = np.array(stand, float)
        start[:2] += self.rng.normal(0, 0.004, 2)       # a taught pose is not exact
        xyz, err, iters = servo.center(start, want_px)
        if err is None or err > servo_center.TOL_PX:
            stage('centre', False,
                  f'still {err if err is None else round(err,1)} px out after '
                  f'{iters} steps')
            return log
        stage('centre', True,
              f'{err:.1f} px ({err*simcam.mm_per_px(self.cam, self.cam.cam.distance):.2f} mm) '
              f'in {iters} steps')

        # descend to grip height, keeping the corrected x and y
        grip = np.array([xyz[0], xyz[1], target[2]])
        hits, _ = self.collisions(grip, knob)
        if hits:
            stage('descend', False, f'fouls {hits[0]}')
            return log
        self.move(grip)
        stage('descend', True, f'at grip height {grip[2]*1000:.0f} mm, clear')

        # grip, turn, verify by reading the pointer
        self.set_jaw(JAW_SHUT)
        before = self.knob_angle(knob)
        self.turn_knob(knob, degrees)
        after = self.knob_angle(knob)
        moved = (after - before + 180) % 360 - 180
        self.set_jaw(PRECLOSE)
        ok = abs(moved - degrees) <= TURN_TOL
        stage('turn', True, f'commanded {degrees:+.0f} deg')
        stage('verify', ok, f'pointer moved {moved:+.1f} deg '
                            f'({"within" if ok else "outside"} {TURN_TOL:.0f} deg)')
        log.update(ok=ok, moved=moved, centre_px=err, iters=iters)

        self.move(stand)
        return log


def run_once(knob='knob1', degrees=90.0, verbose=True):
    c = Cycle()
    out = c.run(knob, degrees, verbose)
    c.cam.close()
    return out


if __name__ == '__main__':
    import sys
    which = f'knob{sys.argv[1]}' if len(sys.argv) > 1 else 'knob1'
    deg = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0

    scene.reset()
    print(f'full cycle on {which}, {deg:+.0f} degrees\n')
    log = run_once(which, deg)
    print(f'\nresult: {"turned" if log["ok"] else "did not complete"}')

    # every knob, at the nominal setup
    print('\nall three knobs:')
    results = {}
    for k in ('knob0', 'knob1', 'knob2'):
        scene.reset()
        r = run_once(k, 90.0, verbose=False)
        results[k] = r
        last = r['stages'][-1]
        print(f'  {k}: {"OK" if r["ok"] else "stopped at " + last["name"]:22s} '
              f'{last["note"]}')
    assert all(r['ok'] for r in results.values()), \
        'the nominal setup should complete on every knob'

    # a refusal must actually refuse: put the knob under the floor
    scene.configure(h=0.010)
    bad = run_once('knob1', 90.0, verbose=False)
    scene.reset()
    assert not bad['ok'], 'a knob below the z floor should not have completed'
    assert bad['stages'][-1]['name'] in ('plan', 'approach', 'descend'), \
        f'stopped at the wrong stage: {bad["stages"][-1]}'
    print(f'\nbelow-floor pedal correctly refused at '
          f'"{bad["stages"][-1]["name"]}": {bad["stages"][-1]["note"]}')
    print('cycle self-checks passed')
