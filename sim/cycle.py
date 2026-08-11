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
PRECLOSE = 0.05      # jaw angle (rad) before descending: NEARLY CLOSED.
# This was 1.15 rad, and it was backwards. The URDF drives the gripper from
# one joint and declares the other three as <mimic> joints; MuJoCo's importer
# drops mimic tags and freezes those joints, so the model had a single finger
# swinging 41 mm while the rest stood still. It was not a gripper closing at
# all, and every clearance measured on it was meaningless.
#
# With all four joints driven, the tip separation runs from 28 mm at jaw 0 to
# 88 mm at jaw 1.4: jaw ZERO is the narrow end. Measured penetration at grip
# height, centre knob:
#
#     jaw    tips    neighbours    pedal
#    0.00   28 mm       0.7 mm    16.0 mm
#    0.60   62 mm       6.1 mm    11.3 mm
#    1.15   84 mm       4.1 mm     0.1 mm
#
# So the fingers have to be nearly SHUT to pass between knobs 24 mm apart, and
# the old advice would have splayed them wide and caused exactly the collision
# it was meant to prevent.
GRIP_LIFT = 0.006    # grip this far above the cap centre, see below
GRAZE_MM = 1.2       # contact shallower than this is inside the model's own
                     # error and is not treated as a collision

# WHAT THIS MODEL CANNOT TELL US. Sweeping the grip height with the gripper
# nearly shut, on the centre knob:
#
#     grip z   above cap   into pedal   into neighbours   onto target knob
#      69 mm       0 mm      15.9 mm           2.0 mm            4.4 mm
#      75 mm       6 mm       9.1 mm           0.0 mm            1.8 mm
#      81 mm      12 mm       2.2 mm           0.0 mm            0.0 mm
#      84 mm      15 mm       0.0 mm           0.0 mm            0.0 mm
#
# There is NO height where the jaws are on the knob and clear of the pedal:
# gripping needs z at most 80, clearing needs at least 84. The fingers in this
# URDF are simply too long for a 14 mm knob standing on a pedal, and the real
# gripper demonstrably manages it. So the model's millimetre clearances are
# not evidence about the hardware, and the pedal contact below is treated as
# a known artefact rather than a refusal.
#
# What survives is directional and worth acting on: the jaws splay to 88 mm
# when open and the knobs are 24 mm apart, so the gripper MUST be nearly shut
# before it goes down between them. The exact opening is a bench measurement,
# not a simulation result.
PEDAL_CONTACT_IS_MODEL_ARTEFACT = True
TURN_TOL = 8.0       # degrees, matches turn_knob.py
EJECT_MM = 4.0       # measured: more lateral error than this and the knob
                     # escapes the jaws. This, not the servo's own aiming
                     # tolerance, is what decides whether a grip is usable.
JAW_OPEN, JAW_SHUT = 0.0, 1.83


# The course invkin chooses the gripper's pitch from HEIGHT alone
# (gripper_angle = z/0.4334 * -pi + pi/2), which at knob height is 61 degrees
# below horizontal. That steep wrist is what limits reach: a pedal 40 mm
# further out than nominal is "unreachable" with it and comfortably reachable
# at 20 to 40 degrees. So rather than accepting the default's verdict, try
# flatter angles before declaring a target out of range.
GRIPPER_ANGLES = [None, np.radians(40), np.radians(30), np.radians(20),
                  np.radians(50), np.radians(10)]


ROLL_LIMIT = 110.0   # the wrist roll joint's range, degrees either way
ROLL_CANDIDATES = [0, -60, 60, -75, 75, -45, 45, -90, 90, -30, 30]


def choose_wrist_roll(target, knob, jaw, turn_deg):
    """Wrist angle to grip at: clear of the neighbours, with room left to turn.

    Two things fight here. The fingers must not reach along the knob row,
    because that is where the neighbouring knobs are: measured penetration at
    a 45 degree pedal rotation runs to 7.4 mm at the default wrist angle and
    to zero at 60 to 90 degrees away from it. But the wrist roll is ALSO what
    turns the knob, so gripping at 90 degrees spends most of the joint's 110
    degree range and leaves nothing to turn with in that direction.

    So candidates are tried nearest-first and the first one that both clears
    the neighbours and leaves room for the commanded turn wins.
    """
    best_clear = None
    for roll in ROLL_CANDIDATES:
        if abs(roll + turn_deg) > ROLL_LIMIT:
            continue                       # no room left to make the turn
        g = scene.grade(target, target_knob=knob, jaw=jaw, graze_mm=GRAZE_MM,
                        wrist_roll=np.radians(roll))
        hits = [h for h in g.get('collisions', [])
                if 'knob' in h and not (PEDAL_CONTACT_IS_MODEL_ARTEFACT
                                        and 'pedal' in h)]
        if g.get('reachable') and not hits:
            return float(np.radians(roll))
        if best_clear is None:
            best_clear = float(np.radians(roll))
    return best_clear if best_clear is not None else 0.0


def choose_gripper_angle(poses, knob, jaw, wrist_roll=None):
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
            g = scene.grade(p, gripper_angle=ga, target_knob=knob, jaw=jaw,
                            graze_mm=GRAZE_MM, wrist_roll=wrist_roll)
            last = g
            hits = [h for h in g.get('collisions', [])
                    if not (PEDAL_CONTACT_IS_MODEL_ARTEFACT and 'pedal' in h)]
            if not (g.get('reachable') and g.get('above_floor') and not hits):
                ok = False
                break
        if ok:
            return True, ga, last
    return False, None, last


class Cycle:
    """Runs one knob. Holds the sim state so stages can be inspected."""

    def __init__(self, cam=None, noise_px=0.5, seed=0, move_noise_mm=1.5):
        self.model = scene.MODEL
        self.data = mujoco.MjData(self.model)
        self.cam = cam or simcam.SimCam(model=self.model)
        self.rng = np.random.default_rng(seed)
        self.noise_px = noise_px
        # The arm lands roughly this far from where it was told. Leaving it at
        # zero made every centring figure a figure for a robot nobody owns.
        self.move_noise_mm = move_noise_mm
        self.jaw = JAW_OPEN
        self.gripper_angle = None      # set by the plan stage
        self.wrist_roll = None         # set by the plan stage
        self.slip = 1.0                # per-knob grip efficiency, see turn_knob
        self.steps = []

    # ---- primitives the stages are built from ---------------------------
    def move(self, xyz):
        actual = np.asarray(xyz, float).copy()
        if self.move_noise_mm:
            actual[:2] += self.rng.normal(0, self.move_noise_mm / 1000.0, 2)
        scene._pose(self.data, actual, self.gripper_angle, wrist_roll=self.wrist_roll)
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
        """Pointer angle in degrees, from the physics state.

        Ground truth, useful for asking whether the CONTROL logic works. It is
        not verification: the bench has no such number, and a 'verify' stage
        built on it certifies geometry against itself. Use read_pointer() for
        the version that has to survive a camera.
        """
        q = self.data.qpos[self.model.joint(f'{name}_turn').qposadr[0]]
        return float(np.degrees(q) % 360.0)

    def read_pointer(self, name):
        """Pointer angle by RENDERING the scene and reading the pixels.

        This is what the bench will do. It runs the real knob finder and the
        real pointer reader over a rendered frame, so the verification path is
        exercised rather than assumed. Returns None when the frame cannot be
        read, which is itself the honest answer.
        """
        import sys as _sys, pathlib as _p
        _sys.path.insert(0, str(_p.Path(__file__).parent.parent / 'cv'))
        import knobs2, pointer
        frame = self.cam.render(self.data)
        uv = self.cam.project(self.data.geom(f'{name}_cap').xpos)
        if uv is None:
            return None
        found = knobs2.find(frame)
        if not found:
            return None
        # pick the detection nearest where this knob should appear
        k = min(found, key=lambda d: (d['cx'] - uv[0]) ** 2 + (d['cy'] - uv[1]) ** 2)
        if np.hypot(k['cx'] - uv[0], k['cy'] - uv[1]) > 3 * k['r_px']:
            return None
        return pointer.angle(frame, k)

    def turn_knob(self, name, deg):
        """Roll the wrist by deg. The knob follows only partly.

        On the bench the knob slips inside the jaws under torque while the
        force reading still says the grip is healthy: commanded +90 produced
        +19, commanded +69 produced +28. `slip` reproduces that, because a
        controller that is only ever tested against a perfect knob will be
        tuned for a robot that does not exist.
        """
        j = self.model.joint(f'{name}_turn').qposadr[0]
        eff = self.slip.get(name, 1.0) if isinstance(self.slip, dict) else self.slip
        self.data.qpos[j] += np.radians(deg * eff)
        mujoco.mj_forward(self.model, self.data)

    def collisions(self, xyz, target_knob):
        # pass the CURRENT jaw angle: the same pose is a collision with the
        # jaws open and clear once pre-closed, which is the whole point
        g = scene.grade(xyz, gripper_angle=self.gripper_angle,
                        target_knob=target_knob, jaw=self.jaw,
                        graze_mm=GRAZE_MM, wrist_roll=self.wrist_roll)
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
        # Stand off along the knob's OWN axis, not straight up. On a tilted
        # pedal the shaft is not vertical, so a vertical retreat approaches the
        # knob at an angle to it and the jaws meet the cap off-square.
        stand = np.array(target, float) + STANDOFF * scene.knob_normal()
        grip_pose = np.array(target, float) + GRIP_LIFT * scene.knob_normal()
        # Wrist FIRST: it decides which way the fingers reach and therefore
        # whether anything collides at all. Choosing the arm's pitch against a
        # wrist orientation that is then changed validates a pose the run never
        # flies.
        self.wrist_roll = choose_wrist_roll(grip_pose, knob, PRECLOSE, degrees)
        found, ga, g = choose_gripper_angle([stand, grip_pose], knob, PRECLOSE,
                                            self.wrist_roll)
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
                 else f'{np.degrees(ga):.0f} deg')
              + f', wrist at {np.degrees(self.wrist_roll):+.0f} deg')
        hits, _ = self.collisions(stand, knob)
        if hits:
            stage('approach', False, f'standoff fouls {hits[0]}')
            return log
        try:
            self.move(stand)
        except ValueError as e:
            stage('approach', False, str(e))
            return log
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
        try:
            xyz, err, iters = servo.center(start, want_px)
        except ValueError as e:
            stage('centre', False, f'ran out of workspace while correcting: {e}')
            return log
        # Judge in millimetres against the distance that actually ejects the
        # knob, not against a pixel count. Comparing to the module's old 2 px
        # constant, rather than the tolerance the servo computed from its own
        # measured scale, rejected every perfectly good grip the moment the
        # arm was given realistic scatter.
        scale = float(np.linalg.norm(servo.J, axis=0).mean()) / 1000.0
        err_mm = None if err is None else err / scale
        if err_mm is None or err_mm > EJECT_MM:
            stage('centre', False,
                  f'still {"lost" if err_mm is None else f"{err_mm:.1f} mm"} out '
                  f'after {iters} steps, which would eject the knob')
            return log
        stage('centre', True,
              f'{err_mm:.2f} mm in {iters} steps'
              + ('' if err <= servo.tol_px else ' (outside the 2 mm target but '
                                               'inside the grip)'))

        # descend to grip height, keeping the corrected x and y
        grip = np.array([xyz[0], xyz[1], target[2]]) + GRIP_LIFT * scene.knob_normal()
        hits, _ = self.collisions(grip, knob)
        if PEDAL_CONTACT_IS_MODEL_ARTEFACT:
            hits = [h for h in hits if 'pedal' not in h]
        if hits:
            stage('descend', False, f'fouls {hits[0]}')
            return log
        try:
            self.move(grip)
        except ValueError as e:
            stage('descend', False, str(e))
            return log
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
