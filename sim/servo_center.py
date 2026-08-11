#!/usr/bin/env python3
"""Uncalibrated visual servoing: drive the gripper onto the knob in the image.

Why this exists. Bench data says 4 mm of sideways offset makes the knob eject
from the jaws, while height is forgiving. Taught poses kept missing by about
that much, because a hobby arm does not return to a commanded position
exactly. Rather than trying to calibrate the arm and the camera well enough to
compute the correct move in advance, close the loop in the image: the camera
sees the gripper and the knob in the same frame, and the arm nudges until they
coincide. Positioning error stops mattering because every approach ends with
the camera confirming alignment rather than assuming it.

The whole controller:

  1. once, at the pre-grasp pose, jog JOG mm in x, then in y, and record how
     far the gripper marker moves in the image. Those two measurements are the
     columns of a 2x2 matrix J, in pixels per mm.

  2. loop:  e  = target_px - gripper_px
            dq = GAIN * inv(J) @ e            (mm, capped per step)
     until |e| < TOL_PX or MAX_STEPS.

GAIN below 1 means each step removes only part of the error, which is what
lets the loop converge even when J is wrong by a third or more. That
tolerance is the point: J is measured crudely, on purpose.

The loop needs two functions and nothing else, so the same code runs in
simulation and on the bench:
    see(state)  -> (u, v) of the gripper, or None
    move(xyz)   -> put the endpoint here
"""
import numpy as np

JOG = 0.005          # metres, the calibration step. Swept in the self-check.
GAIN = 0.5           # fraction of the pixel error to remove per iteration
TOL_PX = 2.0         # stop once the gripper is this close in the image
# Each iteration leaves (1 - GAIN) of the error, so reaching TOL_PX from an
# initial error e0 needs about log(TOL/e0)/log(1-GAIN) steps. At roughly
# 5 px/mm a 10 mm offset starts at ~50 px, which wants 7 at GAIN 0.5. Six
# steps at 0.4 was the first setting here and it stalled at 2 to 3 px, which
# reads as a broken loop but is just an exhausted budget.
MAX_STEPS = 8
MAX_STEP_M = 0.005   # never command more than this in one go


class Servo:
    """Holds the measured image Jacobian and runs the correction loop."""

    def __init__(self, see, move, jog=JOG, gain=GAIN, tol_px=TOL_PX,
                 max_steps=MAX_STEPS):
        self.see, self.move = see, move
        self.jog, self.gain = jog, gain
        self.tol_px, self.max_steps = tol_px, max_steps
        self.J = None

    def calibrate(self, home):
        """Measure J by jogging x and y from `home`. -> J, pixels per metre.

        Each axis is jogged forward, and backward if forward will not go. At
        the edge of the workspace +5 mm can be unreachable while -5 mm is
        fine, and the Jacobian column is the same either way once divided by
        the signed step. Without this the calibration dies inside a move on a
        steeply tilted pedal, which is precisely where centring is most
        needed.
        """
        self.move(home)
        base = self.see()
        if base is None:
            raise RuntimeError('cannot see the gripper at the home pose')
        cols = []
        for axis in (0, 1):
            col = None
            errors = []
            for step in (self.jog, -self.jog):
                probe = np.array(home, float)
                probe[axis] += step
                try:
                    self.move(probe)
                except Exception as e:          # unreachable, joint limits, ...
                    errors.append(f'{step*1000:+.0f} mm: {e.__class__.__name__}')
                    continue
                seen = self.see()
                if seen is None:
                    errors.append(f'{step*1000:+.0f} mm: gripper not visible')
                    continue
                col = (np.array(seen) - np.array(base)) / step
                break
            if col is None:
                raise RuntimeError(
                    f'cannot jog axis {"xy"[axis]} in either direction '
                    f'({"; ".join(errors)})')
            cols.append(col)
        self.move(home)
        self.J = np.column_stack(cols)
        if abs(np.linalg.det(self.J)) < 1e-6:
            raise RuntimeError('the two jogs moved the gripper the same way in '
                               'the image, so J cannot be inverted: is the '
                               'camera looking along one of the axes?')
        return self.J

    def center(self, start, target_px, log=None):
        """Correct from `start` until the gripper reaches target_px.

        -> (final_xyz, final_error_px, iterations). Never moves further than
        MAX_STEP_M at a time, so a bad J stays a slow approach, not a lunge.
        """
        if self.J is None:
            raise RuntimeError('call calibrate() first')
        Jinv = np.linalg.inv(self.J)
        xyz = np.array(start, float)
        self.move(xyz)
        for i in range(self.max_steps):
            seen = self.see()
            if seen is None:
                return xyz, None, i
            err = np.array(target_px, float) - np.array(seen)
            mag = float(np.linalg.norm(err))
            if log is not None:
                log.append(dict(step=i, err_px=round(mag, 2),
                                xy_mm=[round(v * 1000, 2) for v in xyz[:2]]))
            if mag < self.tol_px:
                return xyz, mag, i
            step = self.gain * (Jinv @ err)
            n = float(np.linalg.norm(step))
            if n > MAX_STEP_M:
                step *= MAX_STEP_M / n
            xyz[:2] += step
            self.move(xyz)
        seen = self.see()
        err = (None if seen is None
               else float(np.linalg.norm(np.array(target_px) - np.array(seen))))
        return xyz, err, self.max_steps


# ------------------------------------------------------------------ sim rig
def _rig(noise_px=0.5, seed=0):
    """A see/move pair backed by the MuJoCo scene, with detection noise."""
    import mujoco, ik, urdfmap, scene, simcam
    model = scene.MODEL
    data = mujoco.MjData(model)
    cam = simcam.SimCam()
    rng = np.random.default_rng(seed)
    gid = model.geom('gripper_marker').id

    def move(xyz):
        ang = ik.limit_joint_angles(ik.invkin(xyz))
        for n, q in urdfmap.qpos_from(ang).items():
            data.qpos[model.joint(n).qposadr[0]] = q
        mujoco.mj_forward(model, data)

    def see():
        # Use the marker's true position through the camera model rather than
        # colour-detecting the rendered pixels: this test is about whether the
        # control law converges, and a detector would only add its own bugs.
        # noise_px stands in for what a real detector costs.
        uv = cam.project(data.geom_xpos[gid])
        if uv is None:
            return None
        return (uv[0] + rng.normal(0, noise_px), uv[1] + rng.normal(0, noise_px))

    return see, move, cam, data


if __name__ == '__main__':
    import scene as sc

    home = sc.knob_targets()['knob1']
    see, move, cam, data = _rig()

    # ---- gate 1: converge from 10 mm offsets in 8 directions ---------------
    s = Servo(see, move)
    J = s.calibrate(home)
    print(f'J (px per m):\n{np.round(J, 1)}')
    print(f'  scale {np.abs(J).max()/1000:.1f} px per mm, '
          f'so a {JOG*1000:.0f} mm jog moves ~{np.abs(J).max()*JOG:.0f} px')

    move(home)
    target_px = see()
    worst_err, worst_iters, failures = 0.0, 0, []
    for k in range(8):
        th = k * np.pi / 4
        start = np.array(home, float)
        start[:2] += 0.010 * np.array([np.cos(th), np.sin(th)])
        xyz, err, iters = s.center(start, target_px)
        if err is None or err > TOL_PX:
            failures.append((round(np.degrees(th)), err))
        else:
            worst_err, worst_iters = max(worst_err, err), max(worst_iters, iters)
    print(f'\n8 directions from 10 mm out: {8 - len(failures)}/8 converged, '
          f'worst {worst_err:.2f} px in at most {worst_iters} steps')
    assert not failures, f'did not converge from {failures}'

    # ---- gate 2: does the jog size matter? --------------------------------
    print(f'\n{"jog":>5} {"converged":>10} {"worst px":>9} {"max steps":>10} '
          f'{"px per mm":>10}')
    table = []
    for jog_mm in (3, 5, 8):
        sj = Servo(see, move, jog=jog_mm / 1000.0)
        Jj = sj.calibrate(home)
        move(home)
        tgt = see()
        ok, worst, steps = 0, 0.0, 0
        for k in range(8):
            th = k * np.pi / 4
            st = np.array(home, float)
            st[:2] += 0.010 * np.array([np.cos(th), np.sin(th)])
            _, e, it = sj.center(st, tgt)
            if e is not None and e <= TOL_PX:
                ok += 1
                worst, steps = max(worst, e), max(steps, it)
        table.append((jog_mm, ok, worst, steps, np.abs(Jj).max() / 1000))
        print(f'{jog_mm:4d}mm {ok:>7}/8 {worst:>9.2f} {steps:>10} '
              f'{np.abs(Jj).max()/1000:>10.1f}')
    assert all(ok == 8 for _, ok, _, _, _ in table), \
        'some jog size failed to converge in simulation'
    print('\nAll three jog sizes converge in simulation, which is expected: '
          'nothing here\nshoves the pedal or flexes. The sweep matters again on '
          'the real bench, where\na larger jog disturbs the scene and a smaller '
          'one risks drowning in detection\nnoise. Start at 5 mm and re-run this '
          'comparison there.')

    # ---- gate 3: a wrong J must still converge -----------------------------
    # This is the actual claim of uncalibrated servoing. If it only worked
    # with an accurate Jacobian it would be ordinary calibration wearing a
    # different name, and the bench, where J is measured crudely against a
    # flexing arm, would break it.
    # The error is not symmetric, and the direction matters on the bench.
    # Overestimating J shrinks every step (effective gain GAIN/scale), so the
    # loop stays stable and merely needs more iterations. Underestimating
    # enlarges the steps and is the direction that could overshoot. So the
    # budget, not the gain, is what an inaccurate calibration spends.
    print(f'\n{"J error":>9} {"eff gain":>9} {"budget":>7} {"converged":>10} '
          f'{"worst px":>9} {"max steps":>10}')
    for scale, label, budget in ((0.7, '-30%', MAX_STEPS), (1.3, '+30%', MAX_STEPS),
                                 (1.5, '+50%', MAX_STEPS), (1.5, '+50%', 12)):
        sw = Servo(see, move, max_steps=budget)
        sw.calibrate(home)
        sw.J = sw.J * scale                 # pretend the calibration was off
        move(home)
        tgt = see()
        ok, worst, steps = 0, 0.0, 0
        for k in range(8):
            th = k * np.pi / 4
            st = np.array(home, float)
            st[:2] += 0.010 * np.array([np.cos(th), np.sin(th)])
            _, e, it = sw.center(st, tgt)
            if e is not None and e <= TOL_PX:
                ok += 1
                worst, steps = max(worst, e), max(steps, it)
        print(f'{label:>9} {GAIN/scale:>9.2f} {budget:>7} {ok:>7}/8 '
              f'{worst:>9.2f} {steps:>10}')
        if scale <= 1.3 or budget > MAX_STEPS:
            assert ok >= 7, f'a J off by {label} broke the loop ({ok}/8)'
        else:
            assert ok >= 1, f'a J off by {label} diverged rather than slowed'

    # ---- the loop must not pretend to work when it cannot see -------------
    blind = Servo(lambda: None, move)
    try:
        blind.calibrate(home)
        raise AssertionError('calibrate() succeeded without seeing the gripper')
    except RuntimeError:
        pass
    print('\nblind calibration refuses instead of inventing a Jacobian')
    cam.close()
    print('servo_center self-checks passed')
