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

JOG = 0.015          # metres, the calibration step. Swept in the self-check.
# 5 mm was the first value here and it is too small once the arm's own scatter
# is admitted. A commanded move lands about 1.5 mm off, and a jog measurement
# carries two of those, so a 5 mm jog gives a Jacobian roughly 25 percent
# wrong; 15 mm brings that to 9 percent and 20 mm to 6. At the standoff height
# there is nothing to hit, so the jog can afford to be large. The earlier
# 3/5/8 mm sweep found no difference only because that sim had no scatter to
# be swamped by.
JOG_REPEATS = 3      # average this many jogs per axis, see calibrate()
GAIN = 0.5           # fraction of the pixel error to remove per iteration
TOL_MM = 2.0         # how close is close enough, in MILLIMETRES
# The tolerance has to be a physical distance, not a pixel count. It was 2 px,
# which at this camera is 0.4 mm: finer than the arm can place itself. Each
# commanded move lands roughly 1.5 mm off, so a 0.4 mm target is unreachable
# by construction and the loop simply burned its whole budget and reported
# failure while sitting comfortably inside the 4 mm that actually ejects the
# knob. What matters is beating that 4 mm, so aim at 1.5 mm and convert to
# pixels using the scale the Jacobian already measured.
TOL_PX = 2.0         # only a fallback for when no J has been measured yet
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
        self.tol_mm = TOL_MM
        self.J = None

    def calibrate(self, home):
        """Measure J by jogging x and y from `home`. -> J, pixels per metre.

        Each axis is jogged JOG_REPEATS times and the columns averaged. One
        jog is not enough: the arm places itself about 1.5 mm off a 5 mm
        command, so a single measurement carries roughly 30 percent error and
        the resulting J is wrong enough to cost several extra correction
        steps. Averaging three costs three moves once and is repaid
        immediately.

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
            samples = []
            errors = []
            for rep in range(JOG_REPEATS):
                for step in (self.jog, -self.jog):
                    probe = np.array(home, float)
                    probe[axis] += step
                    try:
                        self.move(probe)
                    except Exception as e:      # unreachable, joint limits, ...
                        errors.append(f'{step*1000:+.0f} mm: {e.__class__.__name__}')
                        continue
                    seen = self.see()
                    if seen is None:
                        errors.append(f'{step*1000:+.0f} mm: gripper not visible')
                        continue
                    samples.append((np.array(seen) - np.array(base)) / step)
                    break
                # re-measure the home reading between repeats, since the arm
                # does not return to exactly the same place either
                self.move(home)
                again = self.see()
                if again is not None:
                    base = np.array(again)
            if not samples:
                raise RuntimeError(
                    f'cannot jog axis {"xy"[axis]} in either direction '
                    f'({"; ".join(errors)})')
            cols.append(np.mean(samples, axis=0))
        self.move(home)
        self.J = np.column_stack(cols)
        # px per mm, from the Jacobian we just measured, so the millimetre
        # tolerance becomes a pixel one without anybody calibrating anything
        scale = float(np.linalg.norm(self.J, axis=0).mean()) / 1000.0
        if scale > 1e-6:
            self.tol_px = max(1.0, self.tol_mm * scale)
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
        best, best_xyz, stale = None, xyz.copy(), 0
        for i in range(self.max_steps):
            seen = self.see()
            if seen is None:
                return xyz, None, i
            err = np.array(target_px, float) - np.array(seen)
            mag = float(np.linalg.norm(err))
            if log is not None:
                log.append(dict(step=i, err_px=round(mag, 2),
                                xy_mm=[round(v * 1000, 2) for v in xyz[:2]]))
            if best is None or mag < best:
                best, best_xyz, stale = mag, xyz.copy(), 0
            else:
                # Once the arm's own scatter dominates, more corrections just
                # shuffle the error around. Two steps without improvement means
                # the floor has been reached, and the best pose seen is a
                # better answer than wherever the last random step landed.
                stale += 1
                if stale >= 2:
                    self.move(best_xyz)
                    return best_xyz, best, i
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
def _rig(noise_px=0.5, seed=0, move_noise_mm=1.5, detect=False):
    """A see/move pair backed by the MuJoCo scene.

    move_noise_mm matters more than the pixel noise and was missing entirely
    from the first version of this rig, which teleported the arm to exactly
    the commanded pose. A real xArm does not: taught poses miss by about 4 mm
    and the plateau spread is 12 counts at 0.42 mm per count, so roughly 5 mm.
    Convergence figures measured against a perfect actuator are figures for a
    robot nobody owns, and the loop exists precisely because this arm is not
    that robot.
    """
    import mujoco, ik, urdfmap, scene, simcam
    model = scene.MODEL
    data = mujoco.MjData(model)
    cam = simcam.SimCam()
    rng = np.random.default_rng(seed)
    gid = model.geom('gripper_marker').id

    def move(xyz):
        # the arm goes roughly where it was told, not exactly
        actual = np.asarray(xyz, float).copy()
        actual[:2] += rng.normal(0, move_noise_mm / 1000.0, 2)
        ang = ik.limit_joint_angles(ik.invkin(actual))
        for n, q in urdfmap.qpos_from(ang).items():
            data.qpos[model.joint(n).qposadr[0]] = q
        mujoco.mj_forward(model, data)

    def see():
        # The marker's true position through the camera model. Fast, and right
        # for testing the control law, but it is not what the bench has:
        # nothing there knows where the gripper is until something looks. Use
        # detect=True for the honest version.
        uv = cam.project(data.geom_xpos[gid])
        if uv is None:
            return None
        return (uv[0] + rng.normal(0, noise_px), uv[1] + rng.normal(0, noise_px))

    def see_detected():
        """Render the frame and actually FIND the gripper in it.

        This is the path the bench will take, and it exercises the renderer,
        the detector and the loop together. It is slower by the cost of a
        render per iteration, which is why it is not the default, but a loop
        that only ever ran on ground-truth coordinates has never been tested
        against pixels at all.
        """
        import sys as _sys, pathlib as _p
        _sys.path.insert(0, str(_p.Path(__file__).parent.parent / 'cv'))
        import gripper
        hit = gripper.find(cam.render(data))
        return None if hit is None else (hit['u'], hit['v'])

    return (see_detected if detect else see), move, cam, data


if __name__ == '__main__':
    import scene as sc

    home = sc.knob_targets()['knob1']
    # The rig now scatters every commanded move by 1.5 mm, because the real arm
    # does. Every number below is therefore about this arm rather than an ideal
    # one; the earlier figures (2 px, "0.3 mm final accuracy") came from a rig
    # that teleported the gripper exactly where it was told.
    see, move, cam, data = _rig(seed=0)

    s = Servo(see, move)
    J = s.calibrate(home)
    scale = float(np.linalg.norm(J, axis=0).mean()) / 1000.0
    print(f'J (px per m):\n{np.round(J, 1)}')
    print(f'  {scale:.1f} px per mm, tolerance {s.tol_mm:.1f} mm = '
          f'{s.tol_px:.1f} px')

    move(home)
    target_px = see()
    EJECT_MM = 4.0        # the measured distance at which the knob escapes
    landed, inside_tol, worst_mm, worst_iters = 0, 0, 0.0, 0
    for k in range(8):
        th = k * np.pi / 4
        start = np.array(home, float)
        start[:2] += 0.010 * np.array([np.cos(th), np.sin(th)])
        xyz, err, iters = s.center(start, target_px)
        assert err is not None, f'lost the gripper from {np.degrees(th):.0f} deg'
        mm = err / scale
        worst_mm = max(worst_mm, mm)
        worst_iters = max(worst_iters, iters)
        landed += mm < EJECT_MM
        inside_tol += err <= s.tol_px
    print(f'\n8 directions from 10 mm out, with a 1.5 mm sloppy actuator:')
    print(f'  {landed}/8 inside the {EJECT_MM:.0f} mm that ejects the knob '
          f'(worst {worst_mm:.2f} mm, at most {worst_iters} steps)')
    print(f'  {inside_tol}/8 also inside the {s.tol_mm:.1f} mm it aims for')
    # The gate is the PHYSICAL requirement. 2 mm is where the loop aims, 4 mm
    # is where the knob actually escapes, and a run landing at 2.6 mm has done
    # its job even though it missed its own tolerance.
    assert landed == 8, f'only {landed}/8 landed inside the ejection distance'
    assert inside_tol >= 6, f'only {inside_tol}/8 reached the aiming tolerance'

    # ---- the jog sweep, now that there is noise for it to matter against ---
    print(f'\n{"jog":>6} {"J off-diagonal error":>21} {"landed":>8} {"worst mm":>9}')
    for jog_mm in (5, 10, 15, 20):
        sj = Servo(see, move, jog=jog_mm / 1000.0)
        Jj = sj.calibrate(home)
        off = (abs(Jj[0, 1]) + abs(Jj[1, 0])) / (abs(Jj[0, 0]) + abs(Jj[1, 1]))
        sc_j = float(np.linalg.norm(Jj, axis=0).mean()) / 1000.0
        move(home)
        tgt = see()
        ok, worst = 0, 0.0
        for k in range(8):
            th = k * np.pi / 4
            st = np.array(home, float)
            st[:2] += 0.010 * np.array([np.cos(th), np.sin(th)])
            _, e, _ = sj.center(st, tgt)
            if e is not None and e <= sj.tol_px:
                ok += 1
                worst = max(worst, e / sc_j)
        print(f'{jog_mm:5d}mm {off:20.0%} {ok:>6}/8 {worst:9.2f}')
    print('\nA bigger jog buys a straighter Jacobian, because the arm\'s 1.5 mm\n'
          'scatter is a smaller fraction of it. The earlier 3/5/8 mm sweep found\n'
          'no difference between them only because that rig had no scatter at all.\n'
          'On the bench the ceiling is whatever starts disturbing the pedal.')

    # ---- a wrong J must still land ---------------------------------------
    print(f'\n{"J error":>9} {"in 4mm":>8} {"worst mm":>9}')
    for scale_err, label in ((0.7, '-30%'), (1.3, '+30%'), (1.5, '+50%')):
        sw = Servo(see, move, max_steps=12)
        sw.calibrate(home)
        sw.J = sw.J * scale_err
        sc_w = float(np.linalg.norm(sw.J, axis=0).mean()) / 1000.0
        move(home)
        tgt = see()
        ok, worst = 0, 0.0
        for k in range(8):
            th = k * np.pi / 4
            st = np.array(home, float)
            st[:2] += 0.010 * np.array([np.cos(th), np.sin(th)])
            _, e, _ = sw.center(st, tgt)
            if e is not None:
                mm = e / sc_w
                worst = max(worst, mm)
                ok += mm < 4.0          # inside the ejection distance
        print(f'{label:>9} {ok:>6}/8 {worst:9.2f}')
        # Judged against the ejection distance, not the aiming tolerance: a
        # deliberately wrong Jacobian is expected to land less precisely, and
        # what matters is whether the knob still ends up in the jaws.
        assert ok >= 7, f'a J off by {label} left {8 - ok} grips outside 4 mm'

    # ---- the same loop, driven by an actual detector on actual pixels -----
    # Everything above reads the marker's position out of the physics state.
    # The bench cannot do that. This runs the identical control law with the
    # gripper found by looking at the rendered image, which is the only
    # version whose result means anything off a laptop.
    see_px, move_px, cam_px, _ = _rig(seed=4, detect=True)
    sp = Servo(see_px, move_px)
    Jp = sp.calibrate(home)
    scale_p = float(np.linalg.norm(Jp, axis=0).mean()) / 1000.0
    move_px(home)
    tgt_px = see_px()
    assert tgt_px is not None, 'the detector could not find the gripper at home'
    landed_px, worst_px_mm = 0, 0.0
    for k in range(8):
        th = k * np.pi / 4
        st = np.array(home, float)
        st[:2] += 0.010 * np.array([np.cos(th), np.sin(th)])
        _, e, _ = sp.center(st, tgt_px)
        if e is not None:
            mm = e / scale_p
            worst_px_mm = max(worst_px_mm, mm)
            landed_px += mm < 4.0
    print(f'\nsame loop driven by the real detector on rendered pixels: '
          f'{landed_px}/8 inside 4 mm, worst {worst_px_mm:.2f} mm')
    assert landed_px == 8, f'only {landed_px}/8 landed when actually looking'
    cam_px.close()

    # ---- refusing to work blind ------------------------------------------
    blind = Servo(lambda: None, move)
    try:
        blind.calibrate(home)
        raise AssertionError('calibrate() succeeded without seeing the gripper')
    except RuntimeError:
        pass
    print('\nblind calibration refuses instead of inventing a Jacobian')
    cam.close()
    print('servo_center self-checks passed')
