#!/usr/bin/env python3
"""Course IK, callable without ROS.

invkin() and limit_joint_angles() are lifted from the ME439 course package
(xarmrob/xarm_kinematics.py, period23 copy) with the ROS wrapper removed and
the declared-parameter defaults inlined. The math is untouched except
np.asfarray -> np.asarray(dtype=float) (asfarray was removed in NumPy 2).

Frame convention matches arm.py: x forward, y left, z up, angles in radians.
counts_from() converts the six joint angles to servo counts in arm.py's
order (base..wrist + gripper), using the same piecewise-linear maps.

Run this file directly for the round-trip self-check against arm.endpoint().
"""
import sys, types, pathlib
import numpy as np

SCRIPTS = pathlib.Path(__file__).parent          # arm.py is the sibling

# ---- course parameter defaults (xarm_kinematics.py lines 48-81) -----------
r_01 = np.array([0., 0., 0.074])
r_12 = np.array([0.010, 0., 0.])
r_23 = np.array([0.101, 0., 0.])
r_34 = np.array([0.0627, 0., 0.0758])
r_6end = np.array([0.133, 0., -0.003])   # centre of the open gripper pincer
y_rotation_sign = 1

ROTLIM = np.radians([[-150, 150], [-180, 0], [0, 180],
                     [-110, 110], [-100, 100], [-110, 111]])


def invkin(endpoint, gripper_angle=None):
    """Joint angles (rad) for an endpoint (m). Course code, ROS stripped.

    gripper_angle: pitch of the gripper below horizontal, radians.
    None = the course's height-adjusted default (option 2 in the original).
    """
    xyz = np.asarray(endpoint, dtype=float)
    gamma3 = 0
    gamma5 = 0

    if gripper_angle is None:
        # Gripper Assumption option 2: gripper angle adjusted by height
        gripper_angle = xyz[2] / .4334 * -np.pi + np.pi / 2

    # Wrist to Gripper in the plane:
    Rgrip = np.array([[np.cos(gripper_angle), 0, np.sin(gripper_angle)],
                      [0, 1, 0],
                      [-np.sin(gripper_angle), 0, np.cos(gripper_angle)]])
    gripper_offset_RTZ = Rgrip.dot(r_6end)

    # First the out-of-plane rotation
    alpha0 = np.arctan2(xyz[1], xyz[0])

    # Radial and vertical distances spanned by the two links of the arm
    R = np.linalg.norm(xyz[0:2])
    dR = R - gripper_offset_RTZ[0] - r_12[0]
    dz = xyz[2] - gripper_offset_RTZ[2] - r_01[2] - r_12[2]

    # "Overall elevation" angle from shoulder to wrist (+y rotations push down)
    psi = -np.arctan2(dz, dR)
    L1 = np.linalg.norm(r_23)
    L2 = np.linalg.norm(r_34)
    H = np.linalg.norm(np.array((dz, dR)))
    phi = np.arccos((L2**2 - L1**2 - H**2) / (-2 * L1 * H))  # nan if H out of reach

    beta1 = psi - phi                                   # elbow-up solution
    beta2VL = np.arctan2(H * np.sin(phi), H * np.cos(phi) - L1)
    beta2_offset_from_VL = np.arctan2(r_34[2], r_34[0])
    beta2 = beta2VL + beta2_offset_from_VL

    beta1 = beta1 * y_rotation_sign
    beta2 = beta2 * y_rotation_sign

    # beta4 cancels beta1+beta2 so the gripper holds gripper_angle
    beta4 = -(beta1 + beta2) + gripper_angle

    return np.asarray([alpha0, beta1, beta2, gamma3, beta4, gamma5], dtype=float)


def limit_joint_angles(angles):
    return np.array([np.clip(a, lo, hi) for a, (lo, hi) in zip(angles, ROTLIM)])


# ---- servo counts, arm.py's own maps ---------------------------------------
def _arm():
    """Import arm.py with the USB layer stubbed (FK + maps only, no hardware).

    When arm.py is ITSELF the running program its module is named __main__, and
    importing `arm` on top of that would execute the file a second time and open
    a second xarm.Controller on the same USB device. So the running program wins.
    """
    main = sys.modules.get('__main__')
    if hasattr(main, '_MAP'):
        return main
    if 'arm' not in sys.modules:
        stub = types.ModuleType('xarm')
        stub.Controller = lambda *a, **k: None
        sys.modules['xarm'] = stub
        sys.path.insert(0, str(SCRIPTS))
    import arm
    return arm


def counts_from(angles_rad, grip=500):
    """Six joint angles (rad) -> servo counts in arm.py order (+ gripper)."""
    A = _arm()
    deg = np.degrees(angles_rad)
    counts = [float(np.interp(a, *A._MAP[j])) for j, a in zip(A._JOINTS, deg)]
    return counts + [grip]


def solve(endpoint, gripper_angle=None):
    """endpoint (m) -> (counts, achieved_xyz, clipped). The one-call API.

    `clipped` means a joint hit its limit, and ONLY that. It is deliberately
    not "the endpoint came out different": a target can be missed because the
    arm cannot get there (clipped) or because the two kinematic models
    disagree (a bug). Reporting both as one flag hides the second inside the
    first, which is exactly how a wrong link length once slipped past the
    round-trip check below.
    """
    A = _arm()
    ang = invkin(endpoint, gripper_angle)
    if np.any(np.isnan(ang)):
        raise ValueError(f'target {tuple(endpoint)} m is out of reach '
                         f'(try a different gripper_angle)')
    limited = limit_joint_angles(ang)
    clipped = not np.allclose(limited, ang, atol=1e-9)
    counts = counts_from(limited)
    achieved = A.endpoint(counts)
    return counts, achieved, clipped


# ---- nudging a taught pose by millimetres ----------------------------------
# The bench problem this exists for: a taught pose is a few millimetres off, the
# gripper sits low on the knob or short of it, and the only fix used to be
# re-teaching the whole pose by hand. A nudge corrects it numerically instead.
NUDGE_MAX_MM = 30.0        # past this it is a move, not a correction
NUDGE_TOL_MM = 2.0         # how far the result may land from where it was asked


def gripper_angle_of(counts):
    """The gripper's pitch below horizontal, recovered from a pose, in radians.

    invkin sets beta4 = -(beta1 + beta2) + gripper_angle, so the three
    y-rotations of a pose sum back to the angle it was solved for. Recovering
    it is the whole reason a nudge stays a nudge: solving with invkin's
    height-based default instead would re-pitch the wrist and swing the fingers
    off the knob, which is the opposite of a small correction.
    """
    a = _arm().angles_of(counts)
    return a[1] + a[2] + a[4]


def nudged(counts, dx_mm=0.0, dy_mm=0.0, dz_mm=0.0, max_mm=NUDGE_MAX_MM):
    """Same pose, shifted by millimetres. Returns (counts, achieved_mm).

    Relative on purpose. arm.endpoint()'s ABSOLUTE accuracy has never been
    verified, so a nudge never asks the models to be right, only to agree with
    themselves: FK says where the arm is, IK is asked for that same point plus
    the offset, and the shared model error cancels out of the difference. This
    is the argument the height floor already rests on.

    The wrist roll, the out-of-plane wrist joint and the gripper opening are
    carried over from the input pose rather than re-solved, because invkin
    returns zero for the first two and writing that back would spin the wrist
    away from the grip the pose was taught with. Restoring them moves the
    endpoint though (the tool point sits 3 mm off the roll axis), so the target
    is re-aimed by whatever that displacement turns out to be, and what the
    caller gets back is the distance actually travelled, not the one requested.
    """
    A = _arm()
    d = np.asarray([dx_mm, dy_mm, dz_mm], dtype=float) / 1000.0
    if np.linalg.norm(d) * 1000 > max_mm:
        raise ValueError(f'{np.linalg.norm(d)*1000:.0f} mm is a move, not a '
                         f'nudge (limit {max_mm:.0f} mm). Re-teach the '
                         f'pose instead.')
    here = A.endpoint(counts)
    goal = here + d
    aim, out = goal.copy(), list(counts)
    for _ in range(4):
        ang = invkin(aim, gripper_angle_of(counts))
        if np.any(np.isnan(ang)):
            raise ValueError(f'no solution {np.linalg.norm(d)*1000:.0f} mm from '
                             f'here: that offset leaves the workspace')
        out = [int(round(c)) for c in counts_from(limit_joint_angles(ang),
                                                  grip=counts[A.GRIP])]
        out[3], out[5] = counts[3], counts[5]     # keep the taught wrist
        err = goal - A.endpoint(out)
        if np.linalg.norm(err) * 1000 < 0.5:      # under one servo count
            break
        aim = aim + err
    achieved = (A.endpoint(out) - here) * 1000
    missed = float(np.linalg.norm(achieved - d * 1000))
    if missed > NUDGE_TOL_MM:
        raise ValueError(
            f'asked for ({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f}) mm but the '
            f'closest reachable pose moves ({achieved[0]:+.1f},'
            f'{achieved[1]:+.1f},{achieved[2]:+.1f}), off by {missed:.1f} mm. '
            f'A joint is at its limit, or this pose is too far out of the '
            f'plane the course IK assumes. Re-teach rather than nudge.')
    return out, achieved


if __name__ == '__main__':
    targets = [
        (0.20, 0.00, 0.10),
        (0.18, 0.05, 0.12),
        (0.22, -0.04, 0.08),
        (0.16, 0.08, 0.15),
        (0.25, 0.00, 0.05, 0.0),      # needs a flatter gripper to reach
    ]
    print(f'{"target (mm)":26s} {"achieved (mm)":26s} {"err mm":7s} clipped')
    worst = 0.0
    unclipped = 0
    for t in targets:
        xyz, ga = (t[:3], t[3]) if len(t) == 4 else (t, None)
        counts, got, clipped = solve(xyz, ga)
        err = float(np.linalg.norm((np.array(xyz) - got))) * 1000
        assert not np.isnan(err), f'nan leaked through solve() for {xyz}'
        if not clipped:                # a clipped target misses by design
            worst = max(worst, err)
            unclipped += 1
        f = lambda v: f'({v[0]*1000:6.1f},{v[1]*1000:6.1f},{v[2]*1000:6.1f})'
        print(f'{f(np.array(xyz)):26s} {f(got):26s} {err:6.2f}  {clipped}')
    # Guard the guard: if a change made every target clip, `worst` would stay
    # 0.0 and the assert below would pass while testing nothing at all.
    assert unclipped >= 4, (f'only {unclipped} targets stayed inside the joint '
                            f'limits, so the round-trip barely ran')
    assert worst < 3.0, f'round-trip error {worst:.2f} mm, IK and FK disagree'
    try:                               # unreachable target must raise, not nan
        solve((0.50, 0.0, 0.05))
        raise AssertionError('out-of-reach target did not raise')
    except ValueError:
        pass
    print(f'\nround-trip OK, worst error {worst:.2f} mm; unreachable raises')

    # ---- nudging a taught pose -------------------------------------------
    # What this has to prove before anyone trusts it against a real pedal:
    # the arm ends up where the millimetres asked, the taught wrist is still
    # where it was, and asking for nothing does nothing.
    base = [int(round(c)) for c in solve((0.20, 0.02, 0.10))[0]]
    worst_n, checked = 0.0, 0
    for roll_cmd in (500, 880, 120, 300):          # a taught pose is rarely at 0
        pose = list(base)
        pose[5] = roll_cmd
        for d in [(0, 0, -4), (3, 0, 0), (0, -5, 2), (-6, 4, -3)]:
            out, got = nudged(pose, *d)
            assert out[5] == roll_cmd, (f'the nudge spun the taught wrist roll '
                                        f'{roll_cmd} -> {out[5]}')
            assert out[3] == pose[3], 'the nudge moved the out-of-plane wrist'
            assert out[6] == pose[6], 'the nudge changed the gripper opening'
            # Landing on the right POINT is not enough. The fingers also have
            # to arrive at the same angle they were taught at, and a nudge
            # solved with the wrong pitch still hits the point: it just gets
            # there with the wrist rolled somewhere else, which lifts the
            # fingers off the knob. Nothing above would notice.
            repitched = abs(gripper_angle_of(out) - gripper_angle_of(pose))
            assert repitched < 0.02, (f'the nudge re-pitched the gripper by '
                                      f'{np.degrees(repitched):.1f} degrees')
            worst_n = max(worst_n, float(np.linalg.norm(got - np.asarray(d))))
            checked += 1
    # One servo count is about half a millimetre out here, so the floor on this
    # number is quantisation, not the maths. It is the ceiling that matters: if
    # it climbs past a millimetre the re-aiming loop has stopped converging.
    assert checked == 16, f'only {checked} nudges ran'
    assert worst_n < 1.0, f'worst nudge landed {worst_n:.2f} mm from where asked'

    # Nudging by zero must be EXACTLY a no-op. Otherwise every inspection of a
    # pose would walk it slightly, and a pose would rot each time it was checked.
    same, moved = nudged(base, 0, 0, 0)
    assert same == base, f'nudging by zero changed the pose: {base} -> {same}'
    assert not np.any(moved), f'nudging by zero moved the arm {moved} mm'

    # A nudge corrects a pose; it does not travel across the bench.
    try:
        nudged(base, 0, 0, -60)
        raise AssertionError('an oversized nudge was allowed through')
    except ValueError:
        pass

    # The gripper pitch has to come back out of a pose, because a nudge solved
    # with the wrong pitch would re-aim the fingers instead of shifting them.
    # A pose whose wrist clipped no longer HOLDS the angle it was asked for, so
    # it cannot be used to test recovery: what comes back is the pitch the arm
    # actually ended up at, which is the right answer to a different question.
    recovered = 0
    for ga in (0.3, 0.6, 0.845, 1.0):
        c, _, clipped = solve((0.20, 0.0, 0.10), ga)
        if clipped:
            continue
        back = gripper_angle_of([int(round(x)) for x in c])
        assert abs(back - ga) < 0.02, (f'gripper angle {ga:.3f} rad came back '
                                       f'as {back:.3f}')
        recovered += 1
    assert recovered >= 3, f'only {recovered} pitches stayed inside the limits'
    print(f'nudge OK, worst {worst_n:.2f} mm over {checked} offsets; '
          f'zero is a no-op, oversize raises, pitch recovers')
