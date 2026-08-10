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

SCRIPTS = pathlib.Path(__file__).parent.parent / 'scripts'   # repo's arm.py

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
    """Import arm.py with the USB layer stubbed (FK + maps only, no hardware)."""
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
    """endpoint (m) -> (counts, achieved_xyz, modified). The one-call API."""
    A = _arm()
    ang = invkin(endpoint, gripper_angle)
    if np.any(np.isnan(ang)):
        raise ValueError(f'target {tuple(endpoint)} m is out of reach '
                         f'(try a different gripper_angle)')
    ang = limit_joint_angles(ang)
    counts = counts_from(ang)
    achieved = A.endpoint(counts)
    modified = not np.allclose(achieved, endpoint, atol=0.003)
    return counts, achieved, modified


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
    for t in targets:
        xyz, ga = (t[:3], t[3]) if len(t) == 4 else (t, None)
        counts, got, mod = solve(xyz, ga)
        err = float(np.linalg.norm((np.array(xyz) - got))) * 1000
        assert not np.isnan(err), f'nan leaked through solve() for {xyz}'
        if not mod:                    # clipped targets miss by design
            worst = max(worst, err)
        f = lambda v: f'({v[0]*1000:6.1f},{v[1]*1000:6.1f},{v[2]*1000:6.1f})'
        print(f'{f(np.array(xyz)):26s} {f(got):26s} {err:6.2f}  {mod}')
    assert worst < 3.0, f'round-trip error {worst:.2f} mm, IK and FK disagree'
    try:                               # unreachable target must raise, not nan
        solve((0.50, 0.0, 0.05))
        raise AssertionError('out-of-reach target did not raise')
    except ValueError:
        pass
    print(f'\nround-trip OK, worst error {worst:.2f} mm; unreachable raises')
