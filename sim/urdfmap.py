#!/usr/bin/env python3
"""Map course joint angles (arm.py / ik.py convention) to URDF qpos.

The URDF's zero pose is the arm pointing straight up; ours has each link
along the course diagram axes. Both use the same Ry sign, so each joint
differs by a constant offset:

  q_arm6 = alpha0                      base yaw, same axis and sign
  q_arm5 = beta1 + 90 deg              upper arm: URDF zero +z, ours +x
  q_arm4 = beta2 - ELBOW deg           forearm zero is tilted ELBOW above the upper arm
  q_arm3 = beta4 + ELBOW deg           wrist pitch, same tilt back
  q_arm2 = gamma5                      wrist roll

ELBOW = atan2(r_34.z, r_34.x) = 50.41 deg, the course lower-arm diagonal.
Run this file directly to verify the map against MuJoCo joint anchors.
"""
import numpy as np
import ik

ELBOW = np.arctan2(ik.r_34[2], ik.r_34[0])          # 50.41 deg


def qpos_from(angles):
    """Course angles [alpha0, beta1, beta2, gamma3, beta4, gamma5] -> URDF qpos
    dict. gamma3 (forearm roll) has no URDF joint; invkin always sets it 0."""
    a0, b1, b2, g3, b4, g5 = angles
    assert abs(g3) < 1e-9, 'URDF has no forearm-roll joint; gamma3 must be 0'
    return {'arm6': a0,
            'arm5': b1 + np.pi / 2,
            'arm4': b2 - ELBOW,
            'arm3': b4 + ELBOW,
            'arm2': g5}


# ---- course-side reference positions for verification ----------------------
def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def _Ry(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def wrist_and_roll(angles):
    """Course-chain positions of the wrist-pitch joint and the wrist-roll
    joint (50 mm further along the gripper axis, per the URDF)."""
    a0, b1, b2, g3, b4, g5 = angles
    wrist = ik.r_01 + _Rz(a0) @ (ik.r_12 + _Ry(b1) @ (ik.r_23 + _Ry(b2) @ ik.r_34))
    grip_dir = _Rz(a0) @ _Ry(b1 + b2 + b4) @ np.array([1., 0., 0.])
    return wrist, wrist + grip_dir * 0.050


if __name__ == '__main__':
    import pathlib, mujoco
    HERE = pathlib.Path(__file__).parent
    model = mujoco.MjModel.from_xml_path(str(HERE / 'xarm_1s_mj.urdf'))
    data = mujoco.MjData(model)

    rng = np.random.default_rng(0)
    worst_w = worst_r = 0.0
    for _ in range(20):
        a0 = rng.uniform(-2.0, 2.0)
        b1 = rng.uniform(-2.5, 0.0)          # course shoulder range
        b2 = rng.uniform(0.3, 2.5)
        b4 = rng.uniform(-1.5, 1.5)
        ang = [a0, b1, b2, 0.0, b4, 0.0]
        for name, q in qpos_from(ang).items():
            data.qpos[model.joint(name).qposadr[0]] = q
        mujoco.mj_forward(model, data)
        w_ref, r_ref = wrist_and_roll(ang)
        dw = np.linalg.norm(data.joint('arm3').xanchor - w_ref) * 1000
        dr = np.linalg.norm(data.joint('arm2').xanchor - r_ref) * 1000
        worst_w, worst_r = max(worst_w, dw), max(worst_r, dr)
    print(f'20 random poses: wrist-pitch anchor worst {worst_w:.2f} mm, '
          f'wrist-roll anchor worst {worst_r:.2f} mm')
    assert worst_w < 2.0, 'wrist map wrong: URDF and course chain disagree'
    assert worst_r < 4.0, 'wrist-pitch offset wrong'
    print('angle map verified')
