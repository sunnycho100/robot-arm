#!/usr/bin/env python3
"""MuJoCo scene: the arm + table + Boss DS-1 pedal + 3 knobs.

grade(xyz) answers, for a candidate gripper target:
  reachable      IK solves without hitting joint limits
  above_floor    endpoint z respects the safety floor
  collisions     arm-vs-pedal/table contact pairs (empty = clean approach)

render(xyz) saves a PNG of the arm posed at the target, for eyeballing.

PEDAL numbers marked MEASURE are bench placeholders: set them from the real
tape-down on bench day (pedal position in the arm base frame, knob spacing).
"""
import pathlib
import numpy as np
import mujoco
import ik, urdfmap

HERE = pathlib.Path(__file__).parent


def _quat(R):
    """Rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])

PEDAL = dict(
    x=0.200,        # MEASURE: pedal centre forward of arm base (m)
    y=0.000,        # MEASURE: sideways offset (m)
    w=0.073,        # DS-1 width (across the knob row)
    d=0.129,        # DS-1 depth
    h=0.055,        # DS-1 body height
    knob_dy=0.024,  # MEASURE: knob centre spacing along the row
    knob_r=0.0095,  # knob body radius
    knob_h=0.014,   # knob body sticks up this far above the pedal top
    cap_r=0.005,    # metal cap radius (CAP_MM/2)
    yaw=0.0,        # pedal rotation about z, radians (0 = square to the arm)
    pitch=0.0,      # tilt about the pedal's y axis: + tips the far edge up,
                    # so the top face leans TOWARD the arm. This is the case
                    # where a pedal is propped on something or on a sloped
                    # board, and it moves the knob tops in x and z at once.
    roll=0.0,       # tilt about x: one end of the knob row higher than the other
)
_PEDAL0 = dict(PEDAL)   # pristine bench values, for reset()
Z_FLOOR = 0.045     # MEASURE: from ~/z_floor.json on the Pi

# where to hang the tracked marker: link2 is the body after the wrist roll,
# and the FK endpoint sits at this fixed offset in its frame (see build())
GRIPPER_BODY = 'link2'
GRIPPER_MARKER = (0.00485, 0.0, 0.0828)

# The URDF drives the gripper from ONE joint and declares the other three as
# <mimic> joints of it. MuJoCo's URDF importer silently drops mimic tags and
# gives those joints a zero range instead, so only a single distal finger
# segment moved and the other three stayed frozen: the model was not closing a
# gripper at all, it was swinging one finger aside. These are the multipliers
# and offsets from the URDF, applied by hand in _apply_jaw.
GRIPPER_MIMIC = {
    'arm1':       (1.00,  0.0),      # the driven joint
    'arm1_left':  (-1.00, 0.0),
    'arm0':       (-1.01, -0.3),
    'arm0_left':  (1.01,  0.3),
}


def set_jaw(data, jaw, model=None):
    """Drive every finger joint from one jaw value, honouring the mimics."""
    m = model or MODEL
    for name, (mult, off) in GRIPPER_MIMIC.items():
        try:
            adr = m.joint(name).qposadr[0]
        except KeyError:
            continue
        data.qpos[adr] = mult * float(jaw) + off


def knob_targets():
    """Grip target (m) for each knob: centre of the cap, in the arm frame.

    The row runs along the pedal's own y axis and is rotated by PEDAL['yaw'],
    so a pedal that is not square to the arm still gives correct targets. This
    is what ArUco buys us on the bench: the tags give the pedal's pose, and
    the knob positions follow from it rather than from a taught guess.
    """
    R = pedal_rotation()
    centre = np.array([PEDAL['x'], PEDAL['y'], PEDAL['h'] / 2])
    up = PEDAL['h'] / 2 + PEDAL['knob_h']      # centre of the box to the cap
    out = {}
    for i in range(3):
        local = np.array([0.0, (i - 1) * PEDAL['knob_dy'], up])
        out[f'knob{i}'] = centre + R @ local
    return out


def pedal_rotation():
    """The pedal's orientation as a rotation matrix, from yaw, pitch and roll.

    Applied yaw, then pitch, then roll, which is also the order the quaternion
    handed to MuJoCo is built in. Keeping one function as the single source of
    that convention is deliberate: the knob positions and the rendered body
    have to agree exactly, and two hand-written rotation chains drift apart.
    """
    y, p, r = (PEDAL.get(k, 0.0) for k in ('yaw', 'pitch', 'roll'))
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def knob_normal():
    """Unit vector out of the pedal's top face, in the arm frame.

    On a tilted pedal the knob shaft is no longer vertical, so approaching
    straight down means approaching at an angle to the shaft. This is what the
    approach direction should follow."""
    return pedal_rotation() @ np.array([0.0, 0.0, 1.0])


def configure(**kw):
    """Move or reshape the pedal and rebuild the model. Returns the new MODEL.

    Used by the edge-case sweep to ask what happens when the pedal is further
    away, rotated, taller, or has a different knob spacing.
    """
    global MODEL
    PEDAL.update(kw)
    MODEL = build()
    return MODEL


def reset():
    """Put the pedal back where the bench measurements say it is."""
    return configure(**_PEDAL0)


def build():
    spec = mujoco.MjSpec.from_file(str(HERE / 'xarm_1s_mj.urdf'))
    # the importer gave the mimic joints a zero range, which pins them shut
    for j in spec.joints:
        if j.name in ('arm1_left', 'arm0', 'arm0_left'):
            j.range = [-2.2, 2.2]
            j.limited = 0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    w = spec.worldbody
    w.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[1, 1, 0.1], rgba=[.55, .5, .45, 1])
    ped = w.add_body(name='pedal', pos=[PEDAL['x'], PEDAL['y'], PEDAL['h'] / 2],
                     quat=list(_quat(pedal_rotation())))
    ped.add_geom(name='pedal_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[PEDAL['d'] / 2, PEDAL['w'] / 2, PEDAL['h'] / 2],
                 rgba=[.95, .45, .1, 1])                      # DS-1 orange
    _R = pedal_rotation()
    _q = list(_quat(_R))
    _centre = np.array([PEDAL['x'], PEDAL['y'], PEDAL['h'] / 2])
    for i, (name, t) in enumerate(knob_targets().items()):
        base = _centre + _R @ np.array([0.0, (i - 1) * PEDAL['knob_dy'],
                                        PEDAL['h'] / 2])
        k = w.add_body(name=name, pos=list(base), quat=_q)
        # A hinge about z so the knob can actually be turned, which is what
        # makes the pointer reading testable rather than decorative.
        # the hinge is about the knob's OWN axis, which on a tilted pedal is
        # not the world z
        k.add_joint(name=f'{name}_turn', type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=[0, 0, 1], damping=0.05)
        k.add_geom(name=f'{name}_body', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[PEDAL['knob_r'], PEDAL['knob_h'] / 2],
                   pos=[0, 0, PEDAL['knob_h'] / 2], rgba=[.1, .1, .1, 1])
        k.add_geom(name=f'{name}_cap', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[PEDAL['cap_r'], 0.001],
                   pos=[0, 0, PEDAL['knob_h'] + 0.001], rgba=[.8, .8, .82, 1])
        # The white pointer, painted across the cap edge and onto the SKIRT,
        # which is where the real one is. It used to sit wholly inside the cap,
        # and the reader searches outward from the cap edge precisely because
        # the real pointer ends on the dark skirt, so nothing was ever found.
        # Sim geometry that quietly disagrees with the hardware makes the
        # perception look broken when it is the model that is wrong.
        _tab_in, _tab_out = PEDAL['cap_r'] * 0.45, PEDAL['knob_r'] * 0.92
        k.add_geom(name=f'{name}_tab', type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[(_tab_out - _tab_in) / 2, PEDAL['cap_r'] * 0.18,
                         0.0011],
                   pos=[(_tab_out + _tab_in) / 2, 0,
                        PEDAL['knob_h'] + 0.0012],
                   rgba=[1, 1, 1, 1])
    w.add_light(pos=[0.3, -0.3, 0.8], dir=[-0.3, 0.3, -0.8])

    # A marker the camera can find on the gripper, which is what the servoing
    # loop actually tracks. It sits at the FK endpoint: measured in link2's own
    # frame it is (4.85, 0, 82.8) mm and stays there to within 0.2 mm across
    # poses, which is the check that link2 is the right body to hang it on.
    # On the real bench this role is played by a sticker or a tag on the
    # gripper; here it is a saturated green ball, since nothing else in the
    # scene is green.
    for b in spec.bodies:
        if b.name == GRIPPER_BODY:
            b.add_geom(name='gripper_marker',
                       type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0],
                       pos=list(GRIPPER_MARKER), rgba=[0, 1, 0, 1],
                       contype=0, conaffinity=0)     # visual only, never collides
            break
    return spec.compile()


MODEL = build()

# Identify scene objects by GEOM name, never by body name. MuJoCo welds a body
# that has no joint into `world` when the model compiles, and the body name is
# gone after that: every pedal and table contact reports its body as 'world'.
# An earlier version of grade() matched on body names, found nothing called
# 'pedal', and returned "no collisions" for every pose ever tested. Geom names
# survive welding, so they are the thing to match on.
SCENE_GEOMS = {'table', 'pedal_box'}
for _i in range(3):
    SCENE_GEOMS |= {f'knob{_i}_body', f'knob{_i}_cap', f'knob{_i}_tab'}


def _owner(geom_name):
    """Which scene object a geom belongs to, or None if it is part of the arm."""
    if geom_name in ('table', 'pedal_box'):
        return geom_name.replace('_box', '')
    if geom_name.startswith('knob'):
        return geom_name.split('_')[0]
    return None


def _pose(data, xyz, gripper_angle=None, jaw=None, wrist_roll=None):
    counts, achieved, clipped = ik.solve(xyz, gripper_angle)
    ang = ik.limit_joint_angles(ik.invkin(xyz, gripper_angle))
    if wrist_roll is not None:
        # gamma5, the wrist roll. Free at grip time: it decides which way the
        # fingers reach, and the turn starts from wherever it is left.
        ang[5] = float(wrist_roll)
    for name, q in urdfmap.qpos_from(ang).items():
        data.qpos[MODEL.joint(name).qposadr[0]] = q
    if jaw is not None:
        set_jaw(data, jaw)
    mujoco.mj_forward(MODEL, data)
    return counts, achieved, clipped


def grade(xyz, gripper_angle=None, target_knob=None, jaw=None,
          graze_mm=0.0, wrist_roll=None):
    """jaw: finger angle in radians. It matters more than it sounds: with the
    jaws wide open the fingers splay out far enough to foul the knobs either
    side, so the same pose is a collision open and clear pre-closed."""
    data = mujoco.MjData(MODEL)
    try:
        counts, achieved, clipped = _pose(data, xyz, gripper_angle, jaw,
                                          wrist_roll)
    except ValueError as e:
        return dict(reachable=False, why=str(e))
    hits = []
    for c in data.contact:
        g1 = MODEL.geom(c.geom1).name or ''
        g2 = MODEL.geom(c.geom2).name or ''
        o1, o2 = _owner(g1), _owner(g2)
        if (o1 is None) == (o2 is None):
            continue                        # arm on arm, or scene on scene
        obj = o1 or o2
        # the arm's own geoms are unnamed in the URDF, so name them by body
        arm_geom = c.geom2 if o1 else c.geom1
        part = MODEL.body(MODEL.geom_bodyid[arm_geom]).name
        if target_knob and obj == target_knob:
            continue                        # touching the knob we are gripping
        depth = -c.dist * 1000
        if depth < graze_mm:
            continue      # shallower than the model's own geometric error
        hits.append(f'{part} vs {obj} ({depth:.1f} mm in)')
    return dict(reachable=not clipped, achieved=np.round(achieved, 4),
                above_floor=achieved[2] >= Z_FLOOR,
                collisions=sorted(set(hits)), counts=[round(c) for c in counts])


def render(xyz, gripper_angle=None, out='scene.png', cam_dist=0.55):
    data = mujoco.MjData(MODEL)
    try:
        _pose(data, xyz, gripper_angle)
    except ValueError:
        mujoco.mj_forward(MODEL, data)
    cam = mujoco.MjvCamera()
    cam.lookat = [PEDAL['x'] * 0.6, 0, 0.10]
    cam.distance, cam.azimuth, cam.elevation = cam_dist, 135, -20
    with mujoco.Renderer(MODEL, 720, 1080) as r:
        r.update_scene(data, camera=cam)
        px = r.render()
    import PIL.Image
    PIL.Image.fromarray(px).save(HERE / out)
    return HERE / out


if __name__ == '__main__':
    print(f'model: {MODEL.ngeom} geoms, {MODEL.nbody} bodies')
    for name, t in knob_targets().items():
        g = grade(t, target_knob=name)
        print(f'{name} at {np.round(t, 3)}: {g}')
    g = grade(knob_targets()['knob1'], target_knob='knob1')
    assert g['reachable'] and g['above_floor'], 'centre knob must be reachable'

    # The knobs must sit ON the pedal. Reachability alone does not notice a
    # wrong spacing: knobs scattered a quarter of a metre apart are still
    # perfectly reachable, and would silently become the geometry the whole
    # pipeline trusts.
    t = knob_targets()
    for name, p in t.items():
        dy = abs(p[1] - PEDAL['y'])
        assert dy <= PEDAL['w'] / 2, (
            f'{name} sits {dy*1000:.0f} mm off centre, outside the '
            f'{PEDAL["w"]*1000:.0f} mm wide pedal')
    span = max(p[1] for p in t.values()) - min(p[1] for p in t.values())
    assert 0.02 <= span <= PEDAL['w'], (
        f'knob row spans {span*1000:.0f} mm, which does not fit a '
        f'{PEDAL["w"]*1000:.0f} mm pedal')
    # The tracked marker must sit ON the gripper endpoint, at every pose. The
    # servoing loop centres whatever feature it is given, so a marker parked
    # somewhere else converges just as happily and puts the jaws in the wrong
    # place. Nothing downstream can detect that; only this check can.
    import mujoco as _mj
    _d = mujoco.MjData(MODEL)
    for _t in list(knob_targets().values()) + [(0.18, 0.03, 0.09)]:
        _pose(_d, _t)
        marker = _d.geom('gripper_marker').xpos
        tip = ik._arm().endpoint(ik.counts_from(
            ik.limit_joint_angles(ik.invkin(_t))))
        off = float(np.linalg.norm(marker - tip)) * 1000
        assert off < 1.0, (f'the tracked marker sits {off:.1f} mm from the '
                           f'gripper endpoint, so servoing would centre the '
                           f'wrong point')

    # Collision detection must actually detect. The first version of grade()
    # matched scene objects by BODY name, but MuJoCo welds jointless bodies
    # into `world` and their names disappear, so it reported "no collisions"
    # for every pose ever graded. A check that cannot go red is not a check:
    # drive the arm straight into the pedal and demand it says so.
    buried = grade([PEDAL['x'], 0.0, 0.030])
    assert buried.get('collisions'), \
        'a pose driven into the pedal reported no collisions'
    assert any('pedal' in h for h in buried['collisions']), \
        f'the pedal was not among the reported hits: {buried["collisions"]}'

    # and it must stay quiet when the arm is nowhere near anything
    clear = grade([PEDAL['x'], 0.0, 0.140])
    assert not clear.get('collisions'), \
        f'a pose well clear of the pedal reported {clear["collisions"]}'

    low = grade([PEDAL['x'], 0, 0.030])           # deliberately through the pedal
    assert (not low.get('above_floor', True)) or low.get('collisions') \
        or not low.get('reachable'), 'a too-low pose must be flagged somehow'
    print('scene self-checks passed')
