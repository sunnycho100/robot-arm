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
)
Z_FLOOR = 0.045     # MEASURE: from ~/z_floor.json on the Pi


def knob_targets():
    """Grip target (m) for each knob: centre of the cap, in the arm frame."""
    z = PEDAL['h'] + PEDAL['knob_h']
    return {f'knob{i}': np.array([PEDAL['x'], (i - 1) * PEDAL['knob_dy'], z])
            for i in range(3)}


def build():
    spec = mujoco.MjSpec.from_file(str(HERE / 'xarm_1s_mj.urdf'))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    w = spec.worldbody
    w.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[1, 1, 0.1], rgba=[.55, .5, .45, 1])
    ped = w.add_body(name='pedal', pos=[PEDAL['x'], PEDAL['y'], PEDAL['h'] / 2])
    ped.add_geom(name='pedal_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[PEDAL['d'] / 2, PEDAL['w'] / 2, PEDAL['h'] / 2],
                 rgba=[.95, .45, .1, 1])                      # DS-1 orange
    for name, t in knob_targets().items():
        k = w.add_body(name=name, pos=[t[0], t[1], PEDAL['h']])
        k.add_geom(name=f'{name}_body', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[PEDAL['knob_r'], PEDAL['knob_h'] / 2],
                   pos=[0, 0, PEDAL['knob_h'] / 2], rgba=[.1, .1, .1, 1])
        k.add_geom(name=f'{name}_cap', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[PEDAL['cap_r'], 0.001],
                   pos=[0, 0, PEDAL['knob_h'] + 0.001], rgba=[.8, .8, .82, 1])
    for g in w.add_light(pos=[0.3, -0.3, 0.8], dir=[-0.3, 0.3, -0.8]),:
        pass
    return spec.compile()


MODEL = build()
SCENE_BODIES = {'pedal', 'knob0', 'knob1', 'knob2'}


def _pose(data, xyz, gripper_angle=None):
    counts, achieved, clipped = ik.solve(xyz, gripper_angle)
    ang = ik.limit_joint_angles(ik.invkin(xyz, gripper_angle))
    for name, q in urdfmap.qpos_from(ang).items():
        data.qpos[MODEL.joint(name).qposadr[0]] = q
    mujoco.mj_forward(MODEL, data)
    return counts, achieved, clipped


def grade(xyz, gripper_angle=None, target_knob=None):
    """Judge a candidate gripper target. Returns a dict; see module docstring."""
    data = mujoco.MjData(MODEL)
    try:
        counts, achieved, clipped = _pose(data, xyz, gripper_angle)
    except ValueError as e:
        return dict(reachable=False, why=str(e))
    hits = []
    for c in data.contact:
        b1 = MODEL.body(MODEL.geom_bodyid[c.geom1]).name
        b2 = MODEL.body(MODEL.geom_bodyid[c.geom2]).name
        scene = {b1, b2} & SCENE_BODIES
        arm = {b1, b2} - SCENE_BODIES - {'world'}
        if scene and arm and not (target_knob and scene == {target_knob}):
            hits.append(f'{arm.pop()} vs {scene.pop()}')
        g1 = MODEL.geom(c.geom1).name or b1
        g2 = MODEL.geom(c.geom2).name or b2
        if 'table' in (g1, g2) and not scene:
            hits.append(f'{b1 if g2 == "table" else b2} vs table')
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
    low = grade([PEDAL['x'], 0, 0.030])           # deliberately through the pedal
    assert (not low.get('above_floor', True)) or low.get('collisions') \
        or not low.get('reachable'), 'a too-low pose must be flagged somehow'
    print('scene self-checks passed')
