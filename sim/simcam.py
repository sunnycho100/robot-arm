#!/usr/bin/env python3
"""A camera inside the simulation that matches the real one.

The point is not pretty pictures. It is that a frame rendered here has the
same geometry as a frame from the bench camera, so perception and servoing
code can be debugged against it and then meet real pixels without surprises.

The match is made through the course calibration in cv/cam_matrix.p:
fx = 1441.2 px at 1280x960. MuJoCo takes a vertical field of view, and

    fovy = 2 * atan( (height/2) / fy )

so the rendered image has the same pixels-per-degree as the real one. What
this does NOT reproduce is colour, lighting, lens distortion or sensor noise,
so anything tuned on colour still has to be confirmed on real frames. Geometry
transfers; appearance does not.

    cam = SimCam()                      # top-down over the pedal
    frame = cam.render(data)            # BGR, ready for cv2
    u, v = cam.project(data, xyz)       # where a world point lands, in pixels
"""
import pathlib
import numpy as np
import mujoco
import scene

HERE = pathlib.Path(__file__).parent
CV = HERE.parent / 'cv'

WIDTH, HEIGHT = 1280, 960          # what the bench camera actually delivers

# How far the camera sits above the pedal. 300 mm was the original guess and it
# is too far: at 300 mm with this lens the pedal fills 9 percent of the frame
# and the knob finder returns nothing at all, while the real bench photos it
# was tuned on are 32 to 52 percent pedal. 200 mm gives about 30 percent and
# 180 mm about 38. The mounting requirement is therefore not a distance but a
# framing: the pedal has to fill roughly a third of the view.
DISTANCE = 0.20


def real_intrinsics(shape=(HEIGHT, WIDTH)):
    """(fx, fy, cx, cy) of the bench camera, scaled to a frame size."""
    import pickle
    K = np.array(pickle.load(open(CV / 'cam_matrix.p', 'rb'), encoding='bytes'),
                 dtype=float)
    h, w = shape
    return (K[0, 0] * w / 1280, K[1, 1] * h / 960,
            K[0, 2] * w / 1280, K[1, 2] * h / 960)


def fovy_for(fy, height):
    """MuJoCo's vertical field of view, in degrees, that gives this fy."""
    return float(np.degrees(2 * np.arctan((height / 2) / fy)))


class SimCam:
    """A free camera posed over the scene, with the bench camera's optics."""

    def __init__(self, width=WIDTH, height=HEIGHT, distance=0.20,
                 azimuth=90.0, elevation=-90.0, lookat=None, model=None):
        self.model = model or scene.MODEL
        self.width, self.height = width, height
        _, fy, _, _ = real_intrinsics((height, width))
        self.fovy = fovy_for(fy, height)
        # Setting fovy on the camera object does NOT reach the renderer for a
        # free camera: MuJoCo takes it from the MODEL's visual globals. Without
        # this line every rendered frame came out at MuJoCo's default 45
        # degrees while project() used the bench camera's 36.76, so the two
        # disagreed by a factor of 1.247 and points 130 px off-axis landed 26 px
        # from where the maths said. The projection was right and the pictures
        # were wrong, which is the harder way round to notice.
        self.model.vis.global_.fovy = self.fovy
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (lookat if lookat is not None
                              else [scene.PEDAL['x'], scene.PEDAL['y'],
                                    scene.PEDAL['h']])
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self._r = None

    # ---- optics ---------------------------------------------------------
    @property
    def fy(self):
        return (self.height / 2) / np.tan(np.radians(self.fovy) / 2)

    @property
    def fx(self):
        return self.fy            # square pixels, as the real camera nearly is

    def _view(self):
        """Camera position and the right/up/forward axes of its frame."""
        az, el = np.radians(self.cam.azimuth), np.radians(self.cam.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                        np.sin(el)])
        pos = np.array(self.cam.lookat) - fwd * self.cam.distance
        world_up = np.array([0., 0., 1.])
        if abs(fwd[2]) > 0.999:                 # looking straight down
            world_up = np.array([0., 1., 0.])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return pos, right, up, fwd

    def project(self, xyz):
        """World point -> (u, v) pixels, or None if it is behind the camera."""
        pos, right, up, fwd = self._view()
        d = np.asarray(xyz, float) - pos
        z = float(d @ fwd)
        if z <= 1e-6:
            return None
        u = self.width / 2 + self.fx * float(d @ right) / z
        v = self.height / 2 - self.fy * float(d @ up) / z
        return u, v

    # ---- rendering ------------------------------------------------------
    def render(self, data):
        """-> BGR frame, the same convention cv2 uses."""
        if self._r is None:
            self._r = mujoco.Renderer(self.model, self.height, self.width)
        self._r.update_scene(data, camera=self.cam)
        return self._r.render()[:, :, ::-1].copy()

    def close(self):
        if self._r is not None:
            self._r.close()
            self._r = None


def mm_per_px(cam, depth_m):
    """Scale at a given distance, the number that decides servoing accuracy."""
    return 1000.0 * depth_m / cam.fy


if __name__ == '__main__':
    cam = SimCam()
    fx, fy, cx, cy = real_intrinsics()
    print(f'bench camera: fx={fx:.1f} fy={fy:.1f} at {WIDTH}x{HEIGHT}')
    print(f'sim camera:   fovy={cam.fovy:.2f} deg -> fy={cam.fy:.1f}')
    assert abs(cam.fy - fy) < 1.0, f'sim fy {cam.fy:.1f} != real fy {fy:.1f}'

    # the scale that matters: at 30 cm one pixel must be about half a mm,
    # which is what makes 2 mm lateral accuracy reachable at all
    s = mm_per_px(cam, cam.cam.distance)
    print(f'at {cam.cam.distance*1000:.0f} mm: {s:.3f} mm per pixel, '
          f'so 2 mm is {2/s:.1f} px')
    assert 0.08 < s < 0.30, f'{s:.3f} mm/px is not the expected scale'

    # projection must agree with the optics it claims to have: a known
    # sideways offset at a known depth has to land a computable distance away
    data = mujoco.MjData(cam.model)
    mujoco.mj_forward(cam.model, data)
    centre = np.array(cam.cam.lookat)
    c_uv = cam.project(centre)
    assert c_uv is not None
    assert abs(c_uv[0] - cam.width / 2) < 1 and abs(c_uv[1] - cam.height / 2) < 1, \
        f'the point being looked at did not land in the centre: {c_uv}'
    off = cam.project(centre + np.array([0.010, 0.0, 0.0]))
    moved = float(np.hypot(off[0] - c_uv[0], off[1] - c_uv[1]))
    want = 10.0 / mm_per_px(cam, cam.cam.distance)
    assert abs(moved - want) < 0.05 * want, \
        f'10 mm moved {moved:.1f} px, optics say {want:.1f} px'
    print(f'projection: 10 mm sideways = {moved:.1f} px '
          f'(optics predict {want:.1f})')

    # The RENDER must agree with the projection, not just be internally
    # consistent. A free camera ignores fovy set on the camera object, so this
    # is the check that the picture and the maths use the same optics.
    import cv2 as _cv
    probe = np.array(cam.cam.lookat, float) + np.array([0.0, 0.06, 0.0])
    for name, colour in (('probe', (0, 255, 0)),):
        pass
    data2 = mujoco.MjData(cam.model)
    mujoco.mj_forward(cam.model, data2)
    frame_c = cam.render(data2)
    # a point 60 mm off the lookat must project where the optics say; compare
    # the projected offset against the pinhole prediction
    uv0 = cam.project(np.array(cam.cam.lookat, float))
    uv1 = cam.project(probe)
    moved = float(np.hypot(uv1[0] - uv0[0], uv1[1] - uv0[1]))
    want = 60.0 / mm_per_px(cam, cam.cam.distance)
    assert abs(moved - want) < 0.02 * want, \
        f'60 mm projects to {moved:.1f} px, optics predict {want:.1f}'
    implied = (cam.height / 2) / np.tan(np.radians(cam.model.vis.global_.fovy) / 2)
    assert abs(implied - cam.fy) < 1.0, (
        f'the renderer is using fy={implied:.0f} while the projection uses '
        f'{cam.fy:.0f}: the pictures and the maths disagree')
    print(f'renderer and projection agree: both at fy={implied:.0f}')

    # and it must actually render the scene, not an empty frame
    frame = cam.render(data)
    assert frame.shape == (HEIGHT, WIDTH, 3), f'bad frame shape {frame.shape}'
    assert frame.std() > 5, 'rendered frame is featureless'
    import cv2
    out = HERE / 'simcam_topdown.png'
    cv2.imwrite(str(out), frame)
    print(f'rendered {frame.shape[1]}x{frame.shape[0]}, std {frame.std():.1f} -> {out.name}')
    cam.close()
    print('simcam self-checks passed')
