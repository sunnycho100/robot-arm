#!/usr/bin/env python3
"""ArUco tag pose from the C270, headless.

  see.py [--size=0.020] [--frames=30] [--shot=/home/pi/tag.jpg]

Prints each tag's position in METERS in the camera frame: +x right, +y down,
+z out of the lens. Saves one annotated frame so the detection can be eyeballed.

Camera calibration comes from the course files, measured on a Logitech C270.
Also importable: look() returns the same data for arm.py to use.
"""
import sys, pickle
import numpy as np, cv2
from cv2 import aruco

CALIB = '/home/pi/cv'
TAG_SIZE = 0.020        # default tag edge, meters. Distance scales directly with it.

# The two tags actually taped to this bench, measured with cv/tagid.py: both
# decode in the 4x4 family, NOT the 6x6_250 this file used to assume. They are
# different sizes, and pose scales linearly with size, so one global TAG_SIZE
# would put one of them at the wrong distance by a factor of two.
TAG_MM = {18: 35.0,     # big tag, on the pedal (user-confirmed from the print PDF)
          12: 15.0}     # little tag, on the robot base plate

_cam_matrix = pickle.load(open(f'{CALIB}/cam_matrix.p', 'rb'), encoding='bytes')
_dist_matrix = pickle.load(open(f'{CALIB}/dist_matrix.p', 'rb'), encoding='bytes')
_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
_params = aruco.DetectorParameters_create()


def look(size=TAG_SIZE, frames=20, shot=None):
    """Average each visible tag's pose over several frames.

    Returns {id: {'xyz': [x,y,z] metres, 'rvec': [..], 'seen': n, 'of': frames}}.
    Empty dict means nothing detected, which is a normal outcome in bad light,
    not an error. Callers decide whether that is fatal.
    """
    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    for _ in range(5):          # let exposure settle, the first frames are dark
        cam.read()

    hits, frame = {}, None
    for _ in range(frames):
        ok, frame = cam.read()
        if not ok:
            continue
        corners, ids, _ = aruco.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), _dict, parameters=_params)
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            edge = TAG_MM.get(int(i), size * 1000.0) / 1000.0
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                c, edge, _cam_matrix, _dist_matrix)
            hits.setdefault(int(i), []).append(
                list(tvec[0][0]) + list(rvec[0][0]))
            if shot:
                cv2.drawFrameAxes(frame, _cam_matrix, _dist_matrix,
                                  rvec, tvec, edge / 2)
        if shot:
            aruco.drawDetectedMarkers(frame, corners, ids)
    cam.release()
    if shot and frame is not None:
        cv2.imwrite(shot, frame)

    out = {}
    for i, obs in hits.items():
        a = np.array(obs)
        out[i] = {'xyz': a[:, :3].mean(0).tolist(),
                  'rvec': a[:, 3:].mean(0).tolist(),
                  'noise': float(a[:, :3].std(0).max()),
                  'seen': len(obs), 'of': frames}
    return out


def snap(path):
    """Plain photo, no detection. For eyeballing what the arm just did."""
    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    for _ in range(8):
        ok, f = cam.read()
    cam.release()
    if ok:
        cv2.imwrite(path, f)
    return ok


def main():
    flag = lambda k, d: next((type(d)(a.split('=')[1])
                              for a in sys.argv[1:] if a.startswith(f'--{k}=')), d)
    size = flag('size', TAG_SIZE)
    shot = flag('shot', '/home/pi/images/tag.jpg')
    tags = look(size, flag('frames', 30), shot)

    if not tags:
        sys.exit('no tags found. Check lighting, focus, that the whole tag plus '
                 'its white border is inside the frame, and that it is a 4x4 tag (run cv/tagid.py to find out which).')
    print(f'tag side length assumed {size*1000:.0f} mm '
          f'(distance scales directly with this, so measure it)')
    for i, t in sorted(tags.items()):
        x, y, z = (v * 1000 for v in t['xyz'])
        print(f"  ID {i}: seen {t['seen']}/{t['of']} frames"
              f'   x={x:+7.1f}  y={y:+7.1f}  z={z:+7.1f} mm'
              f"   (noise +/- {t['noise']*1000:.1f} mm)")
    print(f'annotated frame -> {shot}')


if __name__ == '__main__':
    main()
