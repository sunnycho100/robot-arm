#!/usr/bin/env python3
"""What the tags say about ORIENTATION, which is the part that needs no ruler.

    python3 tagpose.py [frames]

estimatePoseSingleMarkers returns a rotation and a translation. The translation
scales linearly with the printed tag size, so it is only as good as the number
someone typed in. The ROTATION does not: a tag twice as large at twice the
distance projects identically, so the rotation is recovered from the corner
geometry alone and is trustworthy before anything has been measured.

That is worth having on its own. The pedal tag lies flat on the pedal, so its
rotation IS the pedal's:

  tilt      how far the pedal face is from square to the camera. This is the
            number that says whether the knob caps are being seen at an angle,
            which is what smears the pointer reading.
  spin      which way the pedal is turned in its own plane. The knob row runs
            with it, so this is what says whether "up and left" means the
            neighbouring knob or open air. The gripper has to sweep away from
            the neighbour, and this is where that direction comes from.

Relative rotation between the base tag and the pedal tag is the same quantity
in robot coordinates, so it survives the camera being moved, which the fixed
image-frame angles do not.

Sizes are read from see.TAG_MM. They are estimates until someone puts a ruler
on the tags, and everything printed under DISTANCE below is wrong by whatever
factor those estimates are wrong by. Everything under ORIENTATION is not.
"""
import sys
import pathlib

import numpy as np
import cv2
from cv2 import aruco

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts'))
from compat import detect, intrinsics

TAG_MM = {18: 35.0, 12: 15.0}      # user-confirmed from the print PDF
WHERE = {18: 'pedal', 12: 'robot base'}
DICT = '4x4_50'


def _euler(rvec):
    """-> (tilt, spin) degrees. tilt is off the camera axis, spin is in-plane."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))
    # The tag's own z axis points out of its face. The angle between that and
    # the camera's z axis is how far the face is from square to the lens.
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
    # The tag's x axis projected into the image is the in-plane direction.
    spin = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(tilt), float(spin)


def read(frames=9, index=0):
    """-> {id: {'tilt', 'spin', 'xyz_mm', 'seen'}} averaged over frames."""
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    K = dist = None
    hits = {}
    for i in range(frames + 6):
        ok, frame = cam.read()
        if not ok or i < 6:
            continue
        if K is None:
            K, dist = intrinsics(frame.shape)
        corners, ids = detect(frame, DICT)
        for c, tid in zip(corners, ids):
            size = TAG_MM.get(int(tid), 20.0) / 1000.0
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                c.reshape(1, 4, 2).astype(np.float32), size, K, dist)
            tilt, spin = _euler(rvec[0][0])
            hits.setdefault(int(tid), []).append(
                [tilt, spin] + [v * 1000 for v in tvec[0][0]])
    cam.release()
    return {t: {'tilt': float(np.mean([o[0] for o in obs])),
                'spin': float(np.degrees(np.angle(np.mean(
                    [np.exp(1j * np.radians(o[1])) for o in obs])))),
                'xyz_mm': [float(np.mean([o[k] for o in obs])) for k in (2, 3, 4)],
                'seen': len(obs)}
            for t, obs in hits.items()}


def main():
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    tags = read(frames)
    if not tags:
        raise SystemExit('no tags in view. Run cv/tagid.py to see what is there.')

    print('ORIENTATION (no ruler needed, this is scale free)')
    for t, v in sorted(tags.items()):
        print(f'  id {t:<3} {WHERE.get(t, "?"):<11} seen {v["seen"]}/{frames}   '
              f'tilt {v["tilt"]:5.1f} deg off square   '
              f'spin {v["spin"]:+7.1f} deg in plane')
    if 18 in tags and 12 in tags:
        d = (tags[18]['spin'] - tags[12]['spin'] + 180) % 360 - 180
        print(f'  pedal is turned {d:+.1f} deg relative to the robot base.')
        print('  This one survives the camera moving. The image-frame knob '
              'angles do not.')

    print('\nDISTANCE (only as good as the assumed tag sizes below)')
    for t, v in sorted(tags.items()):
        x, y, z = v['xyz_mm']
        print(f'  id {t:<3} assuming {TAG_MM.get(t, 20.0):.0f} mm printed: '
              f'x={x:+7.1f} y={y:+7.1f} z={z:+7.1f} mm')
    print('  Measure the printed edges and correct see.TAG_MM. Distance scales '
          'straight off them, orientation above does not.')


if __name__ == '__main__':
    main()
