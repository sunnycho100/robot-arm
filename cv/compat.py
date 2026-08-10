#!/usr/bin/env python3
"""One ArUco API for every OpenCV we run on.

The Pi has OpenCV 4.6 (Dictionary_get / DetectorParameters_create /
aruco.detectMarkers as a free function). Development Macs have 4.7+ or 5.x,
where those were removed in favour of getPredefinedDictionary /
DetectorParameters() / ArucoDetector. Everything else in cv/ imports from
here so no other file has to care.

Also carries the camera intrinsics: the course calibration pickles that ship
in this folder (Python 2 format, hence encoding='bytes').

    from compat import detect, draw_marker, intrinsics
    corners, ids = detect(frame, '4x4_50')
"""
import pathlib
import numpy as np
import cv2
from cv2 import aruco

HERE = pathlib.Path(__file__).parent
NEW_API = hasattr(aruco, 'ArucoDetector')

DICTS = {'4x4_50': aruco.DICT_4X4_50, '5x5_50': aruco.DICT_5X5_50,
         '6x6_250': aruco.DICT_6X6_250, '7x7_50': aruco.DICT_7X7_50}

# calibrated at 1280x960; scale with intrinsics(shape) for other sizes
CALIB_SIZE = (1280, 960)
_cache = {}


def _dictionary(name):
    if name not in _cache:
        d = DICTS[name]
        _cache[name] = (aruco.getPredefinedDictionary(d) if NEW_API
                        else aruco.Dictionary_get(d))
    return _cache[name]


def _params():
    if 'params' not in _cache:
        p = aruco.DetectorParameters() if NEW_API else aruco.DetectorParameters_create()
        # subpixel corners: the servoing loop wants better than whole-pixel
        p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        _cache['params'] = p
    return _cache['params']


def detect(frame, dict_name='4x4_50'):
    """-> (corners, ids). corners is a list of (4,2) float arrays in pixels,
    ids is a flat int array. Both empty when nothing is found."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    d, p = _dictionary(dict_name), _params()
    if NEW_API:
        corners, ids, _ = aruco.ArucoDetector(d, p).detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, d, parameters=p)
    if ids is None or len(ids) == 0:
        return [], np.array([], dtype=int)
    return [c.reshape(4, 2).astype(float) for c in corners], ids.ravel().astype(int)


def detect_any(frame, dict_names=('4x4_50', '6x6_250')):
    """Try several dictionaries; -> (corners, ids, dict_name) of the first hit."""
    for name in dict_names:
        corners, ids = detect(frame, name)
        if len(ids):
            return corners, ids, name
    return [], np.array([], dtype=int), None


def draw_marker(dict_name, marker_id, px):
    """Render one marker as a px-by-px uint8 image."""
    d = _dictionary(dict_name)
    if hasattr(aruco, 'generateImageMarker'):
        return aruco.generateImageMarker(d, marker_id, px)
    return aruco.drawMarker(d, marker_id, px)


def intrinsics(shape=None):
    """-> (K, dist) for the course-calibrated camera, scaled to shape (h, w)."""
    if 'K' not in _cache:
        import pickle
        _cache['K'] = np.array(pickle.load(open(HERE / 'cam_matrix.p', 'rb'),
                                           encoding='bytes'), dtype=float)
        _cache['d'] = np.array(pickle.load(open(HERE / 'dist_matrix.p', 'rb'),
                                           encoding='bytes'), dtype=float).ravel()
    K, dist = _cache['K'].copy(), _cache['d']
    if shape is not None:
        h, w = shape[:2]
        K[0] *= w / CALIB_SIZE[0]
        K[1] *= h / CALIB_SIZE[1]
    return K, dist


if __name__ == '__main__':
    print(f'OpenCV {cv2.__version__}, {"new" if NEW_API else "legacy"} aruco API')

    # round-trip: render a marker, pad a quiet zone, detect it back
    for name in ('4x4_50', '6x6_250'):
        img = np.full((400, 400), 255, np.uint8)
        img[60:340, 60:340] = draw_marker(name, 7, 280)
        corners, ids = detect(img, name)
        assert list(ids) == [7], f'{name}: expected id 7, got {list(ids)}'
        err = np.abs(np.sort(corners[0][:, 0])[[0, -1]] - [60, 339]).max()
        assert err < 2, f'{name}: corner off by {err:.1f} px'
        print(f'  {name}: detected id 7, corners within {err:.1f} px')

    corners, ids, which = detect_any(np.full((400, 400), 255, np.uint8))
    assert which is None and len(ids) == 0, 'blank image must find nothing'

    K, dist = intrinsics()
    assert 1400 < K[0, 0] < 1500, f'unexpected fx {K[0, 0]}'
    K2, _ = intrinsics((480, 640))
    assert abs(K2[0, 0] - K[0, 0] / 2) < 1, 'intrinsics did not scale'
    print(f'  intrinsics fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
          f'c=({K[0,2]:.1f},{K[1,2]:.1f}), scaling OK')
    print('compat self-checks passed')
