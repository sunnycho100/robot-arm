#!/usr/bin/env python3
"""What ArUco tags are actually on the bench, and in which dictionary.

    python3 tagid.py [frames] [--shot=/tmp/tags.jpg]

Sweeps EVERY predefined ArUco dictionary, not just the one the demo expects.
A tag prints as an id only in the dictionary it was generated from, so this is
the only way to answer "which dictionary is this tag" without being told. Tags
get reprinted and swapped between sessions; guessing DICT_6X6_250 and finding
nothing looks identical to a camera fault.

Reports per tag: dictionary, id, edge length in pixels, and the distance that
implies for a few plausible printed sizes. Only ids seen in most frames are
kept, for the same reason the knob finder votes: a single frame will happily
decode noise as a marker in a large dictionary.
"""
import sys
import pathlib

import cv2
import numpy as np
from cv2 import aruco

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compat import intrinsics, NEW_API

# Every predefined dictionary this OpenCV knows, smallest first. The small
# ones are checked first on purpose: a 4x4 tag also decodes as a 5x5 in some
# builds, and the first hit is the more likely truth.
NAMES = [n for n in dir(aruco) if n.startswith('DICT_')]
NAMES.sort(key=lambda n: (int(n.split('_')[1][0]) if n.split('_')[1][0].isdigit() else 9, n))

CANDIDATE_MM = (30, 40, 50, 61)     # printed edge lengths worth reporting


def _detector(name):
    d = (aruco.getPredefinedDictionary(getattr(aruco, name)) if NEW_API
         else aruco.Dictionary_get(getattr(aruco, name)))
    p = aruco.DetectorParameters() if NEW_API else aruco.DetectorParameters_create()
    return d, p


def scan(frame):
    """-> {(dict_name, id): (edge_px, cx, cy)} for everything in this frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hits = {}
    for name in NAMES:
        d, p = _detector(name)
        if NEW_API:
            corners, ids, _ = aruco.ArucoDetector(d, p).detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, d, parameters=p)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            q = c.reshape(4, 2)
            edge = float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)]))
            hits[(name, int(i))] = (edge, float(q[:, 0].mean()), float(q[:, 1].mean()))
    return hits


def main(frames=9, shot=None, index=0):
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)   # never the default backend: it deadlocks here
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    if not cam.isOpened():
        raise SystemExit('cannot open the camera')
    votes, last, frame = {}, {}, None
    for _ in range(frames):
        ok, frame = cam.read()
        if not ok:
            continue
        for k, v in scan(frame).items():
            votes[k] = votes.get(k, 0) + 1
            last[k] = v
    cam.release()

    fx = intrinsics(frame.shape)[0][0, 0] if frame is not None else 0.0
    keep = sorted((k for k, n in votes.items() if n >= max(2, frames // 2)),
                  key=lambda k: -last[k][0])
    if not keep:
        print(f'NO TAGS in {frames} frames, across {len(NAMES)} dictionaries.')
        print('The camera sees no marker at all: check lighting, focus and framing.')
    for name, i in keep:
        edge, cx, cy = last[(name, i)]
        dists = '  '.join(f'{mm}mm->{fx * mm / edge:.0f}mm' for mm in CANDIDATE_MM)
        print(f'{name:<14} id={i:<4} seen {votes[(name, i)]}/{frames}  '
              f'edge {edge:.1f}px  at ({cx:.0f},{cy:.0f})')
        print(f'{"":14} if printed  {dists}')

    if shot and frame is not None:
        for name, i in keep:
            _, cx, cy = last[(name, i)]
            cv2.circle(frame, (int(cx), int(cy)), 40, (0, 235, 0), 3)
            cv2.putText(frame, f'{name.replace("DICT_", "")} id{i}',
                        (int(cx) - 90, int(cy) - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 235, 0), 2, cv2.LINE_AA)
        cv2.imwrite(shot, frame)
        print(f'wrote {shot}')
    return keep


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    shot = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--shot=')), None)
    main(int(args[0]) if args else 9, shot)
