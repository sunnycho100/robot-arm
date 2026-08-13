#!/usr/bin/env python3
"""Live camera with the REAL knob finder drawn on it, for aiming by hand.

    python3 aimlive.py [port]        # then open http://<pi-ip>:8080

Every frame is run through the same find_knobs and measure that the demo uses,
so what you see is what the run will see. Move the camera until the banner is
green and stays green.

Why not camstream.py: that one reads frames in a background thread while the
HTTP server runs in others, and on this Pi it deadlocks, because pipewire and
wireplumber also hold /dev/video0 and the contention lands inside the capture.
Here the camera is read inside the streaming handler under a single lock, so
there is exactly one reader at a time and nothing to deadlock against.
"""
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, '/home/pi/knobcheck')
sys.path.insert(0, '/home/pi/cv')
import knob
try:
    import compat                       # ArUco, whichever OpenCV this is
except Exception:
    compat = None

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
WANT = 3                       # knobs on a DS-1
WINDOW = 12                    # frames the rolling consensus looks back over

# A rolling record of what has been detected recently. A single frame cannot
# tell a knob from a good impostor: measured on this bench, the three real caps
# appeared in 30 of 30 frames and an impostor on the robot's own furniture in
# 3, and in any one of those 3 frames it looked perfect. Persistence is the
# only thing that separates them, so show BOTH numbers while aiming: what this
# frame thinks, and what has held still.
RECENT = []

# CAP_V4L2 explicitly. OpenCV on this Pi picks GStreamer by default, which
# goes through pipewire, and holding that open DEADLOCKS: five threads stuck in
# futex_wait, no frames, no error. Measured: default backend hangs before the
# first read, V4L2 delivers 20/20 frames at 1280x960 while held open.
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
LOCK = threading.Lock()

PAGE = b"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{background:#111;color:#eee;font:14px system-ui;margin:0;text-align:center}
img{max-width:100%;height:auto}h1{font-size:16px;margin:8px}</style>
<h1>knob finder, live &mdash; move the camera until the banner stays green</h1>
<img src="/stream">"""


# Detection is the whole cost of this stream and it does not need to run on
# every frame. Measured on the Pi at 1280x960: find_knobs 100 ms, ArUco
# another 60, jpeg encode 22, which is why the view crawled at about 3 fps and
# felt broken while aiming. Nothing being looked for moves faster than a hand
# moves a camera, so detect every EVERY_N frames and redraw the cached result
# on the ones between. The picture stays live, the overlay lags a third of a
# second, and that is invisible while pointing a camera.
EVERY_N = 4
_cache = {'n': 0, 'ks': {}, 'found': {}, 'rejects': [], 'tags': ([], [])}


def annotate(frame):
    """Draw what the finder sees, and grade it, exactly as calibrate would."""
    _cache['n'] += 1
    if _cache['n'] % EVERY_N == 1 or not _cache['ks']:
        rejects = []
        _cache['ks'] = knob.find_knobs(frame, rejects)
        _cache['found'] = knob.measure(frame, _cache['ks']) if _cache['ks'] else {}
        _cache['rejects'] = rejects
        _cache['tags'] = ((compat.detect(frame, '4x4_50') if compat else ([], []))
                          or ([], []))
        fresh = True
    else:
        fresh = False
    rejects = _cache['rejects']
    ks, found = _cache['ks'], _cache['found']
    for name, f in found.items():
        good = f['contrast'] >= knob.MIN_CONTRAST
        col = (0, 235, 0) if good else (0, 165, 255)
        cv2.circle(frame, (f['cx'], f['cy']), f['cap_r'], col, 3)
        cv2.circle(frame, (f['cx'], f['cy']), f['skirt_r'], (0, 170, 255), 1)
        a = np.radians(f['angle'])
        cv2.line(frame, (f['cx'], f['cy']),
                 (int(f['cx'] + f['skirt_r'] * np.cos(a)),
                  int(f['cy'] + f['skirt_r'] * np.sin(a))), (255, 60, 220), 2)
        dist = knob.FX * knob.CAP_MM / max(2 * f['cap_r'], 1)
        cv2.putText(frame, f'{name} {2*f["cap_r"]}px {dist:.0f}mm r{f["roundness"]:.2f}',
                    (f['cx'] - 90, f['cy'] - f['skirt_r'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    if fresh:
        RECENT.append([(v['cx'], v['cy']) for v in ks.values()])
        del RECENT[:-WINDOW]
    votes = {}
    for frame_hits in RECENT:
        for c in frame_hits:
            for k in votes:
                if abs(k[0] - c[0]) < 25 and abs(k[1] - c[1]) < 25:
                    votes[k] += 1
                    break
            else:
                votes[c] = 1
    # Until the window has filled, EVERY detection looks like a flicker,
    # because nothing has had the chance to appear twice yet. Saying so beats
    # showing a scary 0/3 for the first second and training you to ignore it.
    warming = len(RECENT) < 4
    need = max(2, int(round(0.7 * len(RECENT))))
    stable = sum(1 for v in votes.values() if v >= need)
    flick = sum(1 for v in votes.values() if v < need)

    n = len(ks)
    ok = (n == WANT) if warming else (stable == WANT and flick == 0)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 62), (20, 60, 20) if ok else (20, 20, 70), -1)
    caps = [2 * v['cap_r'] for v in ks.values()]
    msg = (f'{n}/{WANT} knobs (warming up)' if warming else
           f'STABLE {stable}/{WANT}   (this frame {n})')
    if flick and not warming:
        msg += f'  +{flick} flickering'
    if caps:
        msg += f'   caps {min(caps)}-{max(caps)} px   ' \
               f'{knob.FX*knob.CAP_MM/max(caps):.0f}-{knob.FX*knob.CAP_MM/min(caps):.0f} mm'
    cv2.putText(frame, msg, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (120, 255, 120) if ok else (120, 160, 255), 2, cv2.LINE_AA)
    # When it is NOT finding them, say what it threw away. That is the whole
    # difference between "move the camera somehow" and knowing which way.
    hint = ''
    if warming:
        hint = 'filling the window...'
    elif ok:
        hint = (f'held still over {len(RECENT)} frames: calibrate keeps these')
    elif flick:
        hint = (f'{flick} detection(s) coming and going. calibrate drops them, '
                f'but move the camera if it persists')
    elif rejects:
        big = sum(1 for a, why in rejects if 'TOO CLOSE' in why)
        hint = ('too CLOSE: caps too big, back off' if big else
                f'nearest miss: {sorted(rejects, reverse=True)[0][1][:58]}')
    cv2.putText(frame, hint, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    # The tags matter as much as the knobs while aiming, and they fail in a
    # different way: a partly covered marker still decodes, just as the wrong
    # id. Seeing the id live is the only way to catch that while moving the
    # camera rather than three runs later.
    if compat is not None:
        corners, ids = _cache['tags']
        for c, i in zip(corners, ids):
            q = c.astype(int)
            cv2.polylines(frame, [q.reshape(-1, 1, 2)], True, (255, 200, 0), 2)
            cv2.putText(frame, f'id {int(i)}', (q[:, 0].min(), q[:, 1].min() - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
        want = {12, 18}
        got = set(int(i) for i in ids)
        msg = (f'tags {sorted(got)}' if got else 'NO TAGS')
        if got and got != want:
            msg += f'   expected {sorted(want)}'
        cv2.putText(frame, msg, (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 255, 120) if got == want else (120, 160, 255),
                    2, cv2.LINE_AA)
    return frame


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(PAGE)
            return
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=f')
        self.end_headers()
        try:
            while True:
                with LOCK:
                    ok, frame = cam.read()
                if not ok:
                    time.sleep(0.2)
                    continue
                jpg = cv2.imencode('.jpg', annotate(frame),
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n'
                                 b'Content-Length: ' + str(len(jpg)).encode()
                                 + b'\r\n\r\n' + jpg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    if not cam.isOpened():
        raise SystemExit('cannot open /dev/video0')
    print(f'streaming on {PORT}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
