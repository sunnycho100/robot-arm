#!/usr/bin/env python3
"""Live view of the ampknobs detector, for aiming the camera by hand.

    python3 ampaim.py [port] [want]      then open http://<pi>:<port>

Every knob it finds is outlined with the ellipse it was fitted, labelled with
the two numbers that decided it, and its pointer drawn. What you see is exactly
what a run would see, because it is the same find() call.

Detection runs every EVERY_N frames and the overlay is redrawn from cache on
the ones between. Measured on this Pi, full detection is about 90 ms per frame
at 1280x960, so detecting on all of them left the stream at roughly 3 fps and
made aiming feel broken. Nothing here moves faster than a hand moves a camera.
"""
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ampknobs

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
WANT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
EVERY_N = 3

# Freeze the auto white balance. The C270 re-meters when the arm swings into
# frame and everything goes yellow; the detector's Otsu threshold rides that
# out, but the picture is much easier for a human to judge if it holds still.
try:
    import subprocess
    for c in ('white_balance_automatic=0', 'white_balance_temperature=4000'):
        subprocess.run(['v4l2-ctl', '-d', '/dev/video0', '--set-ctrl', c],
                       check=False, capture_output=True, timeout=3)
except Exception:
    pass

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)     # never the default backend: it deadlocks here
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
LOCK = threading.Lock()
CACHE = {'n': 0, 'knobs': []}

PAGE = b"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{background:#111;color:#eee;font:14px system-ui;margin:0;text-align:center}
img{max-width:100%;height:auto}h1{font-size:16px;margin:8px}</style>
<h1>amp knob finder, live &mdash; aim until the banner is green</h1>
<img src="/stream">"""


def annotate(frame):
    CACHE['n'] += 1
    if CACHE['n'] % EVERY_N == 1 or not CACHE['knobs']:
        CACHE['knobs'] = ampknobs.find(frame)
    ks = CACHE['knobs']

    for i, k in enumerate(ks, 1):
        c = (int(k['cx']), int(k['cy']))
        good = k.get('pointer_contrast', 0) >= 2.0
        col = (0, 235, 0) if good else (0, 165, 255)
        cv2.ellipse(frame, c, (int(k['major'] / 2), int(k['minor'] / 2)),
                    k['angle_deg'], 0, 360, col, 2)
        seg = ampknobs.pointer_segment(k) if good else None
        if seg:
            cv2.line(frame, seg[0], seg[1], (255, 60, 220), 2)
        # The two numbers that decided it, so a near miss is visible as a near
        # miss rather than as a knob that simply is not there.
        cv2.putText(frame, f"{i} f{k['fill']:.2f} c{k['convexity']:.2f}",
                    (c[0] - 55, c[1] - int(k['minor'] / 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    h, w = frame.shape[:2]
    ok = len(ks) == WANT
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 60, 20) if ok else (20, 20, 70), -1)
    msg = f'{len(ks)}/{WANT} knobs'
    if ks:
        sizes = [k['major'] for k in ks]
        msg += f'   {min(sizes):.0f}-{max(sizes):.0f} px across'
    cv2.putText(frame, msg, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (120, 255, 120) if ok else (120, 160, 255), 2, cv2.LINE_AA)

    hint = ('all of them, and the ellipses sit on the caps' if ok else
            'too few: are any cut off by the frame edge, or covered by the arm? '
            'a partly hidden knob is dropped on purpose')
    cv2.putText(frame, hint, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
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
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
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
        raise SystemExit('cannot open /dev/video0. Is something else holding it? '
                         'Free it with:  fuser -k /dev/video0')
    print(f'streaming on {PORT}, expecting {WANT} knobs', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
