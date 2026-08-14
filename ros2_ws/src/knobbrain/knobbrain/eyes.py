#!/usr/bin/env python3
"""The knob camera: grabbing, detecting, and serving the live view.

One thread owns the camera and nothing else touches it. The TUI asks for a
reading and gets the last consensus; the browser asks for a frame and gets the
last annotated one. Two readers, one writer, no locking beyond a swap of a
reference, because a stale frame is never worse than a blocked one.

Consensus rather than a single frame is the point: a knob that appears in three
frames out of nine is a reflection, and this rig has plenty of those.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

try:
    from knobbrain import ampknobs
except ImportError:
    import ampknobs

FRAMES = 9          # frames per consensus read
EVERY_N = 3         # detect on one frame in three; the rest reuse the overlay


def names_of(knobs, names):
    """Left to right, but only when every one of them is there.

    With one missing, every name past the gap belongs to its neighbour. Placing
    them on an even grid does not rescue it either: this panel is not evenly
    spaced, the drive-to-gain gap being 175 px against a 110 px pitch.
    """
    if len(knobs) != len(names):
        return [str(i) for i in range(1, len(knobs) + 1)]
    return list(names)


class Eyes:
    def __init__(self, device, names, port=8080):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)   # never the default
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # backend: it deadlocks
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        if not self.cap.isOpened():
            raise SystemExit(f'cannot open {device}. Free it with: '
                             f'fuser -k {device}')
        self.names = list(names)
        self.want = len(self.names)
        self.latest = None          # last annotated jpeg
        self.knobs = []             # last single-frame detection
        self.note = ''              # what the TUI shows on the cam K line
        self.stop = False
        threading.Thread(target=self._loop, daemon=True).start()
        self.srv = ThreadingHTTPServer(('0.0.0.0', port), self._handler())
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def _loop(self):
        n = 0
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            n += 1
            if n % EVERY_N == 1:
                self.knobs = ampknobs.find(frame)
                self.note = f'{len(self.knobs)}/{self.want} knobs'
            self.latest = cv2.imencode('.jpg', self._draw(frame),
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def _draw(self, frame):
        ks = self.knobs
        labels = names_of(ks, self.names)
        for i, k in enumerate(ks):
            c = (int(k['cx']), int(k['cy']))
            good = k.get('pointer_contrast', 0) >= 2.0
            col = (0, 235, 0) if good else (0, 165, 255)
            cv2.ellipse(frame, c, (int(k['major'] / 2), int(k['minor'] / 2)),
                        k['angle_deg'], 0, 360, col, 2)
            seg = ampknobs.pointer_segment(k) if good else None
            if seg:
                cv2.line(frame, seg[0], seg[1], (255, 60, 220), 2)
            cv2.putText(frame, labels[i], (c[0] - 55,
                        c[1] - int(k['minor'] / 2) - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        ok = len(ks) == self.want
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34),
                      (20, 60, 20) if ok else (20, 20, 70), -1)
        cv2.putText(frame, f'{len(ks)}/{self.want} knobs', (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 255, 120) if ok else (120, 160, 255), 2, cv2.LINE_AA)
        return frame

    def read(self):
        """{name: pointer_rel} over FRAMES frames, only the knobs that hold still.

        Returns {} rather than a guess when the row is not all there, because a
        name assigned to the wrong knob is how the arm grips the neighbour.
        """
        frames = []
        while len(frames) < FRAMES:
            ok, f = self.cap.read()
            if ok:
                frames.append(f)
            else:
                time.sleep(0.05)
        ks = ampknobs.find_stable(frames)       # already sorted left to right
        if len(ks) != self.want:
            self.note = f'{len(ks)}/{self.want} knobs, cannot name them'
            return {}
        self.note = f'{len(ks)}/{self.want} knobs'
        # find_stable replaces 'pointer' with the circular median across frames
        # but leaves 'pointer_rel' at whatever the middle frame happened to say,
        # so take the row off the median here rather than trusting the stale one.
        row = ampknobs.row_angle(ks)
        return {n: (k['pointer'] - row) % 360.0 for n, k in zip(self.names, ks)}

    def _handler(eyes_self):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(
                        b'<body style="background:#111;text-align:center">'
                        b'<img src="/stream" style="max-width:100%">')
                    return
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=f')
                self.end_headers()
                try:
                    while True:
                        jpg = eyes_self.latest
                        if jpg is None:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(
                            b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                            + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        return H
