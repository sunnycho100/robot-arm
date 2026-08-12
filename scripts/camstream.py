#!/usr/bin/env python3
"""Live camera view in a browser, for aiming the camera by hand.

  python3 ~/camstream.py [port]

Then open http://<pi-ip>:8080 on the laptop.

The green box is the middle of the frame: put the pedal inside it. The number is
the sharpness of whatever is in that box, which is what tells you whether the
camera is too close, because the C270 is fixed focus and cannot tell you itself.
Higher is sharper. Move it up and down and watch the number rather than guessing.

It holds the camera open, and only one process can. Stop it before running
see.py or turn_knob.py:

  pkill -f "[c]amstream"
"""
import sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
BOX = 300                       # half-width of the centre region we score

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
lock = threading.Lock()

PAGE = b'''<html><head><title>xArm camera</title></head>
<body style="margin:0;background:#111;text-align:center">
<img src="/stream" style="max-width:100vw;max-height:100vh">
</body></html>'''


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
                with lock:
                    ok, frame = cam.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                centre = cv2.cvtColor(frame[cy - BOX:cy + BOX, cx - BOX:cx + BOX],
                                      cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(centre, cv2.CV_64F).var()
                cv2.rectangle(frame, (cx - BOX, cy - BOX), (cx + BOX, cy + BOX),
                              (0, 255, 0), 2)
                cv2.putText(frame, f'sharpness {sharp:.0f}', (cx - BOX, cy - BOX - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
                ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n'
                                     + jpg.tobytes() + b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    print(f'streaming on port {PORT}. Ctrl-C to stop.')
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
