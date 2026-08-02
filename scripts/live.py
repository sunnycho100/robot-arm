#!/usr/bin/env python3
"""Serve whatever the running script is currently looking at, as MJPEG.

    import live
    live.serve(lambda: my_annotated_frame)      # then open http://<pi>:8080

Only one process can hold the camera, so a separate viewer cannot run alongside a
run that is using it. Instead the run publishes its own view: the frames served
are the exact frames the code made its decisions from, with the detections drawn
on. If the overlay looks wrong, the decision was wrong, which is the point.

Runs in a daemon thread, so it never keeps the script alive.
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

PAGE = b'''<html><head><title>xArm live</title></head>
<body style="margin:0;background:#111;text-align:center">
<img src="/stream" style="max-width:100vw;max-height:100vh">
</body></html>'''


def serve(get_frame, port=8080, fps=8.0):
    """Start the viewer. get_frame() must return a BGR frame or None."""

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
                import time
                while True:
                    f = get_frame()
                    if f is not None:
                        ok, jpg = cv2.imencode('.jpg', f,
                                               [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n'
                                             + jpg.tobytes() + b'\r\n')
                    time.sleep(1.0 / fps)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f'live view on http://<pi-ip>:{port}')
    return srv
