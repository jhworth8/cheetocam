#!/usr/bin/env python3
"""On-demand live view for the Cheeto cam.

Why this lives inside the detector process rather than being its own service:
/dev/video0 can only be opened by one process, and the detector holds it open
permanently. A separate streaming server would fight it for the device. The
detector already decodes every frame for YOLO, so this just re-serves those.

Cost when nobody is watching: one reference assignment per frame. JPEG encoding
only happens while a viewer is actually connected, and it happens on the serving
thread, never on the detection loop.

Two endpoints, deliberately:
  /stream.mjpg   real-time multipart MJPEG, lowest latency
  /snapshot.jpg  a single frame — the fallback, because some proxies buffer
                 streaming responses and a polling client works everywhere

Nothing here may ever raise into the detector. Every public function swallows
its own exceptions; a broken stream must not stop the cat being watched.
"""

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

_lock = threading.Lock()
_frame = None          # most recent BGR frame from the detector
_seq = 0               # bumped per frame so viewers can wait for a *new* one
_viewers = 0           # live MJPEG connections
_token = ''
_started = False

JPEG_QUALITY = 70
IDLE_SLEEP = 0.04
# A viewer that stalls should not hold a thread forever.
MAX_STREAM_SECONDS = 60 * 30


def publish(frame):
    """Hand the newest frame over. Called from the detector's capture loop.

    Stores a reference only — no copy, no encode — so this is effectively free
    and cannot slow detection down.
    """
    global _frame, _seq
    try:
        with _lock:
            _frame = frame
            _seq += 1
    except Exception:
        pass


def viewers():
    return _viewers


def _encode_latest():
    with _lock:
        f = _frame
        s = _seq
    if f is None:
        return None, s
    ok, buf = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None, s
    return buf.tobytes(), s


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        # The default handler logs every request to stderr, which would flood
        # journald with one line per frame.
        pass

    def _authorised(self):
        if not _token:
            return False
        q = parse_qs(urlparse(self.path).query)
        supplied = (q.get('token') or [''])[0] or self.headers.get('X-Stream-Token', '')
        return supplied == _token

    def _deny(self):
        body = b'forbidden'
        self.send_response(403)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global _viewers
        path = urlparse(self.path).path

        if path == '/health':
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
            return

        if not self._authorised():
            self._deny()
            return

        if path == '/snapshot.jpg':
            data, _ = _encode_latest()
            if data is None:
                self.send_response(503)
                self.send_header('Content-Length', '0')
                self.send_header('Connection', 'close')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(data)
            return

        if path == '/stream.mjpg':
            _viewers += 1
            logging.info("Live view: viewer connected (%d watching).", _viewers)
            started = time.time()
            try:
                self.send_response(200)
                self.send_header(
                    'Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                # Tells intermediaries not to buffer or rewrite the stream.
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()

                last_seq = -1
                while time.time() - started < MAX_STREAM_SECONDS:
                    with _lock:
                        seq = _seq
                    if seq == last_seq:
                        time.sleep(IDLE_SLEEP)
                        continue
                    data, seq = _encode_latest()
                    if data is None:
                        time.sleep(IDLE_SLEEP)
                        continue
                    last_seq = seq
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(
                        ('Content-Length: %d\r\n\r\n' % len(data)).encode())
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab; entirely normal
            except Exception as exc:
                logging.warning("Live view stream ended: %s", exc)
            finally:
                _viewers = max(0, _viewers - 1)
                logging.info("Live view: viewer left (%d watching).", _viewers)
            return

        self.send_response(404)
        self.send_header('Content-Length', '0')
        self.send_header('Connection', 'close')
        self.end_headers()


def start(port=8088, token=''):
    """Start the server on a daemon thread. Safe to call once; never raises."""
    global _token, _started
    if _started:
        return True
    if not token:
        logging.warning("Live view disabled: no STREAM_TOKEN set.")
        return False
    _token = token
    try:
        server = ThreadingHTTPServer(('0.0.0.0', port), _Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True,
                         name='live-stream').start()
        _started = True
        logging.info("Live view serving on :%d (/stream.mjpg, /snapshot.jpg).", port)
        return True
    except Exception as exc:
        # A port clash or permission problem must not stop detection.
        logging.error("Live view failed to start: %s", exc)
        return False
