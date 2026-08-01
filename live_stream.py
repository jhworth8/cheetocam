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
import os
import shutil
import subprocess
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
_audio_lock = threading.Lock()

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


def ready():
    """Whether the authenticated HTTP server is accepting connections."""
    return _started


def audio_available():
    """Cheap capability check used by the detector heartbeat.

    Opening ALSA here would briefly steal it from a live listener, so this
    checks the encoder and device nodes. The endpoint still handles a device
    disappearing without affecting video or detection.
    """
    return bool(_started and shutil.which('ffmpeg') and os.path.isdir('/dev/snd'))


def _encode_latest(quality=None, max_width=None):
    """Encode the newest frame, optionally smaller and/or lossier.

    A full-quality 640x480 frame is about 42KB, and the viewer polls a little
    over three times a second -- roughly 8MB per minute, or 480MB an hour. That
    is fine on wifi and expensive on cellular, so the client can ask for less.
    Downscaling is done here rather than on the phone because the point is to
    not send the bytes in the first place.
    """
    with _lock:
        f = _frame
        s = _seq
    if f is None:
        return None, s

    if max_width and f.shape[1] > max_width:
        scale = max_width / float(f.shape[1])
        # INTER_AREA is the right filter for shrinking; the default bilinear
        # aliases badly and the artefacts cost bytes back in the JPEG.
        f = cv2.resize(f, (max_width, int(f.shape[0] * scale)),
                       interpolation=cv2.INTER_AREA)

    q = JPEG_QUALITY if quality is None else max(20, min(95, quality))
    ok, buf = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), q])
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
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

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

        if path == '/audio.mp3':
            if not _audio_lock.acquire(blocking=False):
                body = b'audio already in use'
                self.send_response(409)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                return

            device = os.getenv('STREAM_AUDIO_DEVICE', 'plughw:CARD=B100,DEV=0')
            command = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-f', 'alsa', '-i', device,
                '-ac', '1', '-ar', '16000', '-b:a', '32k',
                '-f', 'mp3', 'pipe:1',
            ]
            process = None
            started = time.time()
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                self.send_response(200)
                self.send_header('Content-Type', 'audio/mpeg')
                self.send_header('Cache-Control', 'no-store, no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()
                while time.time() - started < MAX_STREAM_SECONDS:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                logging.warning("Live audio ended: %s", exc)
            finally:
                if process is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                _audio_lock.release()
            return

        if path == '/snapshot.jpg':
            def _int_param(name, lo, hi):
                try:
                    v = int((query.get(name) or [''])[0])
                except (TypeError, ValueError):
                    return None
                return max(lo, min(hi, v))

            # since=<seq> lets a polling client skip a frame it already has.
            # The viewer polls faster than the detector produces frames, so
            # without this a good fraction of requests re-send identical bytes.
            try:
                since = int((query.get('since') or [''])[0])
            except (TypeError, ValueError):
                since = None

            with _lock:
                current_seq = _seq
            if since is not None and since == current_seq:
                self.send_response(304)
                self.send_header('X-Frame-Seq', str(current_seq))
                self.send_header('Content-Length', '0')
                # Keep the TLS/proxy connection reusable. The Flutter viewer
                # polls several times per second through Cloudflare; forcing a
                # new connection for every unchanged frame was a large and
                # intermittent part of live-start latency.
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                return

            data, seq = _encode_latest(
                quality=_int_param('q', 20, 95),
                max_width=_int_param('w', 160, 1920),
            )
            if data is None:
                self.send_response(503)
                self.send_header('Content-Length', '0')
                self.send_header('Connection', 'close')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Frame-Seq', str(seq))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Connection', 'keep-alive')
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
        logging.info(
            "Live view serving on :%d (/stream.mjpg, /snapshot.jpg, /audio.mp3).",
            port,
        )
        return True
    except Exception as exc:
        # A port clash or permission problem must not stop detection.
        logging.error("Live view failed to start: %s", exc)
        return False
