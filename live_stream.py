#!/usr/bin/env python3
"""On-demand live view for the Cheeto cam.

Why this lives inside the detector process rather than being its own service:
/dev/video0 can only be opened by one process, and the detector holds it open
permanently. A separate streaming server would fight it for the device. The
detector already decodes every frame for YOLO, so this just re-serves those.

Cost when nobody is watching: one reference assignment per frame. JPEG encoding
only happens while a viewer is actually connected, and it happens on the serving
thread, never on the detection loop.

Endpoints, deliberately few:
  /stream.mjpg   real-time multipart MJPEG, lowest latency
  /snapshot.jpg  a single frame — the fallback, because some proxies buffer
                 streaming responses and a polling client works everywhere
  /audio.mp3     live microphone
  /rewind.mp4    the last few seconds, muxed on demand

Rewind is the one thing here that costs something while nobody is watching: it
has to already be recording by the time somebody decides they want it. That
cost is kept to a JPEG encode a few times a second, on its own thread, out of
the detection loop, into a ring bounded in bytes — see ``rewind.py``.

Nothing here may ever raise into the detector. Every public function swallows
its own exceptions; a broken stream must not stop the cat being watched.
"""

import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

import rewind

_lock = threading.Lock()
_frame = None          # most recent BGR frame from the detector
_seq = 0               # bumped per frame so viewers can wait for a *new* one
_viewers = 0           # live MJPEG connections
_last_viewer_request = 0.0  # polling Flutter client's most recent request
_token = ''
_started = False

# Bytes per frame are the whole cost of live view once a stream leaves the LAN.
# On the Pi these defaults are free; through the tunnel they are the bottleneck,
# so a camera reached from outside can trade sharpness for frames without
# touching the one at the front door. The defaults are the values that were
# hardcoded here, so an .env setting neither behaves exactly as before.
JPEG_QUALITY = max(20, min(95, int(os.getenv('STREAM_JPEG_QUALITY', '70'))))
# 0 means "no cap" — send the frame at its native width.
STREAM_MAX_WIDTH = max(0, int(os.getenv('STREAM_MAX_WIDTH', '0')))
IDLE_SLEEP = 0.04
# A viewer that stalls should not hold a thread forever.
MAX_STREAM_SECONDS = 60 * 30

# Rewind. The frame budget is bytes rather than seconds on purpose: fifteen
# seconds of a still driveway and fifteen seconds of a hedge in the wind are
# the same duration and nothing like the same size, and this Pi is already
# holding two detectors, Florence and CLIP.
REWIND_ENABLED = os.getenv(
    'REWIND_ENABLED', '1').strip().lower() not in ('0', 'false', 'no', 'off')
REWIND_SECONDS = max(1.0, float(os.getenv('REWIND_SECONDS', '15')))
# 4, not the detector's 10: measured at 3.2ms to encode a 640x480 frame on
# this Pi, so 4fps costs about 1.3% of one core. That matters because the
# front-door detector already runs pinned against its 150% quota, and anything
# added there is taken from detection rather than from idle.
REWIND_FPS = max(1.0, float(os.getenv('REWIND_FPS', '4')))
REWIND_WIDTH = max(160, int(os.getenv('REWIND_WIDTH', '640')))
REWIND_QUALITY = max(20, min(95, int(os.getenv('REWIND_QUALITY', '60'))))
REWIND_MAX_BYTES = max(
    1_000_000, int(os.getenv('REWIND_MAX_BYTES', str(16 * 1024 * 1024))))
# Separate from REWIND_ENABLED because it is a different decision: recording
# sound means holding the microphone open continuously, which is a bigger thing
# to agree to than keeping a few seconds of pictures.
REWIND_AUDIO = os.getenv(
    'REWIND_AUDIO', '1').strip().lower() not in ('0', 'false', 'no', 'off')
# Recorded a little longer than offered, so asking for the last fifteen
# seconds does not race the trimmer and come back with fourteen.
_RING_SECONDS = REWIND_SECONDS + 3

AUDIO_RATE = 16000
AUDIO_CHUNK = 4096
# s16le mono at 16kHz is 32KB a second, so the whole audio ring is under a
# megabyte and never the reason anything gets trimmed.
_rewind_frames = rewind.MediaRing(_RING_SECONDS, REWIND_MAX_BYTES)
_rewind_pcm = rewind.MediaRing(_RING_SECONDS, AUDIO_RATE * 2 * 60)
_rewind_mux_lock = threading.Lock()


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


def clear():
    """Discard the last frame when a wireless camera returns to sleep."""
    global _frame, _seq
    try:
        with _lock:
            _frame = None
            _seq += 1
    except Exception:
        pass


def viewers():
    """Number of active viewers, including the app's snapshot-polling mode.

    Flutter uses short authenticated snapshot requests rather than holding an
    MJPEG socket open. Treat requests within the last 12 seconds as one viewer
    so a sleeping wireless camera can be asked to wake and remain awake.
    """
    return _viewers + (1 if time.time() - _last_viewer_request < 12 else 0)


def ready():
    """Whether the authenticated HTTP server is accepting connections."""
    return _started


def audio_available():
    """Cheap capability check used by the detector heartbeat.

    Opening ALSA here would briefly steal it from a live listener, so this
    checks the encoder and device nodes. The endpoint still handles a device
    disappearing without affecting video or detection.
    """
    enabled = os.getenv('STREAM_AUDIO_ENABLED', '1').strip().lower()
    if enabled in ('0', 'false', 'no', 'off') or not _started:
        return False
    if not shutil.which('ffmpeg'):
        return False
    # A network camera carries its audio inside its own stream, so there is no
    # local sound card to look for and /dev/snd says nothing about whether
    # listening will work.
    if os.getenv('STREAM_AUDIO_SOURCE', '').strip():
        return True
    return os.path.isdir('/dev/snd')


class _AudioSource:
    """One ALSA capture, fanned out to everyone who wants it.

    Rewind has to be recording audio continuously or it has nothing to offer,
    and live listening cannot be locked out for the whole time it does. So the
    device is opened once, as raw PCM, and both the live endpoint and the
    rewind ring are subscribers. It starts on the first subscriber and stops on
    the last, so a camera nobody has ever listened to still never opens a mic.

    A slow subscriber is dropped frames, never waited for: a phone on a bad
    connection must not be able to stall the recorder.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = set()
        self._process = None

    def subscribe(self):
        channel = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.add(channel)
            if self._process is None:
                self._start_locked()
        return channel

    def unsubscribe(self, channel):
        with self._lock:
            self._subscribers.discard(channel)
            if not self._subscribers:
                self._stop_locked()

    def _start_locked(self):
        # Two shapes of microphone. A USB mic on the Pi is an ALSA device and
        # belongs to the camera it sits next to. A network camera carries its
        # own audio inside its stream, and STREAM_AUDIO_SOURCE names that
        # stream. Everything downstream is identical: one mono 16-bit PCM
        # pipe, fanned out to live listeners and the rewind ring.
        source = os.getenv('STREAM_AUDIO_SOURCE', '').strip()
        if source:
            # TCP for the same reason the detector reads RTSP over TCP: UDP
            # loss on this stream arrives as audible dropouts.
            input_args = ['-rtsp_transport', 'tcp', '-i', source]
        else:
            device = os.getenv('STREAM_AUDIO_DEVICE', 'plughw:CARD=B100,DEV=0')
            input_args = ['-f', 'alsa', '-i', device]
        try:
            self._process = subprocess.Popen(
                ['ffmpeg', '-hide_banner', '-loglevel', 'error']
                + input_args
                + ['-vn', '-ac', '1', '-ar', str(AUDIO_RATE),
                   '-f', 's16le', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except Exception as exc:
            logging.warning('Microphone unavailable: %s', exc)
            self._process = None
            return
        threading.Thread(target=self._pump, args=(self._process,),
                         daemon=True, name='audio-source').start()

    def _stop_locked(self):
        process, self._process = self._process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

    def _pump(self, process):
        try:
            while True:
                chunk = process.stdout.read(AUDIO_CHUNK)
                if not chunk:
                    break
                with self._lock:
                    if process is not self._process:
                        break
                    channels = list(self._subscribers)
                for channel in channels:
                    try:
                        channel.put_nowait(chunk)
                    except queue.Full:
                        pass
        except Exception:
            pass


_audio_source = _AudioSource()


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

    def _text(self, code, body):
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def _deny(self):
        self._text(403, b'forbidden')

    def do_GET(self):
        global _viewers, _last_viewer_request
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

        # Reaching any authenticated media endpoint means a person is actively
        # trying to view this camera, even if no frame exists yet.
        _last_viewer_request = time.time()

        if path == '/audio.mp3':
            if not audio_available():
                self._text(404, b'audio unavailable')
                return
            # No longer exclusive: the microphone is opened once by
            # _AudioSource and shared, so a second listener — or the rewind
            # recorder — no longer locks anybody out.
            channel = _audio_source.subscribe()
            encoder = None
            started = time.time()
            try:
                encoder = subprocess.Popen(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'error',
                     '-f', 's16le', '-ar', str(AUDIO_RATE), '-ac', '1',
                     '-i', 'pipe:0', '-b:a', '32k', '-f', 'mp3', 'pipe:1'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0)

                def _feed():
                    try:
                        while True:
                            chunk = channel.get(timeout=5)
                            encoder.stdin.write(chunk)
                    except Exception:
                        pass

                threading.Thread(target=_feed, daemon=True,
                                 name='audio-encode').start()
                self.send_response(200)
                self.send_header('Content-Type', 'audio/mpeg')
                self.send_header('Cache-Control', 'no-store, no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()
                while time.time() - started < MAX_STREAM_SECONDS:
                    chunk = encoder.stdout.read(AUDIO_CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                logging.warning("Live audio ended: %s", exc)
            finally:
                _audio_source.unsubscribe(channel)
                if encoder is not None:
                    encoder.terminate()
                    try:
                        encoder.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        encoder.kill()
            return

        if path == '/rewind.mp4':
            if not (REWIND_ENABLED and rewind.available()):
                self._text(404, b'rewind unavailable')
                return
            try:
                seconds = float((query.get('seconds') or [''])[0])
            except (TypeError, ValueError):
                seconds = REWIND_SECONDS
            seconds = max(1.0, min(REWIND_SECONDS, seconds))

            now = time.monotonic()
            frames = _rewind_frames.window(now, seconds)
            if not frames:
                # Nothing recorded: an asleep wireless camera, or a detector
                # that has only just started. A 503 rather than an empty file,
                # so the app can say "nothing to rewind yet" and mean it.
                self._text(503, b'nothing buffered yet')
                return

            # One encode at a time. Two at once on a Pi already at three
            # quarters of its cores would make both of them slow and neither
            # of them wrong, which is the worst way to fail.
            if not _rewind_mux_lock.acquire(blocking=False):
                self._text(409, b'a clip is already being made')
                return
            try:
                clip = rewind.mux(frames, _rewind_pcm.window(now, seconds),
                                  audio_rate=AUDIO_RATE)
            except Exception as exc:
                logging.warning('Rewind failed: %s', exc)
                self._text(500, b'could not build the clip')
                return
            finally:
                _rewind_mux_lock.release()

            logging.info('Rewind: %.1fs, %d frames, %dKB.',
                         frames[-1][0] - frames[0][0], len(frames),
                         len(clip) // 1024)
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(len(clip)))
            self.send_header('Content-Disposition',
                             'attachment; filename="rewind.mp4"')
            self.send_header('Cache-Control', 'no-store, no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(clip)
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
                    data, seq = _encode_latest(
                        max_width=STREAM_MAX_WIDTH or None)
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


def _rewind_video_loop():
    """Sample the published frames into the ring at REWIND_FPS.

    Deliberately not every frame: the detector can publish at 10fps and a clip
    does not need it, so this takes roughly one in two and leaves the CPU to
    the thing that actually finds cats.
    """
    interval = 1.0 / REWIND_FPS
    last_seq = -1
    next_at = 0.0
    while True:
        now = time.monotonic()
        with _lock:
            seq = _seq
        if seq == last_seq or now < next_at:
            time.sleep(IDLE_SLEEP)
            continue
        data, seq = _encode_latest(quality=REWIND_QUALITY,
                                   max_width=REWIND_WIDTH)
        if data is None:
            # No frame at all: a wireless camera asleep, or between wakes.
            time.sleep(IDLE_SLEEP)
            continue
        last_seq = seq
        next_at = now + interval
        _rewind_frames.add(now, data)


def _rewind_audio_loop():
    """Keep the PCM ring filled for as long as this camera has a microphone."""
    channel = _audio_source.subscribe()
    try:
        while True:
            try:
                chunk = channel.get(timeout=5)
            except queue.Empty:
                continue
            _rewind_pcm.add(time.monotonic(), chunk)
    except Exception as exc:
        logging.warning('Rewind audio stopped: %s', exc)
        _audio_source.unsubscribe(channel)


def buffered_seconds():
    """How much rewind is actually available, for the detector heartbeat."""
    return round(_rewind_frames.span(), 1)


def buffered_bytes():
    return _rewind_frames.bytes_held + _rewind_pcm.bytes_held


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
        if REWIND_ENABLED and rewind.available():
            threading.Thread(target=_rewind_video_loop, daemon=True,
                             name='rewind-video').start()
            if REWIND_AUDIO and audio_available():
                threading.Thread(target=_rewind_audio_loop, daemon=True,
                                 name='rewind-audio').start()
        logging.info(
            "Live view serving on :%d (/stream.mjpg, /snapshot.jpg, "
            "/audio.mp3%s).",
            port,
            ', /rewind.mp4' if REWIND_ENABLED and rewind.available() else '',
        )
        return True
    except Exception as exc:
        # A port clash or permission problem must not stop detection.
        logging.error("Live view failed to start: %s", exc)
        return False
