"""A bounded ring of recent media, and the muxing of it into a clip.

Kept separate from ``live_stream`` for the same reason ``capture_gate`` is kept
separate from the detector: the interesting part is a trimming rule that is
cheap to get wrong and cheap to test, and getting it wrong on a Pi that is
already holding two detectors, Florence and CLIP means an out-of-memory kill
rather than a slightly wrong picture.

The ring is bounded by **bytes first**, seconds second. Fifteen seconds is not
a size: the same fifteen seconds is 3MB of a still, dark driveway and 40MB of a
sunlit hedge moving in the wind, and only one of those fits in the headroom
this Pi actually has.
"""

import os
import shutil
import subprocess
import tempfile
import threading


class MediaRing:
    """Timestamped payloads, dropped oldest-first to stay inside two budgets.

    Used for both JPEG frames and raw PCM chunks — a chunk of audio and a
    frame of video have the same shape of problem, so they share the rule.
    """

    def __init__(self, max_seconds, max_bytes):
        self.max_seconds = max(0.0, float(max_seconds))
        self.max_bytes = max(0, int(max_bytes))
        self._items = []
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def bytes_held(self):
        with self._lock:
            return self._bytes

    def __len__(self):
        with self._lock:
            return len(self._items)

    def clear(self):
        with self._lock:
            self._items = []
            self._bytes = 0

    def add(self, timestamp, payload):
        """Append, then drop from the front until both budgets are met."""
        if not payload:
            return
        with self._lock:
            self._items.append((float(timestamp), payload))
            self._bytes += len(payload)
            # Bytes first: this is the budget that decides whether the process
            # survives. The newest item is never dropped, even if it alone is
            # over budget — a ring that can return nothing is not a ring.
            while len(self._items) > 1 and self._bytes > self.max_bytes:
                self._bytes -= len(self._items.pop(0)[1])
            newest = self._items[-1][0]
            while (len(self._items) > 1
                   and newest - self._items[0][0] > self.max_seconds):
                self._bytes -= len(self._items.pop(0)[1])

    def window(self, now, seconds):
        """Everything recorded in the last ``seconds``, oldest first."""
        cutoff = float(now) - max(0.0, float(seconds))
        with self._lock:
            return [item for item in self._items if item[0] >= cutoff]

    def span(self):
        """Seconds between the oldest and newest item, or 0.0 when short."""
        with self._lock:
            if len(self._items) < 2:
                return 0.0
            return self._items[-1][0] - self._items[0][0]


def concat_script(frames, tail_seconds=0.2):
    """An ffmpeg concat demuxer script for unevenly spaced frames.

    The detector's frame rate is not constant — it changes with load, with
    battery saver, and with whether anything is moving — so a fixed output fps
    would play a busy stretch back at the wrong speed. Each frame instead
    carries the gap to the one after it. The last frame has no successor, so it
    gets ``tail_seconds``; without a duration ffmpeg drops it entirely.
    """
    lines = []
    for index, (timestamp, _payload) in enumerate(frames):
        lines.append("file '%s'" % _frame_name(index))
        if index + 1 < len(frames):
            gap = frames[index + 1][0] - timestamp
        else:
            gap = tail_seconds
        lines.append('duration %.3f' % max(0.001, gap))
    if frames:
        # The concat demuxer needs the final file repeated for its duration to
        # be honoured. This is documented ffmpeg behaviour, not a workaround.
        lines.append("file '%s'" % _frame_name(len(frames) - 1))
    return '\n'.join(lines) + '\n'


def _frame_name(index):
    return 'f%05d.jpg' % index


def available():
    """Whether a clip can be muxed at all."""
    return bool(shutil.which('ffmpeg'))


def mux(frames, pcm_chunks=None, audio_rate=16000, timeout=60,
        video_encoder='libx264'):
    """Mux JPEG frames and optional PCM into an MP4, returned as bytes.

    Raises on failure rather than returning None: the endpoint above needs to
    tell the difference between "no clip" and "a clip that did not encode", and
    a silent empty file is the worst of both.
    """
    if not frames:
        raise ValueError('nothing recorded yet')

    with tempfile.TemporaryDirectory(prefix='rewind-') as work:
        for index, (_timestamp, payload) in enumerate(frames):
            with open(os.path.join(work, _frame_name(index)), 'wb') as handle:
                handle.write(payload)
        list_path = os.path.join(work, 'frames.txt')
        with open(list_path, 'w') as handle:
            handle.write(concat_script(frames))

        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'concat', '-safe', '0', '-i', 'frames.txt']

        audio = b''.join(chunk for _ts, chunk in (pcm_chunks or []))
        if audio:
            pcm_path = os.path.join(work, 'audio.pcm')
            with open(pcm_path, 'wb') as handle:
                handle.write(audio)
            command += ['-f', 's16le', '-ar', str(int(audio_rate)), '-ac', '1',
                        '-i', 'audio.pcm']

        command += ['-c:v', video_encoder]
        if video_encoder == 'libx264':
            # ultrafast because this runs on a Pi that is already at three
            # quarters of its cores, and nobody is grading a 15 second clip on
            # compression efficiency.
            command += ['-preset', 'ultrafast', '-crf', '26']
        command += ['-pix_fmt', 'yuv420p']
        if audio:
            command += ['-c:a', 'aac', '-b:a', '48k', '-shortest']
        command += ['-movflags', '+faststart', 'out.mp4']

        result = subprocess.run(command, cwd=work, capture_output=True,
                                timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or b'').decode('utf-8', 'replace').strip()
                or 'ffmpeg failed')
        with open(os.path.join(work, 'out.mp4'), 'rb') as handle:
            return handle.read()
