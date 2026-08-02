"""Camera capture adapters used by the detector."""

import logging
import time

import cv2
import numpy as np
import requests


class HttpSnapshotCapture:
    """A ``cv2.VideoCapture``-compatible adapter for JPEG endpoints.

    Some embedded cameras expose a reliable one-shot JPEG endpoint but an
    MJPEG stream that FFmpeg cannot consume. Polling snapshots here keeps that
    device-specific behavior out of the detector loop.
    """

    def __init__(self, url, interval_seconds=0.2, timeout_seconds=5.0,
                 rotation_degrees=0):
        self.url = url
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self.rotation_degrees = int(rotation_degrees) % 360
        if self.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("rotation_degrees must be 0, 90, 180, or 270")
        # Mutable on purpose: a camera that gets repositioned is turned right
        # way up from the app, and the detector applies it to the next frame
        # rather than needing a restart.
        self._session = requests.Session()
        self._opened = False
        self._cached_frame = None
        self._last_fetch = 0.0
        self._cached_frame = self._fetch()
        self._opened = self._cached_frame is not None

    def _fetch(self):
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < self.interval_seconds:
            time.sleep(self.interval_seconds - elapsed)
        try:
            response = self._session.get(
                self.url,
                headers={"Cache-Control": "no-cache"},
                timeout=(2.0, self.timeout_seconds),
            )
            response.raise_for_status()
            encoded = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("response was not a decodable JPEG")
            if self.rotation_degrees == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_degrees == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_degrees == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self._last_fetch = time.monotonic()
            self._opened = True
            return frame
        except (requests.RequestException, ValueError) as exc:
            logging.debug("Snapshot camera read failed: %s", exc)
            self._opened = False
            return None

    def isOpened(self):
        return self._opened

    def read(self):
        if self._cached_frame is not None:
            frame = self._cached_frame
            self._cached_frame = None
            return True, frame
        frame = self._fetch()
        return frame is not None, frame

    def set(self, _property, _value):
        return True

    def release(self):
        self._opened = False
        self._cached_frame = None
        self._session.close()
