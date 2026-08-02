from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_source import HttpSnapshotCapture


def _jpeg_response():
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 180
    ok, encoded = cv2.imencode('.jpg', frame)
    assert ok
    response = Mock()
    response.content = encoded.tobytes()
    response.raise_for_status.return_value = None
    return response


def test_snapshot_capture_decodes_repeated_frames():
    session = Mock()
    session.get.return_value = _jpeg_response()
    with patch('camera_source.requests.Session', return_value=session):
        capture = HttpSnapshotCapture('http://camera/snapshot', interval_seconds=0)
        assert capture.isOpened()
        first_ok, first = capture.read()
        second_ok, second = capture.read()
        assert first_ok and second_ok
        assert first.shape == (24, 32, 3)
        assert second.shape == (24, 32, 3)
        assert session.get.call_count == 2
        capture.release()
        assert not capture.isOpened()
        session.close.assert_called_once()


def test_snapshot_capture_rejects_invalid_jpeg():
    session = Mock()
    response = Mock()
    response.content = b'not a jpeg'
    response.raise_for_status.return_value = None
    session.get.return_value = response
    with patch('camera_source.requests.Session', return_value=session):
        capture = HttpSnapshotCapture('http://camera/snapshot', interval_seconds=0)
        assert not capture.isOpened()
        ok, frame = capture.read()
        assert not ok
        assert frame is None


def test_snapshot_capture_rotates_clockwise():
    session = Mock()
    session.get.return_value = _jpeg_response()
    with patch('camera_source.requests.Session', return_value=session):
        capture = HttpSnapshotCapture(
            'http://camera/snapshot', interval_seconds=0, rotation_degrees=90)
        ok, frame = capture.read()
        assert ok
        assert frame.shape == (32, 24, 3)
