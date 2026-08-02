from capture_gate import capture_interval, should_capture, should_detect


def test_an_ordinary_camera_is_never_gated():
    assert should_capture(False, 0, False)
    assert should_detect(False, False)


def test_a_motion_gated_camera_idles_until_its_sensor_fires():
    assert not should_capture(True, 0, False)
    assert should_capture(True, 0, True)


def test_a_live_viewer_gets_frames_with_nothing_moving():
    assert should_capture(True, 1, False)


def test_but_a_live_view_can_never_create_an_observation():
    # The whole point of the sensor gate: watching the yard must not be able to
    # manufacture a visit out of a passing shadow.
    assert not should_detect(True, False)
    assert should_detect(True, True)


def test_saver_off_polls_at_the_configured_rate():
    assert capture_interval(0.2, 1.0, False, 0, False) == 0.2


def test_saver_slows_a_quiet_camera():
    assert capture_interval(0.2, 1.0, True, 0, False) == 1.0


def test_saver_gets_out_of_the_way_for_a_viewer_or_a_sighting():
    assert capture_interval(0.2, 1.0, True, 1, False) == 0.2
    assert capture_interval(0.2, 1.0, True, 0, True) == 0.2


def test_saver_never_makes_a_slow_camera_faster():
    # A camera already polled once every two seconds is not sped up to one.
    assert capture_interval(2.0, 1.0, True, 0, False) == 2.0


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
