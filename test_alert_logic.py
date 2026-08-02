from alert_logic import should_alert


def test_once_per_visit_alerts_only_for_new_visit():
    settings = {"alert_mode": "first_of_visit"}
    assert should_alert(settings, {"is_new_visit": True}, 100, 0)
    assert not should_alert(settings, {"is_new_visit": False}, 100, 0)


def test_once_per_visit_fails_closed_without_visit_state():
    settings = {"alert_mode": "first_of_visit"}
    assert not should_alert(settings, None, 100, 0)
    assert not should_alert(settings, {}, 100, 0)


def test_repeat_cadence_alerts_at_the_start_and_then_on_the_cooldown():
    settings = {"alert_mode": "visit_then_repeat"}
    # A new visit alerts immediately, even inside a previous cooldown.
    assert should_alert(settings, {"is_new_visit": True}, 100, 900)
    # A visitor who stays is silent until the cooldown elapses, then alerts.
    assert not should_alert(settings, {"is_new_visit": False}, 899, 900)
    assert should_alert(settings, {"is_new_visit": False}, 900, 900)


def test_repeat_cadence_still_fails_closed_without_visit_state():
    settings = {"alert_mode": "visit_then_repeat"}
    assert not should_alert(settings, None, 900, 100)
    assert not should_alert(settings, {}, 900, 100)


def test_every_detection_keeps_independent_alert_cooldown():
    settings = {"alert_mode": "every_detection"}
    assert not should_alert(settings, {"is_new_visit": False}, 99, 100)
    assert should_alert(settings, {"is_new_visit": False}, 100, 100)


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
