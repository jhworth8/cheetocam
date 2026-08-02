"""Pure decisions about when a motion-gated camera should be pulled from.

Kept free of camera/model imports for the same reason as ``alert_logic``: the
rules are cheap to get wrong and cheap to test, and getting them wrong is
either a camera that records nothing or one that streams an empty yard all day.

Two separate questions, deliberately not collapsed into one:

``should_capture``
    Is it worth asking the camera for a frame at all? A live viewer counts,
    because somebody watching deserves pictures whether or not anything moved.
``should_detect``
    May this frame create an observation? A live view or a scheduled check must
    never manufacture a visit, so only the sensor can open this gate.
"""


def should_capture(capture_on_motion, live_viewers, motion_active):
    """Whether the detector should fetch a frame right now."""
    if not capture_on_motion:
        return True
    if live_viewers > 0:
        return True
    return bool(motion_active)


def should_detect(motion_gated, motion_active):
    """Whether a captured frame may become an observation."""
    if not motion_gated:
        return True
    return bool(motion_active)
