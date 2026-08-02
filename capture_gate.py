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
``capture_interval``
    How hard to poll a camera that is already worth polling. Battery saver
    lives here rather than in the two gates above because it must never stop a
    capture, only slow one.
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


def capture_interval(base_interval, saver_interval, saver_enabled,
                     live_viewers, activity_seen):
    """Seconds to leave between frame fetches.

    A wireless camera answers every snapshot request with a full sensor read,
    JPEG encode and Wi-Fi transmit, so the poll rate is most of what the Pi
    costs its battery. Saver mode slows the quiet case only. Three things put
    it back to full speed, because each is a moment where missing frames costs
    more than the battery does:

    - somebody is watching live,
    - the detector just saw something, even unconfirmed,
    - the base interval is already the slower of the two.

    That second one also keeps the post-detection settle drain honest: it
    reads ten frames in a row expecting them to be cheap.
    """
    base = max(0.0, float(base_interval))
    if not saver_enabled or live_viewers > 0 or activity_seen:
        return base
    return max(base, max(0.0, float(saver_interval)))
