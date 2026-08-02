"""Pure alert-cadence decisions for the detector.

Capturing an observation and notifying a person are deliberately separate
decisions.  Keeping this module free of camera/model imports makes the two
modes cheap to unit test on development machines and on the Pi.
"""


def should_alert(settings, visit_result, now, next_alert_epoch):
    """Return whether the current saved observation should notify people.

    Three cadences, all of which leave observation capture untouched:

    ``first_of_visit``
        One alert when the visit starts, silence until a new visit begins.
    ``visit_then_repeat``
        The visit start always alerts, and a visitor who stays alerts again
        every cooldown period.  A cat that settles in for an hour is still
        news; one alert an hour into the nap is not spam.
    ``every_detection``
        Cooldown only, with no notion of visits at all.

    Both visit modes fail closed when visit attachment is unavailable.  A
    missed alert during an API problem is preferable to turning one seated cat
    into an alert every observation.
    """
    mode = settings.get("alert_mode", "every_detection")
    if mode in ("first_of_visit", "visit_then_repeat"):
        if not visit_result:
            return False
        if visit_result.get("is_new_visit") is True:
            return True
        # A continuing visit repeats only in the repeat cadence, and only once
        # the same cooldown that spaces every_detection alerts has elapsed.
        return (mode == "visit_then_repeat"
                and visit_result.get("is_new_visit") is False
                and now >= next_alert_epoch)
    return now >= next_alert_epoch
