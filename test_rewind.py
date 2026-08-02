from rewind import MediaRing, concat_script


def test_an_empty_ring_asks_for_nothing():
    ring = MediaRing(max_seconds=15, max_bytes=1000)
    assert len(ring) == 0
    assert ring.bytes_held == 0
    assert ring.window(now=100, seconds=15) == []
    assert ring.span() == 0.0


def test_the_byte_budget_is_the_one_that_binds():
    # Well inside fifteen seconds, well outside the byte budget. Memory is what
    # kills the process, so memory wins.
    ring = MediaRing(max_seconds=15, max_bytes=300)
    for i in range(10):
        ring.add(i * 0.1, b'x' * 100)
    assert ring.bytes_held <= 300
    assert len(ring) == 3


def test_the_second_budget_still_trims_a_quiet_camera():
    # Tiny frames, so bytes never bind; the age rule has to do the work or a
    # still driveway would keep frames from an hour ago.
    ring = MediaRing(max_seconds=5, max_bytes=10_000_000)
    for i in range(20):
        ring.add(i, b'x')
    assert ring.span() <= 5
    assert len(ring) == 6


def test_one_oversized_frame_is_kept_rather_than_leaving_nothing():
    ring = MediaRing(max_seconds=15, max_bytes=10)
    ring.add(1.0, b'x' * 500)
    assert len(ring) == 1
    assert ring.window(now=1.0, seconds=15)


def test_a_window_is_the_recent_end_not_the_whole_ring():
    ring = MediaRing(max_seconds=60, max_bytes=10_000_000)
    for i in range(10):
        ring.add(float(i), b'frame%d' % i)
    recent = ring.window(now=9.0, seconds=3)
    assert [t for t, _ in recent] == [6.0, 7.0, 8.0, 9.0]


def test_clearing_drops_the_bytes_too():
    ring = MediaRing(max_seconds=15, max_bytes=10_000)
    ring.add(1.0, b'x' * 100)
    ring.clear()
    assert len(ring) == 0 and ring.bytes_held == 0


def test_each_frame_carries_the_gap_to_the_next_one():
    # An unevenly spaced capture must not play back at a constant rate.
    script = concat_script([(0.0, b'a'), (0.5, b'b'), (0.6, b'c')],
                           tail_seconds=0.2)
    assert 'duration 0.500' in script
    assert 'duration 0.100' in script
    assert 'duration 0.200' in script


def test_the_last_frame_is_repeated_so_it_is_not_dropped():
    script = concat_script([(0.0, b'a'), (0.5, b'b')])
    assert script.count("file 'f00001.jpg'") == 2


def test_a_zero_gap_never_becomes_a_zero_duration():
    # Two frames with the same timestamp would otherwise emit duration 0.000,
    # which ffmpeg treats as an error rather than an instant frame.
    script = concat_script([(1.0, b'a'), (1.0, b'b')])
    assert 'duration 0.000' not in script


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
