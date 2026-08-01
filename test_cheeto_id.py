#!/usr/bin/env python3
"""Tests for the cropping helpers in cheeto_id.py.

The crop geometry is fiddly and silently consequential: crop_animal must
match between training and inference or the prototype stops matching, and
crop_with_context must never hand a captioner a sliver.

Run directly (no pytest needed):  python3 test_cheeto_id.py
"""

import numpy as np
import PIL.Image

import cheeto_id
from cheeto_id import crop_animal, crop_with_context, MIN_CROP_PX

W, H = 640, 480


def frame():
    return PIL.Image.new('RGB', (W, H))


# --- crop_animal: tight crop used for embedding -------------------------

def test_crop_animal_pads_symmetrically():
    c = crop_animal(frame(), [200, 200, 300, 300], pad_frac=0.2)
    # 100px box + 20px each side.
    assert c.size == (140, 140), c.size


def test_crop_animal_rejects_degenerate_and_tiny():
    assert crop_animal(frame(), None) is None
    assert crop_animal(frame(), [300, 300, 100, 100]) is None   # inverted
    assert crop_animal(frame(), [10, 10, 12, 12]) is None       # below MIN_CROP_PX


def test_crop_animal_clamps_to_frame():
    c = crop_animal(frame(), [600, 440, 900, 700], pad_frac=0.15)
    assert c is not None
    assert c.size[0] <= W and c.size[1] <= H


# --- crop_with_context: padded crop used for captioning -----------------

def test_context_crop_respects_minimum_fraction():
    # A tiny box must still yield >= min_frac of each dimension, or the
    # captioner gets mush.
    c = crop_with_context(frame(), [300, 240, 320, 260], pad_frac=0.5, min_frac=0.4)
    assert c.size[0] >= W * 0.4 - 1, c.size
    assert c.size[1] >= H * 0.4 - 1, c.size


def test_context_crop_pads_a_large_box():
    c = crop_with_context(frame(), [220, 140, 420, 340], pad_frac=0.5, min_frac=0.4)
    # 200px box * 1.5 = 300.
    assert c.size == (300, 300), c.size


def test_context_crop_shifts_instead_of_shrinking_at_edges():
    # Animal jammed in the corner: the crop must slide inward and keep full
    # size rather than get clipped to a sliver.
    for bbox in ([0, 0, 60, 60], [W - 60, H - 60, W, H], [0, H - 60, 60, H]):
        c = crop_with_context(frame(), bbox, pad_frac=0.5, min_frac=0.4)
        assert c.size[0] >= W * 0.4 - 1, (bbox, c.size)
        assert c.size[1] >= H * 0.4 - 1, (bbox, c.size)


def test_context_crop_never_exceeds_the_frame():
    for bbox in ([0, 0, W, H], [-50, -50, W + 50, H + 50], [10, 10, 630, 470]):
        c = crop_with_context(frame(), bbox, pad_frac=0.5, min_frac=0.4)
        assert c.size[0] <= W and c.size[1] <= H, (bbox, c.size)


def test_context_crop_passes_through_unusable_boxes():
    f = frame()
    assert crop_with_context(f, None) is f
    assert crop_with_context(f, [300, 300, 100, 100]) is f


def test_context_crop_is_strictly_more_context_than_crop_animal():
    # The captioner must always see at least as much as the embedder.
    bbox = [250, 200, 350, 300]
    tight = crop_animal(frame(), bbox, pad_frac=0.15)
    wide = crop_with_context(frame(), bbox, pad_frac=0.5, min_frac=0.4)
    assert wide.size[0] > tight.size[0] and wide.size[1] > tight.size[1]


def test_learned_profile_match_requires_threshold_and_margin():
    original = cheeto_id.embed_images
    cheeto_id.embed_images = lambda *args, **kwargs: np.asarray(
        [[1.0, 0.0]], dtype=np.float32)
    profiles = [
        {
            'animal_id': 1,
            'name': 'Mango',
            'slug': 'mango',
            'prototype': np.asarray([0.96, 0.28], dtype=np.float32),
            'threshold': 0.90,
        },
        {
            'animal_id': 2,
            'name': 'Pumpkin',
            'slug': 'pumpkin',
            'prototype': np.asarray([0.89, 0.46], dtype=np.float32),
            'threshold': 0.90,
        },
    ]
    try:
        match, score, margin = cheeto_id.match_learned_profiles(
            frame(), [200, 120, 400, 360], profiles)
        assert match['name'] == 'Mango'
        assert score > 0.95 and margin > 0.015

        profiles[1]['prototype'] = np.asarray([0.955, 0.296], dtype=np.float32)
        match, _, margin = cheeto_id.match_learned_profiles(
            frame(), [200, 120, 400, 360], profiles)
        assert match is None and margin < 0.015
    finally:
        cheeto_id.embed_images = original


def test_learned_cheeto_profile_can_rescue_a_static_rejection():
    """Reviewed Cheeto samples are a valid profile, not a special case the
    generic matcher must exclude. The detector decides when to ask the matcher;
    this verifies a reviewed profile is returned like any other regular."""
    original = cheeto_id.embed_images
    cheeto_id.embed_images = lambda *args, **kwargs: np.asarray(
        [[1.0, 0.0]], dtype=np.float32)
    profiles = [{
        'animal_id': 1,
        'name': 'Cheeto',
        'slug': 'cheeto',
        'prototype': np.asarray([0.94, 0.341], dtype=np.float32),
        'threshold': 0.90,
    }]
    try:
        match, score, _ = cheeto_id.match_learned_profiles(
            frame(), [160, 100, 480, 420], profiles)
        assert match is profiles[0]
        assert score >= 0.90
    finally:
        cheeto_id.embed_images = original


if __name__ == '__main__':
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
