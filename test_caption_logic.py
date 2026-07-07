#!/usr/bin/env python3
"""Tests for caption_logic.py — the YOLO-label + VLM-caption decision logic.

Run directly (no pytest needed):  python3 test_caption_logic.py
"""

from caption_logic import (
    animals_in_caption,
    looks_like_false_positive,
    should_suppress,
    resolve_reported_classes,
    parse_gemini_confirmation,
)


def test_cow_label_corrected_to_cat_by_caption():
    # The headline bug: YOLO says cow/elephant/giraffe, caption says cat.
    for wrong in (['cow'], ['elephant'], ['giraffe'], ['zebra', 'cow']):
        caption = "A large orange cat is sitting on a striped mat by the door."
        assert not should_suppress(wrong, caption)
        reported, note = resolve_reported_classes(wrong, caption)
        assert reported == ['cat'], f"{wrong} -> {reported}"
        assert note and 'detector guessed' in note


def test_lookalike_with_no_caption_hedges_to_cat():
    # Both VLMs failed: never report "Elephant at the door", hedge to cat.
    reported, note = resolve_reported_classes(['elephant'], "")
    assert reported == ['cat']
    assert note and 'unverified' in note


def test_lookalike_with_animal_free_caption_is_suppressed():
    caption = "A porch with a striped mat covered in snow. A white door frame."
    assert should_suppress(['giraffe'], caption)


def test_phantom_cat_with_animal_free_caption_is_suppressed():
    # User-reported: false positives OF the cat. Caption sees an empty scene.
    caption = "The image shows a wooden door and a welcome mat in the evening."
    assert should_suppress(['cat'], caption)


def test_short_or_empty_caption_never_suppresses():
    assert not should_suppress(['cat'], "")
    assert not should_suppress(['cat'], "A blurry image.")


def test_real_cat_caption_passes_through():
    caption = "A gray tabby cat walking across the snowy porch."
    assert not should_suppress(['cat'], caption)
    reported, note = resolve_reported_classes(['cat'], caption)
    assert reported == ['cat']
    assert note is None


def test_negation_still_suppresses():
    assert looks_like_false_positive("No cat is visible; the porch is empty.")
    assert should_suppress(['cat'], "No cat is visible; the porch is empty.")


def test_person_without_animal_suppresses():
    caption = "A man in a red jacket is standing at the door holding a package."
    assert should_suppress(['cat'], caption)


def test_caption_names_animal_yolo_has_no_class_for():
    caption = "A raccoon is eating from a bowl on the porch."
    reported, note = resolve_reported_classes(['dog'], caption)
    assert reported == ['raccoon']
    assert note and 'detector guessed' in note
    # A raccoon caption is an animal — must NOT be suppressed.
    assert not should_suppress(['dog'], caption)


def test_animals_in_caption_priority_and_wordboundaries():
    assert animals_in_caption("A cat and a dog play.") == ['cat', 'dog']
    # 'category' must not match 'cat'; 'mat' must not match 'rat'.
    assert animals_in_caption("A category of doormats on the mat.") == []
    assert animals_in_caption("") == []


def test_parse_confirmation_structured_yes():
    text = "VISIBLE: yes\nDESCRIPTION: An orange tabby cat sitting by the door."
    confirmed, desc = parse_gemini_confirmation(text, ['cow'])
    assert confirmed
    assert desc.startswith("An orange tabby")


def test_parse_confirmation_structured_no():
    text = "VISIBLE: no\nDESCRIPTION: An empty porch with a striped mat."
    confirmed, desc = parse_gemini_confirmation(text, ['cat'])
    assert not confirmed
    assert 'empty porch' in desc.lower()


def test_parse_confirmation_freeform_wrong_label_but_real_animal():
    # Gemini ignores the format but names a cat while YOLO guessed cow —
    # must count as confirmed, not suppressed.
    text = "I can see a small cat sitting near the door."
    confirmed, _ = parse_gemini_confirmation(text, ['cow'])
    assert confirmed


def test_parse_confirmation_freeform_negation():
    confirmed, _ = parse_gemini_confirmation(
        "I don't see any cat in this image.", ['cat'])
    assert not confirmed


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
