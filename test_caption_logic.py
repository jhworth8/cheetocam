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
        reported, note, hedged = resolve_reported_classes(wrong, caption)
        assert reported == ['cat'], f"{wrong} -> {reported}"
        assert not hedged


def test_notes_never_leak_yolos_wrong_label():
    # The user sees these notes. "(detector guessed: elephant)" is noise.
    for wrong in (['cow'], ['elephant'], ['bear']):
        for caption in ("An orange cat by the door.", "", "A fox on the mat."):
            _, note, _ = resolve_reported_classes(wrong, caption)
            if note:
                assert wrong[0] not in note, f"{wrong} leaked into {note!r}"


def test_lookalike_with_no_caption_hedges_to_cat():
    # Both VLMs failed: never report "Elephant at the door", hedge to cat.
    reported, note, hedged = resolve_reported_classes(['elephant'], "")
    assert reported == ['cat']
    assert note and 'unverified' in note


def test_small_critter_contradicting_big_yolo_box_is_the_cat():
    # The reported bug: "Squirrel at the door! (guessed bear)". A red squirrel
    # cannot produce a bear-sized box — that blob is Cheeto.
    for wrong in (['bear'], ['cow'], ['elephant'], ['giraffe', 'horse']):
        for caption in ("A red squirrel sitting on the porch.",
                        "A small rabbit near the doorway."):
            reported, _, hedged = resolve_reported_classes(wrong, caption)
            assert reported == ['cat'], f"{wrong} + {caption!r} -> {reported}"
            assert not hedged


def test_real_squirrel_with_plausible_yolo_label_is_reported():
    # YOLO saying 'cat' on a squirrel is a normal, size-consistent mistake —
    # no contradiction, so the caption stands.
    caption = "A red squirrel is eating a nut on the railing."
    reported, _, _ = resolve_reported_classes(['cat'], caption)
    assert reported == ['squirrel']


def test_speculative_species_from_florence_becomes_reviewable_animal():
    # Production frames showed that an orange back/tail is repeatedly called a
    # fox. With no usable identity evidence, do not turn one caption into a
    # species claim; route it to the visitor review flow.
    caption = "A fox standing on the porch steps."
    reported, note, hedged = resolve_reported_classes(
        ['bear'], caption, 'florence', 'unknown')
    assert reported == ['animal']
    assert note and 'review' in note
    assert hedged


def test_speculative_species_with_confirmed_nonmatch_is_reported():
    caption = "A fox standing on the porch steps."
    reported, _, hedged = resolve_reported_classes(
        ['bear'], caption, 'florence', 'not_cheeto')
    assert reported == ['fox']
    assert hedged


def test_speculative_species_from_gemini_is_not_hedged():
    caption = "A fox standing on the porch steps."
    reported, _, hedged = resolve_reported_classes(['bear'], caption, 'gemini')
    assert reported == ['fox']
    assert not hedged


def test_cheeto_prototype_match_overrides_a_bad_caption():
    # The prototype is trained on THIS cat; a 230M captioner is not. When it
    # fires, nothing the caption claims about species should survive.
    for caption in ("A red squirrel on the porch.",
                    "A fox standing by the door.",
                    "A raccoon eating from a bowl."):
        reported, _, hedged = resolve_reported_classes(
            ['bear'], caption, 'florence', 'cheeto')
        assert reported == ['cat'], f"{caption!r} -> {reported}"
        assert not hedged


def test_cheeto_match_wins_even_with_no_caption():
    reported, note, hedged = resolve_reported_classes([], "", None, 'cheeto')
    assert reported == ['cat']
    assert note is None and not hedged


def test_not_cheeto_disables_the_small_critter_override():
    # Size contradiction says "that's the cat", but the prototype actively
    # disagreed — real evidence beats an inference about box size.
    reported, _, _ = resolve_reported_classes(
        ['bear'], "A red squirrel on the railing.", 'gemini', 'not_cheeto')
    assert reported == ['squirrel']


def test_not_cheeto_with_no_caption_is_flagged_uncertain():
    reported, note, hedged = resolve_reported_classes(
        ['cow'], "", None, 'not_cheeto')
    assert reported == ['cat']
    assert note and 'may not be' in note
    assert hedged


def test_unknown_verdict_keeps_confident_species_but_not_weak_wild_guess():
    assert resolve_reported_classes(
        ['cow'], "An orange cat by the door.", 'florence', 'unknown')[0] == ['cat']
    assert resolve_reported_classes(
        ['cat'], "A red fox on the porch.", 'florence', 'unknown')[0] == ['animal']


def test_confident_species_beats_speculative_in_same_caption():
    # "fox-like orange cat" is one cat, not a cat AND a fox.
    caption = "An orange cat with a bushy fox-like tail sits by the door."
    reported, _, hedged = resolve_reported_classes(['cow'], caption, 'florence')
    assert reported == ['cat']
    assert not hedged


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
    reported, note, hedged = resolve_reported_classes(['cat'], caption)
    assert reported == ['cat']
    assert note is None
    assert not hedged


def test_negation_still_suppresses():
    assert looks_like_false_positive("No cat is visible; the porch is empty.")
    assert should_suppress(['cat'], "No cat is visible; the porch is empty.")


def test_person_without_animal_suppresses():
    caption = "A man in a red jacket is standing at the door holding a package."
    assert should_suppress(['cat'], caption)


def test_caption_names_animal_yolo_has_no_class_for():
    caption = "A raccoon is eating from a bowl on the porch."
    reported, note, _ = resolve_reported_classes(['dog'], caption)
    assert reported == ['raccoon']
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
