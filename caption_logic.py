#!/usr/bin/env python3
"""Pure decision logic for turning YOLO labels + a VLM caption into what the
detector actually reports. No camera/model/network dependencies, so it can be
unit tested off the Pi — see test_caption_logic.py."""

import re

# Animals that can actually show up at the door — reported as-is when the
# caption doesn't say otherwise.
PLAUSIBLE_ANIMAL_CLASSES = ['cat', 'dog', 'bird']

# COCO classes the nano model commonly MISTAKES the cat for (a "giraffe" on a
# Michigan porch is the cat). Still detected so a mislabeled cat can't slip
# through unalerted, but never reported verbatim — the VLM caption decides
# what to call it, and with no caption we hedge to 'cat'.
LOOKALIKE_ANIMAL_CLASSES = ['horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

ANIMAL_CLASSES = PLAUSIBLE_ANIMAL_CLASSES + LOOKALIKE_ANIMAL_CLASSES

_ANIMAL_VOCAB = [
    'cat', 'cats', 'kitten', 'feline', 'tabby', 'kitty',
    'dog', 'dogs', 'puppy', 'canine', 'hound', 'pup',
    'bird', 'birds', 'sparrow', 'robin', 'crow', 'pigeon', 'hawk', 'turkey',
    'horse', 'sheep', 'cow', 'bear', 'deer', 'rabbit', 'bunny', 'squirrel',
    'raccoon', 'possum', 'opossum', 'fox', 'skunk', 'chipmunk', 'coyote',
    'groundhog', 'woodchuck', 'mouse', 'rat',
    'animal', 'animals', 'pet', 'pets', 'creature', 'creatures',
    'paw', 'paws', 'fur', 'whisker', 'whiskers', 'tail',
]

# Caption words → the animal we report. Checked in order; every match is
# included. Lets the VLM overrule YOLO's guess (YOLO says "cow", caption says
# "an orange cat" → we report a cat) and name critters YOLO has no class for
# (raccoon, possum, fox, ...).
_CAPTION_ANIMAL_MAP = [
    ('cat', ['cat', 'cats', 'kitten', 'feline', 'tabby', 'kitty']),
    ('dog', ['dog', 'dogs', 'puppy', 'pup', 'hound']),
    ('raccoon', ['raccoon']),
    ('opossum', ['opossum', 'possum']),
    ('fox', ['fox']),
    ('skunk', ['skunk']),
    ('squirrel', ['squirrel', 'chipmunk']),
    ('rabbit', ['rabbit', 'bunny']),
    ('deer', ['deer']),
    ('bird', ['bird', 'birds', 'sparrow', 'robin', 'crow', 'pigeon', 'hawk', 'turkey']),
]


def animals_in_caption(description):
    """Animals the VLM actually named in its caption, in
    _CAPTION_ANIMAL_MAP priority order."""
    if not description:
        return []
    text = description.lower()
    found = []
    for name, words in _CAPTION_ANIMAL_MAP:
        if any(re.search(r'\b' + w + r'\b', text) for w in words):
            found.append(name)
    return found


def looks_like_false_positive(description):
    """Return True if the VLM description suggests the YOLO detection was a
    false positive. Two signals:
    1. Explicit negation phrases ("no animal", "empty doorway").
    2. Description focuses on a person / objects but mentions NO
       animal-related vocabulary. This catches the common case where YOLO
       hallucinates a cat from a person's shadow or a phone screen."""
    if not description:
        return False
    text = description.lower()

    negation_patterns = [
        r"\bno animal[s]?\b",
        r"\bno cat[s]?\b",
        r"\bno dog[s]?\b",
        r"\bno pet[s]?\b",
        r"\bno one\b",
        r"\bis empty\b",
        r"\bappears (to be )?empty\b",
        r"\bempty (doorway|view|frame|scene|porch|room|area|space)\b",
        r"\bnothing (is )?(present|visible|in (the |this )?(view|frame|image))\b",
        r"\bnot visible\b",
        r"\bno (animal|cat|dog|pet|one)s? (is|are) (present|visible|in)\b",
    ]
    if any(re.search(p, text) for p in negation_patterns):
        return True

    # Signal 2: description mentions a person (or is clearly about objects)
    # but doesn't mention any animal vocab at all.
    has_animal_word = any(re.search(r'\b' + w + r'\b', text) for w in _ANIMAL_VOCAB)
    if not has_animal_word:
        person_markers = re.search(
            r'\b(person|man|woman|people|human|child|girl|boy)\b', text
        )
        if person_markers:
            return True

    return False


def should_suppress(detected_classes, description):
    """Decide whether to drop the detection as a false positive, based on the
    VLM caption. True = suppress the whole alert.

    Beyond looks_like_false_positive: a non-trivial caption that names no
    animal vocabulary at all means the VLM described the scene and saw no
    critter — YOLO hallucinated one from shadows or the doormat pattern. A
    real animal at the door is the salient subject, so DETAILED_CAPTION
    reliably mentions it. An empty/failed caption never suppresses."""
    if looks_like_false_positive(description):
        return True
    if description and len(description) >= 20:
        text = description.lower()
        if not any(re.search(r'\b' + w + r'\b', text) for w in _ANIMAL_VOCAB):
            return True
    return False


def resolve_reported_classes(detected_classes, description):
    """Decide what animal to REPORT, given YOLO's labels and the VLM caption.

    The caption wins when it names an animal. Otherwise plausible YOLO labels
    are reported as-is and lookalike-only labels are hedged to 'cat'.
    Returns (classes_to_report, note_or_None); the note is a short
    transparency line for the notification body when we overrode YOLO."""
    detected_unique = list(dict.fromkeys(detected_classes))
    caption_animals = animals_in_caption(description)
    if caption_animals:
        if set(caption_animals) != set(detected_unique):
            return caption_animals, f"(detector guessed: {', '.join(detected_unique)})"
        return caption_animals, None

    plausible = [c for c in detected_unique if c in PLAUSIBLE_ANIMAL_CLASSES]
    if plausible:
        return plausible, None

    # Only lookalike labels and no usable caption: almost certainly the cat.
    return ['cat'], f"(unverified — detector guessed {', '.join(detected_unique)})"


def build_confirmation_prompt(detected_classes):
    """Structured prompt that returns a parseable VISIBLE/DESCRIPTION block.

    Asks about ANY animal, not just the detector's guess — the detector often
    mislabels the cat as cow/elephant/etc, and "is a cow visible? no" must
    not suppress a real cat."""
    classes_str = ", ".join(detected_classes)
    return (
        "An object detector on a porch camera thinks it saw an animal "
        f"(its guess: {classes_str}), but it often mislabels animals.\n\n"
        "Look at the image and respond in exactly this format with no extra "
        "commentary:\n"
        "VISIBLE: yes  (if ANY animal is visible; otherwise: no)\n"
        "DESCRIPTION: One or two short sentences. If an animal is visible, "
        "say what kind it is, describe it (color/markings), and what it's "
        "doing (sitting, walking, eating, etc.). If not, briefly say what "
        "the image actually shows."
    )


def parse_gemini_confirmation(response_text, detected_classes):
    """Return (confirmed: bool, description: str).

    Prefers the structured VISIBLE/DESCRIPTION block. Falls back to keyword
    matching when Gemini returns free-form text.
    """
    if not response_text:
        return False, ""
    text = response_text.strip()
    description = text

    visible_match = re.search(r"VISIBLE\s*:\s*(yes|no)\b", text, re.IGNORECASE)
    desc_match = re.search(r"DESCRIPTION\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if desc_match:
        description = desc_match.group(1).strip()

    if visible_match:
        return visible_match.group(1).lower() == "yes", description

    # Fallback: free-form text — use the keyword/negation heuristic.
    lowered = text.lower()
    negation_patterns = [
        r"\bdon'?t see\b",
        r"\bdo not see\b",
        r"\bcannot see\b",
        r"\bcan'?t see\b",
        r"\bis not visible\b",
        r"\bnot visible\b",
    ]
    if any(re.search(p, lowered) for p in negation_patterns):
        return False, description
    for cls in detected_classes:
        if re.search(r'\b' + re.escape(cls.lower()) + r'\b', lowered):
            return True, description
    # YOLO's label may be wrong while the animal is real ("I see a cat", but
    # the detector guessed cow) — any named animal counts as confirmation.
    if animals_in_caption(lowered):
        return True, description
    return False, description
