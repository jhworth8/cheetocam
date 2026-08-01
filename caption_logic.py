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

# Fine-grained wild species. Florence-2-base (230M) reliably tells cat from
# dog from bird, but on anything beyond that its species name is a guess, and
# on a dark porch frame an orange tabby is its favourite fox/squirrel. These
# are still reported — a raccoon really does visit — but never asserted
# flatly; the caller hedges the wording when the caption came from the local
# model, so they are always presented as uncertain unless pet recognition
# independently identifies the visitor.
SPECULATIVE_CAPTION_ANIMALS = [
    'raccoon', 'opossum', 'fox', 'skunk', 'squirrel', 'rabbit', 'deer',
]

# Critters far smaller than a cat. Every LOOKALIKE class is a large quadruped,
# so for the nano model to fire one it had to see a big animal-shaped blob
# filling real frame area — which a squirrel or a rabbit physically cannot
# produce at porch distance. "YOLO: bear" + "caption: red squirrel" is
# therefore a size contradiction, and the only thing that makes that box is
# the cat.
SMALL_CRITTERS = ['squirrel', 'rabbit']

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


def resolve_reported_classes(detected_classes, description, caption_source=None,
                             cheeto_verdict=None):
    """Decide what animal to REPORT, given YOLO's labels and the VLM caption.

    The caption wins when it names an animal, but not blindly: a weak local
    caption naming a fine-grained wild species is treated as a guess, and a
    guess that contradicts the size of YOLO's box is overruled outright.

    caption_source is 'florence' (the local model) or None. cheeto_verdict is
    'cheeto' / 'not_cheeto' / 'unknown' / None from
    cheeto_id.identify(). Returns (classes_to_report, note_or_None, hedged).
      note   — short transparency line for the notification body, or None.
               Deliberately never repeats YOLO's raw wrong label; that's
               noise to the reader and lives in the log instead.
      hedged — True when the species is a guess, so the caller should say
               "Possible fox at the door" rather than "Fox at the door"."""
    detected_unique = list(dict.fromkeys(detected_classes))
    caption_animals = animals_in_caption(description)

    # A prototype match outranks everything. It's the only signal here trained
    # on THIS cat rather than on cats in general, so when it fires we don't
    # care what a 230M captioner thought it saw. The caption still ships in
    # the notification body — it's the species claim we're overriding, not
    # the description.
    if cheeto_verdict == 'cheeto':
        return ['cat'], None, False

    if caption_animals:
        # A caption naming both a confident and a speculative species ("an
        # orange cat, almost fox-like") is one animal, not two — the species
        # the model actually knows wins.
        confident = [a for a in caption_animals
                     if a not in SPECULATIVE_CAPTION_ANIMALS]
        if confident:
            return confident, None, False

        # Size contradiction: only large-quadruped labels from YOLO, but the
        # caption named a critter far too small to have produced that box.
        # Skipped when the prototype actively says this ISN'T the cat —
        # that's better evidence than an inference about box size.
        if (cheeto_verdict != 'not_cheeto'
                and detected_unique
                and all(c in LOOKALIKE_ANIMAL_CLASSES for c in detected_unique)
                and any(a in SMALL_CRITTERS for a in caption_animals)):
            return ['cat'], None, False

        # A partial orange back or tail is exactly where Florence calls Cheeto
        # a fox/squirrel. When the identity model could not get a usable crop,
        # one weak caption must not invent a species. Save it as an unknown
        # animal for the review/training flow instead. A confirmed identity
        # non-match may still name the visitor, but the local species guess is
        # kept visibly uncertain.
        if (caption_source == 'florence'
                and cheeto_verdict in (None, 'unknown')):
            return ['animal'], "(identity uncertain — saved for review)", True

        # A wild species with independent evidence that it is not Cheeto.
        return caption_animals, None, True

    plausible = [c for c in detected_unique if c in PLAUSIBLE_ANIMAL_CLASSES]
    if plausible:
        return plausible, None, False

    # Only lookalike labels and no usable caption. Normally that's the cat,
    # but if the prototype disagreed we say so rather than quietly asserting
    # a cat the classifier just rejected.
    if cheeto_verdict == 'not_cheeto':
        return ['cat'], "(unverified — may not be Cheeto)", True
    return ['cat'], "(unverified — no clear look at the animal)", False
