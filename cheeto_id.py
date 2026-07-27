#!/usr/bin/env python3
"""Is this Cheeto, or is it some other critter?

A 230M captioner guesses at species — that's how an orange tabby becomes a
"red squirrel". This module answers the narrower question the captioner keeps
getting wrong, using the one thing no general model has: a few hundred photos
of *this* cat, on *this* porch, from *this* camera.

Method is few-shot prototype matching. train_cheeto_prototype.py embeds the
Cheeto library with CLIP and averages it into a single unit vector — the
"prototype". Here we embed the animal crop and take a cosine similarity
against it. Above the calibrated threshold, it's Cheeto.

Deliberately NOT a comparison against CLIP *text* embeddings ("a photo of a
raccoon"). Image and text embeddings occupy the same CLIP space but with a
well-known modality gap, so an image-to-image score and an image-to-text
score aren't on the same scale and can't be argmax'd together. Comparing
image to image only, against one calibrated threshold, sidesteps that
entirely.

Runs on the Pi in ~1-2s per detection (ViT-B-32, CPU). Captioning is
untouched — Florence still writes the sentence, this only decides the animal.
"""

import logging
import os

import numpy as np
import PIL.Image

# Padding around YOLO's box before cropping. A tight box clips ears and tail;
# some context helps CLIP. Must match between training and inference or the
# embeddings come from different distributions and the prototype won't match.
DEFAULT_PAD_FRAC = 0.15

# Minimum crop size in pixels. Smaller than this and there aren't enough
# pixels for a meaningful embedding — a distant bird, say.
MIN_CROP_PX = 48

_MODEL = None
_PREPROCESS = None
_MODEL_KEY = None


def crop_animal(image, bbox, pad_frac=DEFAULT_PAD_FRAC):
    """Crop a PIL image to a padded YOLO bbox. bbox is [x1, y1, x2, y2] in
    pixels. Returns None if the box is degenerate or too small to be useful."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width, x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)

    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
        return None
    return image.crop((x1, y1, x2, y2))


def crop_with_context(image, bbox, pad_frac=0.5, min_frac=0.4):
    """Crop around a bbox while keeping surrounding context, or return the
    image unchanged when there's no usable box.

    Unlike crop_animal (which crops tight, for embedding), this keeps enough
    scene for a captioner to say what the animal is DOING and where. Two
    knobs pulling opposite ways: pad_frac widens around the box, min_frac is
    a floor on the result as a fraction of each frame dimension so a distant
    animal's tiny box doesn't become a handful of pixels.

    Crops that would overflow an edge are SHIFTED inward rather than
    shrunk, so an animal at the frame edge still yields a full-size crop
    instead of a clipped sliver."""
    if bbox is None:
        return image
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return image

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # Full width/height of the crop, floored at min_frac of the frame and
    # capped at the frame itself.
    out_w = min(float(image.width),
                max((x2 - x1) * (1.0 + pad_frac), image.width * min_frac))
    out_h = min(float(image.height),
                max((y2 - y1) * (1.0 + pad_frac), image.height * min_frac))

    left = min(max(0.0, cx - out_w / 2.0), image.width - out_w)
    top = min(max(0.0, cy - out_h / 2.0), image.height - out_h)
    return image.crop((int(round(left)), int(round(top)),
                       int(round(left + out_w)), int(round(top + out_h))))


def load_clip(model_name='ViT-B-32-quickgelu', pretrained='openai',
              num_threads=None):
    """Load and cache the CLIP image encoder. Idempotent per (model,
    pretrained) pair, so the Pi loads the ~150MB of weights once and keeps
    them resident alongside Florence."""
    global _MODEL, _PREPROCESS, _MODEL_KEY
    key = (model_name, pretrained)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL, _PREPROCESS

    import torch
    import open_clip

    if num_threads:
        torch.set_num_threads(num_threads)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model.eval()

    _MODEL, _PREPROCESS, _MODEL_KEY = model, preprocess, key
    return model, preprocess


def embed_images(images, model_name='ViT-B-32-quickgelu', pretrained='openai',
                 batch_size=16, num_threads=None):
    """Embed PIL images into L2-normalized CLIP vectors, shape (N, D).
    Normalizing here means cosine similarity is a plain dot product."""
    import torch

    if not images:
        return np.zeros((0, 512), dtype=np.float32)

    model, preprocess = load_clip(model_name, pretrained, num_threads)
    out = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        tensors = torch.stack([preprocess(im.convert('RGB')) for im in batch])
        with torch.inference_mode():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def load_prototype(path):
    """Load a prototype produced by train_cheeto_prototype.py. Returns a dict,
    or None if the file is missing — a missing prototype must degrade to the
    old caption-only behaviour, never crash the detector."""
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        proto = data['prototype'].astype(np.float32)
        # Stored normalized, but re-normalize defensively: a non-unit vector
        # silently rescales every similarity and invalidates the threshold.
        norm = np.linalg.norm(proto)
        if norm == 0:
            logging.error(f"Prototype in {path} is a zero vector — ignoring.")
            return None
        model_name = str(data['model_name'])
        # 'ViT-B-32' + 'openai' is ambiguous across open_clip versions: old
        # ones silently apply QuickGELU, new ones don't and only warn. A
        # prototype trained under one and queried under the other lands in a
        # different embedding space and every score is quietly wrong, with no
        # error anywhere. Refuse to guess.
        if model_name == 'ViT-B-32' and str(data['pretrained']) == 'openai':
            logging.error(
                f"Prototype in {path} names the ambiguous 'ViT-B-32'+'openai' "
                "pair, whose activation function differs between open_clip "
                "versions. Retrain with --model ViT-B-32-quickgelu. Ignoring "
                "the prototype rather than scoring against the wrong space."
            )
            return None

        return {
            'prototype': proto / norm,
            'threshold': float(data['threshold']),
            'model_name': str(data['model_name']),
            'pretrained': str(data['pretrained']),
            'pad_frac': float(data['pad_frac']),
            'n_images': int(data['n_images']),
        }
    except Exception as e:
        logging.error(f"Could not load Cheeto prototype from {path}: {e}")
        return None


def identify(image, bbox, prototype, num_threads=None):
    """Score one detection against the Cheeto prototype.

    Returns (verdict, score) where verdict is:
      'cheeto'     — matches the trained cat above threshold
      'not_cheeto' — a confident non-match; the caption's species is credible
      'unknown'    — no prototype, no usable crop, or an embedding failure;
                     the caller should fall back to caption-only logic

    'unknown' is the safe default for every error path. Misidentifying a real
    raccoon as the cat is worse than not answering."""
    if prototype is None:
        return 'unknown', 0.0

    try:
        if isinstance(image, str):
            image = PIL.Image.open(image)
        crop = crop_animal(image, bbox, prototype['pad_frac'])
        if crop is None:
            return 'unknown', 0.0

        vec = embed_images(
            [crop],
            model_name=prototype['model_name'],
            pretrained=prototype['pretrained'],
            num_threads=num_threads,
        )
        if vec.shape[0] == 0:
            return 'unknown', 0.0

        score = float(np.dot(vec[0], prototype['prototype']))
        verdict = 'cheeto' if score >= prototype['threshold'] else 'not_cheeto'
        return verdict, score
    except Exception as e:
        logging.error(f"Cheeto identification failed: {e}")
        return 'unknown', 0.0
