#!/usr/bin/env python3
"""Build a Cheeto prototype from the detection history in Supabase.

Runs on Windows (or anywhere with the deps), NOT on the Pi. Output is a tiny
.npz — one vector plus a threshold — that gets copied to the Pi, where
cheeto_id.py uses it to answer "is this Cheeto?" in ~1-2s.

The wrinkle: upload_detection_to_supabase() never wrote the detected class,
so the rows are unlabeled. We can't just SELECT the cat pictures. So this
bootstraps labels instead:

  1. Pull every detection image and crop it to YOLO's box.
  2. Seed — CLIP zero-shot picks crops where "an orange tabby cat" beats a
     field of distractor prompts by a margin. High precision, low recall:
     we want a clean seed, not a complete one.
  3. Refine — average the seed into a prototype, re-score every crop against
     it, keep the top fraction, recompute. Image-to-image similarity is a
     much sharper signal than image-to-text, so this pulls in the awkward
     poses and night shots that step 2's text prompts missed.
  4. Calibrate a threshold from the kept distribution.
  5. Write a contact sheet so you can SEE what went into the prototype.

Step 5 matters. The prototype is only as good as the crops behind it, and a
handful of raccoons averaged in will quietly poison it. Look at the sheet
before you deploy.

Usage:
    python train_cheeto_prototype.py --limit 800
    python train_cheeto_prototype.py --limit 800 --skip-fetch   # reuse cache
"""

import argparse
import base64
import json
import os
import sys

import numpy as np
import PIL.Image

from cheeto_id import crop_animal, embed_images, DEFAULT_PAD_FRAC

# Prompts for the zero-shot seed. The distractors matter as much as the
# target: they're the exact species Florence keeps hallucinating, so a crop
# that beats all of them is a crop we're confident about.
TARGET_PROMPTS = [
    "a photo of an orange tabby cat",
    "a photo of a ginger cat",
    "an orange cat on a porch",
]
DISTRACTOR_PROMPTS = [
    "a photo of a red squirrel",
    "a photo of a fox",
    "a photo of a raccoon",
    "a photo of an opossum",
    "a photo of a rabbit",
    "a photo of a deer",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a skunk",
    "an empty porch with no animal",
    "a doormat on a porch",
    "a person standing at a door",
]


def fetch_detections(cache_dir, limit, page_size=100):
    """Page through the detections table, decode the base64 main_image column,
    and cache each frame as a JPEG. Returns a list of cached paths.

    Cached by row id, so re-runs are cheap and --skip-fetch works."""
    from supabase import create_client

    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY') or os.environ.get('SUPABASE_ANON_KEY')
    if not url or not key:
        sys.exit(
            "Set SUPABASE_URL and SUPABASE_KEY first.\n"
            "  PowerShell:  $env:SUPABASE_URL='https://xxx.supabase.co'\n"
            "               $env:SUPABASE_KEY='eyJ...'"
        )

    client = create_client(url, key)
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    offset = 0

    while len(paths) < limit:
        want = min(page_size, limit - len(paths))
        resp = (client.table('detections')
                .select('id,epoch,main_image')
                .order('epoch', desc=True)
                .range(offset, offset + want - 1)
                .execute())
        rows = resp.data or []
        if not rows:
            break

        for row in rows:
            b64 = row.get('main_image')
            if not b64:
                continue
            path = os.path.join(cache_dir, f"det_{row['id']}.jpg")
            if not os.path.exists(path):
                try:
                    with open(path, 'wb') as f:
                        f.write(base64.b64decode(b64))
                except Exception as e:
                    print(f"  skip row {row.get('id')}: {e}")
                    continue
            paths.append(path)

        offset += len(rows)
        print(f"  fetched {len(paths)} images...")

    return paths


def crop_all(paths, yolo_model, pad_frac, conf=0.25):
    """Run YOLO over each cached frame and crop to the highest-confidence
    animal box. Frames with no animal box are dropped — training on an empty
    porch would drag the prototype toward the doormat.

    Uses the same crop_animal() the Pi uses at inference, so training and
    inference embeddings come from the same distribution."""
    from caption_logic import ANIMAL_CLASSES

    crops, kept_paths = [], []
    for i, path in enumerate(paths):
        if i % 50 == 0:
            print(f"  cropping {i}/{len(paths)}...")
        try:
            image = PIL.Image.open(path).convert('RGB')
        except Exception:
            continue

        results = yolo_model(path, conf=conf, verbose=False)
        best_box, best_conf = None, 0.0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                name = yolo_model.names[int(box.cls[0])]
                score = float(box.conf[0])
                if name in ANIMAL_CLASSES and score > best_conf:
                    best_box = box.xyxy[0].cpu().numpy()
                    best_conf = score

        crop = crop_animal(image, best_box, pad_frac)
        if crop is not None:
            crops.append(crop)
            kept_paths.append(path)

    return crops, kept_paths


def zero_shot_seed(embeddings, model_name, pretrained, margin=0.02):
    """Pick crops where an orange-tabby prompt beats every distractor by
    `margin`. Text-vs-text comparison, so the modality gap cancels out and the
    scores are directly comparable."""
    import torch
    import open_clip
    from cheeto_id import load_clip

    model, _ = load_clip(model_name, pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    prompts = TARGET_PROMPTS + DISTRACTOR_PROMPTS
    with torch.inference_mode():
        text_feats = model.encode_text(tokenizer(prompts))
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy().astype(np.float32)

    sims = embeddings @ text_feats.T
    n_target = len(TARGET_PROMPTS)
    best_target = sims[:, :n_target].max(axis=1)
    best_distractor = sims[:, n_target:].max(axis=1)
    return (best_target - best_distractor) > margin, best_target - best_distractor


def build_prototype(embeddings, seed_mask, keep_frac=0.6, rounds=2):
    """Average the seed, then re-score and re-average against the prototype
    itself. Image-to-image similarity is sharper than image-to-text, so each
    round recovers true Cheeto crops the text prompts were too blunt to catch.

    keep_frac is a deliberate trade: keeping only the most prototypical 60%
    tightens the prototype and drops any stragglers the seed got wrong."""
    if seed_mask.sum() < 5:
        sys.exit(
            f"Only {int(seed_mask.sum())} seed images found — too few to train.\n"
            "Either the library has very few cat pictures, or the crops are bad. "
            "Try --limit higher, or lower --seed-margin."
        )

    proto = embeddings[seed_mask].mean(axis=0)
    proto /= np.linalg.norm(proto)

    for round_i in range(rounds):
        sims = embeddings @ proto
        n_keep = max(5, int(len(embeddings) * keep_frac))
        keep_idx = np.argsort(sims)[-n_keep:]
        proto = embeddings[keep_idx].mean(axis=0)
        proto /= np.linalg.norm(proto)
        print(f"  round {round_i + 1}: prototype from {n_keep} crops, "
              f"mean sim {sims[keep_idx].mean():.3f}")

    return proto, embeddings @ proto


def write_contact_sheet(path, kept_paths, sims, threshold, thumb_px=110):
    """A visual audit of the prototype: every crop, sorted by similarity, with
    the threshold marked. Scan it for raccoons above the line and Cheeto below
    it — those are the two failure modes that matter."""
    order = np.argsort(-sims)
    rows = []
    for idx in order:
        src = kept_paths[idx].replace('\\', '/')
        score = sims[idx]
        cls = 'in' if score >= threshold else 'out'
        rows.append(
            f'<figure class="{cls}"><img src="{src}" loading="lazy">'
            f'<figcaption>{score:.3f}</figcaption></figure>'
        )

    n_in = int((sims >= threshold).sum())
    html = f"""<!doctype html><meta charset="utf-8">
<title>Cheeto prototype — contact sheet</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee;
         margin:24px; }}
  p {{ color:#aaa; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:16px; }}
  figure {{ margin:0; width:{thumb_px}px; }}
  img {{ width:{thumb_px}px; height:{thumb_px}px; object-fit:cover;
         border-radius:4px; display:block; }}
  figcaption {{ font-size:11px; text-align:center; padding-top:2px; }}
  .in img {{ outline:2px solid #4ade80; }}
  .in figcaption {{ color:#4ade80; }}
  .out img {{ outline:2px solid #52525b; opacity:.55; }}
  .out figcaption {{ color:#71717a; }}
</style>
<h1>Cheeto prototype — contact sheet</h1>
<p>{n_in} of {len(sims)} crops score at or above the
threshold ({threshold:.3f}) and count as Cheeto.
<strong>Green</strong> = would be called Cheeto, <strong>grey</strong> = would not.</p>
<p>Look for two things: any non-cat critter sitting in the green band, and any
obvious Cheeto stranded in the grey. A few grey Cheetos are fine and expected
&mdash; the threshold is deliberately strict.</p>
<div class="grid">{''.join(rows)}</div>
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=600,
                    help='max detections to pull from Supabase')
    ap.add_argument('--cache-dir', default='.cheeto_cache')
    ap.add_argument('--out', default='cheeto_prototype.npz')
    ap.add_argument('--contact-sheet', default='cheeto_contact_sheet.html')
    # Name the QuickGELU variant EXPLICITLY. OpenAI's weights were trained
    # with QuickGELU activations; older open_clip silently applied it to the
    # bare 'ViT-B-32' name while newer versions do not, only warning. Two
    # machines on different open_clip versions therefore build two different
    # architectures from the same name, embed into two different spaces, and
    # produce confident, meaningless similarity scores. The explicit name
    # pins the same architecture everywhere.
    ap.add_argument('--model', default='ViT-B-32-quickgelu')
    ap.add_argument('--pretrained', default='openai')
    ap.add_argument('--yolo', default='yolo11n.pt')
    ap.add_argument('--pad-frac', type=float, default=DEFAULT_PAD_FRAC)
    ap.add_argument('--seed-margin', type=float, default=0.02)
    ap.add_argument('--keep-frac', type=float, default=0.6)
    ap.add_argument('--percentile', type=float, default=5.0,
                    help='threshold percentile over kept crops; lower is more '
                         'permissive (more things called Cheeto). Only used '
                         'when --threshold is not given.')
    ap.add_argument('--threshold', type=float,
                    help='set the decision threshold explicitly, overriding '
                         '--percentile. Prefer this once you have looked at '
                         'the contact sheet: the percentile is circular (the '
                         'prototype defines the kept set, so the percentile '
                         'just carves out a fixed fraction of it) and is not '
                         'anchored to any real Cheeto/not-Cheeto boundary. '
                         'Pick a value above the best-scoring NON-Cheeto '
                         'animal you can find in the sheet.')
    ap.add_argument('--skip-fetch', action='store_true',
                    help='reuse whatever is already in --cache-dir')
    ap.add_argument('--local-dir',
                    help='train from a folder of JPEGs instead of Supabase '
                         '(e.g. detection frames copied off the Pi)')
    args = ap.parse_args()

    if args.local_dir:
        paths = []
        for root, _, files in os.walk(args.local_dir):
            for name in sorted(files):
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(root, name))
        paths = paths[:args.limit]
        print(f"Using {len(paths)} images from {args.local_dir}.")
    elif args.skip_fetch:
        paths = sorted(
            os.path.join(args.cache_dir, f)
            for f in os.listdir(args.cache_dir) if f.endswith('.jpg')
        )[:args.limit]
        print(f"Using {len(paths)} cached images.")
    else:
        print("Fetching detections from Supabase...")
        paths = fetch_detections(args.cache_dir, args.limit)
        print(f"Fetched {len(paths)} images.")

    if len(paths) < 20:
        sys.exit(f"Only {len(paths)} images — need at least ~20 to train.")

    print("Loading YOLO and cropping to animal boxes...")
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo)
    crops, kept_paths = crop_all(paths, yolo_model, args.pad_frac)
    print(f"Got {len(crops)} animal crops from {len(paths)} frames.")
    if len(crops) < 20:
        sys.exit("Too few crops. Is --yolo pointing at the right weights?")

    print("Embedding crops with CLIP...")
    embeddings = embed_images(crops, args.model, args.pretrained)

    print("Seeding with zero-shot prompts...")
    seed_mask, margins = zero_shot_seed(
        embeddings, args.model, args.pretrained, args.seed_margin)
    print(f"  {int(seed_mask.sum())} seed crops "
          f"(margin range {margins.min():.3f} to {margins.max():.3f})")

    print("Refining prototype...")
    proto, sims = build_prototype(
        embeddings, seed_mask, args.keep_frac)

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        kept = sims[np.argsort(sims)[-max(5, int(len(sims) * args.keep_frac)):]]
        threshold = float(np.percentile(kept, args.percentile))

    # The score distribution is bimodal in practice: a low mode of YOLO false
    # positives (leaves, snow, doormats, someone's legs — no animal at all)
    # and a high mode of real cat. Warn when the threshold lands inside the
    # high mode, which means it's rejecting genuine Cheeto rather than
    # separating anything.
    hist, edges = np.histogram(sims, bins=24)
    valley = int(np.argmin(hist[2:-2])) + 2
    valley_score = float(edges[valley])
    if threshold > valley_score:
        upper = sims[sims > valley_score]
        if len(upper) and threshold > np.percentile(upper, 10):
            print(f"\n  NOTE: threshold {threshold:.3f} sits well inside the "
                  f"upper mode (valley near {valley_score:.3f}). It is "
                  f"rejecting real Cheeto, not just false positives. That is "
                  f"a safe direction to err — rejected frames fall back to "
                  f"caption logic — but check the contact sheet.")

    np.savez(
        args.out,
        prototype=proto.astype(np.float32),
        threshold=threshold,
        model_name=args.model,
        pretrained=args.pretrained,
        pad_frac=args.pad_frac,
        n_images=len(crops),
    )

    write_contact_sheet(args.contact_sheet, kept_paths, sims, threshold)

    above = int((sims >= threshold).sum())
    print(f"\nWrote {args.out} (threshold {threshold:.3f}, "
          f"from {len(crops)} crops)")
    print(f"  {above}/{len(sims)} crops would be called Cheeto "
          f"({100 * above / len(sims):.0f}%)")
    print(f"  similarity: min {sims.min():.3f}  median "
          f"{np.median(sims):.3f}  max {sims.max():.3f}")
    print(f"\nOpen {args.contact_sheet} and check the green band before "
          f"deploying.")


if __name__ == '__main__':
    main()
