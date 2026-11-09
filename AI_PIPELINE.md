# How Critter Cam actually works

A technical walkthrough of the whole system, with emphasis on the AI: what
detects, what identifies, what describes, how the models are trained, and how
the pieces disagree with each other on purpose.

Everything here is local. No frame, crop or caption is sent to a cloud model.
The three models run on one Raspberry Pi 5.

---

## 1. The shape of the system

Three repositories, three tiers, and a deploy order that is always
**database → detector → clients**.

| Repo | Runs on | Job |
|---|---|---|
| `cheeto-infra` | the Cheeto box (`192.168.86.113`) | Postgres/Supabase, migrations, device firmware, docs |
| `cheetocam` | the Pi (`192.168.86.51`) | capture, detection, identity, captioning, live view, rewind |
| `cheetocam-app` | phones | Flutter client for iOS and Android |

One detector **process per camera**, as systemd units: `cat_detector`,
`cat_detector@xiao-camera`, `cat_detector@amb82-camera`. Each reads its own
`~/cheetocam/.env.<camera-id>`. This is deliberate isolation — a wireless
camera that stops answering cannot stall the front-door webcam, because it is
a different process with a different `/dev/video0` handle.

Each unit gets equal systemd CPU weight and a **150% CPU ceiling**, leaving one
core of headroom on the four-core Pi when two feeds are busy.

---

## 2. Capture, before any AI runs

A frame has to be worth having before anything expensive touches it.

**Camera sources** (`camera_source.py`) — two shapes behind one interface:
- `cv2.VideoCapture` for USB webcams and MJPEG streams.
- `HttpSnapshotCapture` for embedded cameras whose one-shot JPEG endpoint is
  more reliable than their MJPEG framing (both ESP32 cameras). It polls at
  `CAMERA_SNAPSHOT_INTERVAL` (default 0.2s), decodes, and applies rotation.

**Rotation** is applied at capture, not at display, so saved photos, detection
and live view all straighten together. It is a database column
(`camera_status.rotation_degrees`) that the detector re-reads on the settings
cadence, so repositioning a camera never needs an SSH session.

**Three gates** decide whether to spend anything (`capture_gate.py` — pure
functions, unit tested, no camera or model imports):

- `should_capture(...)` — is it worth asking the camera for a frame at all? A
  motion-gated camera idles until its sensor fires, but a live viewer always
  counts.
- `should_detect(...)` — may this frame become an observation? Only the sensor
  opens this gate. A live view or a scheduled wake must never be able to
  manufacture a visit out of a passing shadow.
- `capture_interval(...)` — battery saver. Slows a quiet snapshot camera's poll
  from 0.2s to 1.0s, and restores full rate for a live viewer, for any sighting
  (even unconfirmed) for 20s, or when the base interval is already slower.

---

## 3. Stage one — detection (YOLO)

**Model:** `yolo11n.pt` (Ultralytics YOLO11 nano). Falls back to `yolov8n.pt`,
then to YOLOv3-tiny via OpenCV DNN with the weights in `yolo/`. The fallbacks
exist so a bad model download degrades the system rather than stopping it.

**Confidence:** `DETECTION_CONFIDENCE`, default **0.7**.

**Classes.** This is the first non-obvious decision. The detector does not just
look for `cat`; it watches a deliberately widened list from `caption_logic.py`:

```python
PLAUSIBLE_ANIMAL_CLASSES  = ['cat', 'dog', 'bird']
LOOKALIKE_ANIMAL_CLASSES  = ['horse','sheep','cow','elephant','bear','zebra','giraffe']
IGNORED_ANIMALS           = ['bird']            # env IGNORE_ANIMALS
ANIMAL_CLASSES            = (PLAUSIBLE + LOOKALIKE) - IGNORED
```

A nano model on a dark Michigan porch will label the cat a "giraffe". Those
lookalike classes are therefore **detected but never reported verbatim** —
they exist so a mislabeled cat cannot slip past unalerted. What the animal is
actually called gets decided later, in stage four.

**Ignored species.** `IGNORE_ANIMALS` (default `bird`) subtracts from that
list before YOLO ever sees it, which is the only place a species can be
dropped for free: no box, no confirmation, no capture, no GIF, no ~30s of
Florence, no CLIP. A feeder or a hedge full of sparrows would otherwise run
the whole pipeline all day on a Pi that §3 already notes is CPU-bound.
Blank the variable to alert on everything again. Stage four carries the other
half of this rule.

**Confirmation.** A single frame with a box is not a detection.
`CONFIRM_FRAMES` (default 2) consecutive frames must contain a target class
before an observation fires. This kills single-frame hallucinations.

**Settling.** Once confirmed, the detector drains `POST_DETECTION_SETTLE_FRAMES`
(default 10) frames before saving the photo. The "detection moment" frame is
often dark because USB auto-exposure was still adjusting when YOLO fired.

**Cadence.** After a saved observation, `next_observation_time` is pushed out by
`OBSERVATION_INTERVAL_SECONDS` (default 60). Note the consequence: when nothing
is detected, that timer is never advanced, so **YOLO runs on every captured
frame indefinitely**. That is why the front-door detector sits pinned at its
150% quota — measured 149.2% steady, and the loop wants 5fps but achieves 2fps
because it is CPU-bound on exactly this.

---

## 4. Stage two — identity (CLIP few-shot prototypes)

YOLO says *"there is a cat"*. It cannot say *"that is Cheeto"*. A 230M
captioner is worse than useless here — it is how an orange tabby becomes a "red
squirrel". So identity is a separate, narrower question answered with the one
thing no general model has: a few hundred photos of **this** cat, on **this**
porch, from **this** camera.

**Method: few-shot prototype matching.**

1. Crop the animal from YOLO's box with `crop_animal()`, padded by
   `pad_frac` (default 0.15). A tight box clips ears and tail.
2. Embed the crop with CLIP `ViT-B-32-quickgelu` / `openai`, L2-normalized.
3. Cosine-similarity it against a stored prototype vector — which, because both
   sides are unit vectors, is a plain dot product.
4. Above the calibrated threshold, it's a match.

Three details that matter more than they look:

**Image-to-image only, never image-to-text.** CLIP's image and text embeddings
share a space but with a well-known *modality gap*. An image-to-image score and
an image-to-text score are not on the same scale and cannot be argmax'd
together. Comparing image to image against one calibrated threshold sidesteps
that entirely.

**`pad_frac` must match between training and inference**, or the embeddings come
from different distributions and the prototype quietly stops matching.

**The model name is pinned to `ViT-B-32-quickgelu`, explicitly.** OpenAI's
weights were trained with QuickGELU activations. Older `open_clip` silently
applied it to the bare name `ViT-B-32`; newer versions do not and only warn. Two
machines on different versions therefore build *different architectures from the
same name*, embed into different spaces, and produce confident, meaningless
scores with no error anywhere. `load_prototype()` refuses to load a prototype
that names the ambiguous pair rather than guess.

Every error path returns `'unknown'`, not a guess. Misidentifying a real raccoon
as the cat is worse than not answering.

---

## 5. Two kinds of trained model

There are **two** identity systems, and the second supersedes the first.

### 5a. The Cheeto prototype — offline, bootstrapped

`train_cheeto_prototype.py` runs on a desktop, not the Pi, and emits a tiny
`.npz`: one vector plus a threshold.

The problem it solves: the detections table never recorded the detected class,
so the history is **unlabeled**. You cannot simply `SELECT` the cat pictures. So
it bootstraps its own labels:

1. **Fetch** — page through `detections`, decode the base64 image column, cache
   each frame.
2. **Crop** — run YOLO over each frame, keep the highest-confidence animal box,
   crop with the same `crop_animal()` the Pi uses. Frames with no animal are
   dropped; training on an empty porch drags the prototype toward the doormat.
3. **Seed (zero-shot)** — CLIP scores each crop against target prompts
   (*"a photo of an orange tabby cat"*) and a field of distractors — squirrel,
   fox, raccoon, opossum, "an empty porch with no animal". A crop is seeded only
   if the target beats *every* distractor by a margin. This is text-vs-text, so
   the modality gap cancels. High precision, low recall on purpose: a clean
   seed, not a complete one. The distractor list is literally the species
   Florence keeps hallucinating.
4. **Refine** — average the seed into a prototype, re-score every crop against
   it, keep the top 60%, recompute. Twice. Image-to-image is a much sharper
   signal than image-to-text, so each round recovers the awkward poses and night
   shots the text prompts were too blunt to catch.
5. **Calibrate** — take the threshold as a percentile (default 5th) of the kept
   distribution.
6. **Contact sheet** — write an HTML page of every crop, sorted by similarity,
   with the threshold marked in green and grey.

Step 6 is not decoration. The prototype is only as good as the crops behind it,
and a handful of raccoons averaged in will quietly poison it. You look for two
things: a non-cat in the green band, and an obvious Cheeto stranded in grey.

### 5b. Learned visitor profiles — online, from your taps

The better system. When you confirm or correct a visitor in the app, that
review becomes training data, and the Pi rebuilds prototypes from it every
`LEARNED_PROFILE_REFRESH_SECONDS` (default 300).

The database stores **only references** — `animal_profile_samples` rows joining
an `animal_profiles` row to a `detections` row, exposed through the
`profile_training_samples(household_id)` RPC. Embeddings and prototypes never
leave the Pi.

Building a profile (`get_learned_profiles()`), per animal, up to 24 samples:

1. Download each reviewed sample, crop to its stored `bbox`. Rows that predate
   stored boxes get their box recovered locally by re-running YOLO — otherwise
   tapping "This is Cheeto" on an old photo would silently produce nothing.
2. Embed all crops.
3. **Prune** — compute a rough centroid, keep the 80% most consistent crops
   (minimum 3). One mislabeled empty doormat must not drag Cheeto toward Rocky
   forever.
4. **Centroid** — mean of the kept vectors, normalized.
5. **Exemplars** — greedily pick up to 6 *deliberately dissimilar* kept crops:
   start with the most representative, then repeatedly add whichever crop is
   least like the ones already chosen. These preserve distinct viewpoints
   (side, front, back) that a single average erases.

**Cache invalidation** is by SHA-256 signature over the `(animal_id,
detection_id)` set, so an unchanged review set costs one RPC and no embedding
work.

**Matching** (`match_learned_profiles`) scores each profile as:

```
score = 0.60 × cosine(crop, centroid) + 0.40 × mean(top-2 exemplar cosines)
```

The centroid is stable; the exemplars keep viewpoint detail. Averaging the best
*two* exemplars stops one lucky historical crop winning on its own.

A match is only returned if it clears **both**:
- an absolute threshold (`LEARNED_PROFILE_THRESHOLD`, default **0.90**), and
- a **margin** over the runner-up (`LEARNED_PROFILE_MIN_MARGIN`, default
  **0.015**).

That margin rule is the whole point. Two similar orange cats stay *unknown*
rather than being confidently swapped for each other.

One usable owner-confirmed crop is allowed to participate, at a stricter
threshold (≥0.95). Excluding a brand-new visitor entirely would leave only
Cheeto in the comparison set, making the classifier structurally incapable of
ever choosing the new animal.

---

## 6. Stage three — captioning (Florence-2)

**Model:** `microsoft/Florence-2-base`, ~230M params, local via HuggingFace
transformers, loaded once and kept resident.

Florence does not follow free-form prompts. It takes a fixed task token —
`FLORENCE_TASK`, default `<DETAILED_CAPTION>` — and returns a sentence.
Generation is greedy, capped at `FLORENCE_MAX_NEW_TOKENS` (default 128).

It is fed a **context crop**, not the tight identity crop.
`crop_with_context()` uses two knobs pulling opposite ways: `pad_frac` widens
around the box so the caption can say what the animal is *doing* and *where*,
while `min_frac` floors the result at a fraction of the frame so a distant
animal's tiny box doesn't become a handful of pixels. Crops that would overflow
an edge are **shifted inward rather than shrunk**, so an animal at the frame
edge still yields a full-size crop instead of a clipped sliver.

---

## 7. Stage four — reconciliation (the interesting part)

Three sources now disagree: YOLO's class, Florence's sentence, and CLIP's
identity verdict. `caption_logic.py` is the pure, unit-tested arbiter.

**The caption can overrule YOLO.** YOLO says `cow`, the caption says "an orange
cat" → report a cat. This is how lookalike classes get rescued, and how the
system names critters YOLO has no class for at all (raccoon, possum, fox).

**Species from the captioner are hedged, never asserted.** Florence reliably
tells cat from dog from bird. Beyond that its species name is a guess, and on a
dark frame an orange tabby is its favourite fox or squirrel. So
`SPECULATIVE_CAPTION_ANIMALS` — raccoon, opossum, fox, skunk, squirrel, rabbit,
deer — are still reported (a raccoon really does visit) but the wording is
hedged, *unless identity independently recognises the visitor*.

**The size contradiction.** This is my favourite rule in the codebase. Every
`LOOKALIKE` class is a large quadruped. For the nano model to fire one, it had
to see a big animal-shaped blob filling real frame area — which a squirrel or
rabbit physically cannot produce at porch distance. So "YOLO: bear" + "caption:
red squirrel" is a geometric impossibility, and the only thing that makes that
box is the cat.

**False-positive rejection** (`looks_like_false_positive()`) — two signals:
explicit negation ("no animal", "empty doorway"), or a description that talks
about a person and objects while mentioning *no* animal vocabulary at all. That
second one catches YOLO hallucinating a cat out of a person's shadow.

**Ignored species, second half.** Stage one removed `bird` from YOLO's class
list, so the only bird that can still reach here is one YOLO called something
else — a crow read as a "cat" at porch distance. A caption naming *only*
ignored species therefore suppresses the detection, on the same standing rule
that the caption outranks YOLO's label. The identity model is the sole veto:
a confirmed Cheeto (or any recognised household pet) is never dropped over one
bad caption, and an ignored species is never *reported* even when something
else keeps the detection alive.

---

## 8. Sharing one Pi between three cameras

Capture is per-process and isolated. Florence and CLIP are not — letting every
camera saturate all four cores makes every feed worse.

`inference_gate.py` is a **file-lock semaphore** across processes:
`AI_SHARED_SLOTS` (default 1) lock files under `/tmp/critter-cam-ai`, claimed
with non-blocking `flock`. The kernel releases them automatically if a detector
crashes, so a dead process cannot wedge the others.

Callers choose their own timeout policy. Identity gets a short leash —
`AI_IDENTITY_WAIT_SECONDS` (default 3) — because if another camera is already
captioning, this observation is still useful and reviewable, and freezing this
camera's capture and live loop for half a minute is the worse outcome.
Captioning gets a longer one.

---

## 9. Visits, alerts, and what reaches your phone

Observations and notifications are deliberately separate decisions.

Detections are grouped into **visits** by a 10-minute gap rule. `alert_logic.py`
then decides who gets told, with three cadences:

- `first_of_visit` — one alert when a visit starts, then silence.
- `visit_then_repeat` — the visit start always alerts; a visitor who stays
  alerts again each cooldown. A cat that settles in for an hour is still news;
  an alert every minute of the nap is spam.
- `every_detection` — cooldown only, no notion of visits.

Both visit modes **fail closed** when visit attachment is unavailable. A missed
alert during an API problem beats turning one seated cat into an alert per
frame.

Push is a data-only FCM wake-up carrying **no detection detail**; the phone
fetches the actual content from the Cheeto box. That gives instant alerts
without the app running a background service.

---

## 10. Live view and rewind

Live view lives *inside* the detector process, because `/dev/video0` can only be
opened once and the detector holds it. Publishing a frame stores a **reference
only** — no copy, no encode — so it costs nothing when nobody is watching. JPEG
encoding happens on the serving thread.

**Rewind** keeps the last 15 seconds continuously, because it has to already be
recording by the time you decide you want it. A separate thread samples frames
at 4fps into a ring bounded **in bytes first, seconds second** — the same
fifteen seconds is a few megabytes of a still driveway and considerably more of
a hedge in the wind, and memory is what kills a process holding two detectors,
Florence and CLIP. On request, ffmpeg muxes the ring to H.264 + AAC on demand.

Audio is captured **once** as raw PCM and fanned out to both the rewind ring and
any live listeners, so they no longer lock each other out.

Measured on the Pi: a 640×480 frame is 39KB and 3.2ms to encode (≈1.3% of a
core at 4fps); 15s is ~2.4MB of video plus 0.5MB of audio; the mux takes ~1.7s
with libx264 ultrafast. The Pi lists `h264_v4l2m2m` but has no device behind it,
so libx264 it is.

---

## 11. Data model and access control

Postgres with row-level security, and **every** client call goes through a
`security definer` function that begins by calling
`require_household_member(household_id)` or, for shared settings,
`is_household_manager(household_id)`. The app never touches tables directly.

The split is by consequence, not seniority:
- **Manager only** — rotation, battery saver. These change what *everyone's*
  pictures look like.
- **Any member** — camera location. It changes a label, and whoever carried the
  camera into the other room is the one who knows where it went.

---

## 12. Known limits, honestly

- **Front-door runs pinned at its 150% CPU quota**, because YOLO runs on every
  frame of an empty driveway (§3). Throttling it is a real trade against how
  fast a cat is spotted, so it has been left alone deliberately.
- **The XIAO tops out around 1.1fps** at 1024×768 live. That is the ESP32's
  wifi, not the Pi and not battery saver — the poll interval was verified at
  0.20s during the measurement.
- **Florence's species names beyond cat/dog/bird are guesses**, which is the
  entire reason §4 and §7 exist.
- **The prototype is only as good as its contact sheet.** Look at it.
- **Pi memory is ~890MB available** with all three detectors warm. Measured
  stable, not growing.
