# CrowdLens Open

**Video evidence analysis with anonymous tracking, possible-weapon detection and human review.**

CrowdLens Open is an open-source **research prototype**, not an autonomous security decision system. It processes a local video, detects people, assigns temporary IDs, tries to reconnect an appearance after a person leaves and re-enters, and emits reviewable observations. With a separately supplied **trained weapon detector**, it also flags possible weapons. The tool does **not** recognize people's names/faces, infer criminal intent, or determine that a nearby person owns a detected object.

> **Important:** This is a newly implemented, non-validated analysis pipeline. No model weights, identifiable footage, or labeled evaluation dataset are bundled. Neither stable re-identification nor weapon detection can be guaranteed. Do not use automated outputs as the sole basis for detention, targeting, accusation, or other consequential decisions.

## What works and what is conditional

| Capability | Implementation | Limitation |
| --- | --- | --- |
| Person detection | Ultralytics YOLO using `yolov8n.pt` by default | Downloads weights on first use; quality depends on scene |
| Continuous person tracking | BoT-SORT frame-to-frame track IDs | Occlusion and camera motion cause ID switches |
| Exit/re-entry association | Short-lived anonymous appearance gallery, with one-to-one assignment and ambiguity rejection | Probabilistic. Similar clothes, lighting changes, occlusion or long gaps can defeat it |
| Appearance encoder | HSV histogram by default; optional ImageNet ResNet18 features | Neither is a benchmarked person-ReID model; tune and evaluate |
| Weapon flags | Any compatible YOLO detection checkpoint whose class names match configured weapon labels | **Requires external licensed/trained weights**; default person model **cannot** detect firearms |
| Observable behavior flags | Zone entry, sustained rapid image-space motion, prolonged appearance | Not malicious intent; thresholds need camera-specific validation |
| Human review | Local Streamlit viewer, annotated MP4, JSONL events, track CSV | No built-in role management or authenticated multi-user workflow |
| Evidence provenance | Timestamp, model/config metadata, optional source SHA-256 | Does not establish legal chain of custody |

## Quickstart

Requires Python 3.10+ and FFmpeg-compatible local video decoding via OpenCV.

```bash
git clone https://github.com/YOUR-ACCOUNT/crowdlens-open.git
cd crowdlens-open
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e '.[all]'
```

First, inspect the settings (no model inference needed):

```bash
crowdlens check-config --config config/example.yaml
```

For **person tracking only**, with no weapon inference:

```bash
crowdlens analyze /path/to/video.mp4 --output runs/persons_001
```

For **people and potential weapons**, supply the trained/validated weights and a new run directory:

```bash
crowdlens analyze /path/to/video.mp4 \
  --config config/example.yaml \
  --weapon-model /path/to/weapon-best.pt \
  --reid-backend resnet18 \
  --output runs/review_001
```

Run the **local** review interface:

```bash
streamlit run streamlit_app.py --server.address 127.0.0.1
```

The first use of the YOLO model and optional ResNet18 encoder may download pretrained weights. For offline environments, pre-cache/check in licensed model artifacts through your secure deployment process; **do not commit restricted weights or footage to GitHub**. Video inference is CPU-capable but may be slow; a compatible GPU is recommended for long videos.

### Run outputs

Each run writes a **new** output directory; existing output directories are never overwritten.

```
runs/review_001/
  annotated.mp4   # boxes, anonymous IDs and review labels; audio not retained
  events.jsonl    # one JSON event per line; review_status=pending
  tracks.csv      # frame/time, temporary ID, bounding box and match mode
  report.json     # source hash, FPS, settings, model names and event counts
```

The app does not connect detections to named individuals, social media, watchlists or databases. Appearance features stay in process memory for the run; they are not saved to the report. IDs such as `P0001` are **unique only inside that single video analysis**, and are not cross-camera identifiers.

## How temporary tagging works

1. YOLO and BoT-SORT produce per-frame person boxes and local track IDs.
2. A continued local track keeps its anonymous ID.
3. When a previously unseen local track appears, it is compared with recently inactive gallery appearances.
4. An inactive ID is reused **only** if similarity meets a threshold, the best candidate is sufficiently more similar than alternatives, and the ID isn't already assigned to another current observation.
5. Otherwise a new ID is assigned. Ambiguous matches are marked for review, never silently forced.

The gallery expires after a configured number of seconds. It is **not** a biometric identity store. The default histogram backend is lightweight and particularly vulnerable to similar clothing. The ResNet18 option produces generic visual features, **not** calibrated person-ReID scores. The same ID can be incorrectly reused; two visits by the same person can receive different IDs. A reliable deployment needs dedicated person-ReID training and benchmarked failure rates.

## Review events, not accusations

- `possible_weapon`: a custom weapon model detected a configured class. `nearby_anonymous_tracks` represents box overlap/proximity, **not ownership or intent**.
- `zone_entry`: a person's box center entered a normalized image polygon configured by the operator. The example `demo_zone` is illustrative, not a real restricted area.
- `rapid_motion`: normalized image-space displacement exceeded a configurable threshold across consecutive frames. Moving cameras, perspective, scene cuts and tracking errors invalidate the heuristic.
- `prolonged_presence`: one anonymous tracked appearance remained visible for a configured interval. Not a judgment of loitering or threat.
- `appearance_reassociation`: a returning appearance was provisionally linked to an older local track; similarity is a model score, not identity certainty.
- `ambiguous_appearance`: gallery comparison did not produce a unique safe match; a fresh ID was issued.

All events default to `review_status: pending`; an authorized person must inspect the source footage, record their decision, and maintain their own audit process. The example UI displays pending events but does not implement authenticated review writes.

## Train a weapon detector

The COCO weights used for person tracking **do not include firearm classes**. Obtain lawfully sourced, licensed annotated data and train a separate YOLO model:

```bash
python scripts/train_weapon_model.py \
  --data /path/to/weapon_dataset.yaml \
  --base yolov8n.pt --epochs 50 --imgsz 640
```

See [`data/weapon_dataset.example.yaml`](data/weapon_dataset.example.yaml) for the required class schema, and [`docs/EVALUATION.md`](docs/EVALUATION.md) for minimum validation expectations. Weapon classes must match `weapon_labels` in your YAML exactly (case-insensitive). **Model artifacts and training data are not included.**

## Test

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Unit tests verify ID continuity, reacquisition, ambiguity, expiry, objective events, config validation and event deduplication **without downloading vision models**. They do not measure live detector accuracy. For a real deployment, run video-level integration tests and the evaluation plan in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Safety, privacy and limitations

- Use only footage you are authorized to process. Observe local consent, retention, access-control and other applicable requirements.
- Keep raw footage, output videos, model files and tracked observations **out of public repositories**. Outputs can still reveal identifiable people even though IDs are anonymous.
- No face recognition, named-person matching, population-scale search or cross-video linkage is implemented.
- Do not infer **malicious intent**, guilt, weapon possession or a personal identity from a bounding box, motion score or clothing similarity.
- Multiple humans with similar appearance, rare camera angles, crowded scenes and changing light can produce wrong matches. Report failures rather than hiding them.
- Do not publicly deploy the Streamlit UI without authentication, encryption, upload limits, storage segregation, retention and audit logging.
- This code has no certified security guarantees, formal benchmark, or legal evidentiary certification.

## Project structure

```
src/crowdlens/
  cli.py          # CLI
  config.py       # validated thresholds and zones
  pipeline.py     # local video inference and export
  identity.py     # short-lived anonymous ID gallery
  appearance.py   # histogram and optional CNN features
  events.py       # observable review rules
streamlit_app.py  # local review interface
scripts/          # training utility
tests/            # lightweight deterministic unit tests
config/           # example configuration
docs/             # model evaluation, governance and release notes
```

## GitHub publication

Create an empty GitHub repository named `crowdlens-open`, then from this project root:

```bash
git init
git add .
git commit -m "Initial CrowdLens Open research prototype"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/crowdlens-open.git
git push -u origin main
```

Suggested GitHub topics: `computer-vision`, `video-analytics`, `object-detection`, `multi-object-tracking`, `person-reidentification`, `yolov8`, `human-in-the-loop`, `research-prototype`, `security-research`.

Before public release, ensure the license obligations for **all detector weights, dependencies and training datasets** are met. This code is released under `AGPL-3.0-only` to align with the Ultralytics AGPL offering. Ultralytics also offers a separate enterprise license; consult the vendor for closed-source use. See [`LICENSE`](LICENSE), [`SECURITY.md`](SECURITY.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md).

