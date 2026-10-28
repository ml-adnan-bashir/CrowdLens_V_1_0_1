# Validation and evaluation protocol

This repository contains **no measured accuracy claim**. Unit tests of the matching algorithm are not evidence of deployment fitness.

## Dataset and ground truth

Use an authorized, representative video set with diverse lighting, occlusion, density, camera angle, clothing similarity, entrances/exits, and both positive/negative weapon examples. Split by camera, recording date and scene—not by random neighboring frames. Do not publish identifiable samples.

Have trained annotators independently label person bounding boxes, exact return/re-entry identity within each test video, weapon boxes and classes, and event intervals. Reconcile disagreements. Annotate unseen individuals as unknown; retain uncertain ground-truth entries.

## Metrics to publish with versions and confidence intervals

| Layer | Report |
| --- | --- |
| Person boxes | Precision, recall, mAP50-95 by camera and lighting |
| Tracking | IDF1, HOTA, ID switches, fragmentation and track loss |
| Return association | Correct-reconnection precision/recall, false merges, false splits, unknown rejection, by absence duration |
| Weapon classes | Class-wise precision/recall, mAP50-95, calibration, misses and false alerts per video-hour |
| Motion and zone events | Event-level precision/recall, per-hour false alarms, latency, inter-reviewer agreement |
| Operations | FPS, latency, GPU RAM, failure rate, privacy/retention compliance |

Publish confusion matrices and stratified metrics by scene density, lighting, occlusion, distance, clothing similarity and camera movement. Include false-positive and false-negative example analyses. Do not make safety-critical deployment decisions based on one average accuracy number.

## Mandatory adversarial cases

- Two simultaneously visible people wearing the same color and similar clothes.
- A person exits and re-enters after 1 s, 10 s, 45 s and longer than the configured gallery lifetime.
- A track disappears behind an obstacle while a second person enters.
- Scene cuts, changed cameras, panning cameras and moving light sources.
- Small, partially occluded, toy, replica, holstered or non-weapon lookalike objects.
- A weapon-shaped object overlaps a bystander who does not possess it.
- Missing and incorrect FPS/video metadata.

## Operational gate

Do not deploy without a documented domain-specific threat model, legal/privacy review, human decision policy, minimum acceptable metrics, regression benchmarks, and escalation/fallback path. Any change to data, model weights, thresholds or camera configuration requires reevaluation.

