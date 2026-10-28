"""Offline, local-first video inference and auditable review artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .appearance import make_encoder
from .config import Settings
from .events import BehaviorMonitor, WeaponEventMemory
from .identity import IdentityGallery
from .models import AssignedPerson, Event, ObjectObservation, PersonObservation


def _boxes(result: Any) -> list[tuple[tuple[int, int, int, int], float, int, int | None]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    positions = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.int().cpu().numpy()
    ids = boxes.id.int().cpu().numpy() if boxes.id is not None else [None] * len(positions)
    return [(tuple(int(round(v)) for v in box), float(score), int(cls),
             None if local_id is None else int(local_id))
            for box, score, cls, local_id in zip(positions, scores, classes, ids)]


def _crop(frame: Any, box: tuple[int, int, int, int]) -> Any:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    return frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]


def _names(result: Any) -> dict[int, str]:
    names = result.names
    return {int(k): str(v).lower() for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _annotate(frame: Any, people: list[AssignedPerson], weapons: list[ObjectObservation],
              events: list[Event], frame_no: int, fps: float) -> None:
    import cv2

    for person in people:
        x1, y1, x2, y2 = person.bbox
        color = (0, 180, 255) if person.match_kind == "reidentified" else (50, 200, 90)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{person.stable_id} [{person.match_kind}]", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    for weapon in weapons:
        x1, y1, x2, y2 = weapon.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 240), 2)
        cv2.putText(frame, f"POSSIBLE {weapon.label} {weapon.confidence:.2f}",
                    (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, (35, 35, 240), 2)
    cv2.putText(frame, f"CrowdLens / review only / {frame_no / fps:.1f}s", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (250, 250, 250), 2)
    if events:
        cv2.putText(frame, "REVIEW EVENT", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 160, 255), 2)


def analyze(video: str | Path, output: str | Path, config: Settings,
            on_progress: Any = None) -> dict[str, Any]:
    """Analyze one video. Writes new artifacts only; refuses existing run dirs.

    Detection models are real YOLO inference, not staged demo overlays. Model
    training/validation, calibrated alerts and operator review are separate.
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the video dependencies: pip install -e '.[video]'") from exc

    config.validate()
    source = Path(video).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    dest = Path(output).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f"Output directory already exists: {dest} (choose a fresh run name)")

    person_model = YOLO(config.person_model)
    person_names = {int(k): str(v).lower() for k, v in (person_model.names.items() if isinstance(person_model.names, dict)
                                                        else enumerate(person_model.names))}
    person_class = next((k for k, name in person_names.items() if name == "person"), None)
    if person_class is None:
        raise ValueError("Person model must contain a class named 'person'")
    weapon_model = YOLO(config.weapon_model) if config.weapon_model else None
    if weapon_model is None:
        print("WARNING: no --weapon-model configured; weapon detection DISABLED.", file=sys.stderr)
    else:
        # A mislabeled checkpoint is easy to mistake for a successful zero-alert
        # run. Reject it before opening outputs or processing video.
        model_names = {str(v).casefold().strip() for v in
                       (weapon_model.names.values() if isinstance(weapon_model.names, dict) else weapon_model.names)}
        enabled_names = {str(v).casefold().strip() for v in config.weapon_labels}
        if not model_names.intersection(enabled_names):
            raise ValueError("Weapon checkpoint has no configured weapon classes. "
                             f"Model classes: {sorted(model_names)}; expected any of: {sorted(enabled_names)}")
    encoder = make_encoder(config.reid_backend)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0  # explicit estimate; provenance records this fallback
        fps_estimated = True
    else:
        fps_estimated = False
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1:
        capture.release()
        raise ValueError("Video has invalid frame dimensions")

    dest.mkdir(parents=True)
    writer = cv2.VideoWriter(str(dest / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open an MP4 writer on this system")
    gallery = IdentityGallery(config.reid_similarity, config.reid_margin,
                              max(1, round(config.reid_gallery_seconds * fps)), config.reid_ema)
    monitor = BehaviorMonitor(config.zone_names, config.rapid_motion_body_heights_per_second,
                              config.rapid_motion_min_frames, config.prolonged_presence_seconds,
                              config.absence_reset_seconds, config.rapid_motion_cooldown_seconds,
                              config.zone_cooldown_seconds)
    weapon_memory = WeaponEventMemory(config.event_cooldown_seconds)
    event_counts: Counter[str] = Counter()
    stable_ids: set[str] = set()
    frame_no = 0

    try:
        with (dest / "events.jsonl").open("w", encoding="utf-8") as event_stream, \
                (dest / "tracks.csv").open("w", encoding="utf-8", newline="") as track_stream:
            track_writer = csv.writer(track_stream)
            track_writer.writerow(["frame", "time_seconds", "anonymous_id", "local_tracker_id",
                                   "x1", "y1", "x2", "y2", "detection_confidence", "match_kind", "appearance_similarity"])
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_no += 1
                results = person_model.track(frame, persist=True, classes=[person_class],
                                             conf=config.person_confidence, tracker=config.tracker,
                                             verbose=False)
                observations: list[PersonObservation] = []
                for bbox, confidence, cls, local_id in _boxes(results[0]):
                    if local_id is None or cls != person_class:
                        continue
                    crop = _crop(frame, bbox)
                    if crop.size == 0:
                        continue
                    observations.append(PersonObservation(local_id, bbox, confidence, encoder.encode(crop)))
                persons = gallery.update(observations, frame_no)
                stable_ids.update(p.stable_id for p in persons)

                weapons: list[ObjectObservation] = []
                if weapon_model is not None:
                    found = weapon_model.predict(frame, conf=config.weapon_confidence, verbose=False)
                    allowed = {w.casefold().strip() for w in config.weapon_labels}
                    labels = _names(found[0])
                    for bbox, confidence, cls, _ in _boxes(found[0]):
                        name = labels.get(cls, "unknown")
                        if name in allowed:
                            weapons.append(ObjectObservation(name, bbox, confidence))

                events = monitor.observe(persons, frame_no, fps, width, height)
                events.extend(weapon_memory.observe(weapons, persons, frame_no, fps))
                for person in persons:
                    track_writer.writerow([frame_no, round(frame_no / fps, 3), person.stable_id,
                                           person.local_id, *person.bbox, round(person.confidence, 4),
                                           person.match_kind, "" if person.match_similarity is None else round(person.match_similarity, 4)])
                    if person.match_kind in {"reidentified", "ambiguous"}:
                        events.append(Event(frame_no, frame_no / fps,
                                            "appearance_reassociation" if person.match_kind == "reidentified" else "ambiguous_appearance",
                                            person.stable_id, person.match_similarity,
                                            {"local_track_id": person.local_id, "not_verified_identity": True}))
                for event in events:
                    event_stream.write(json.dumps(asdict(event), sort_keys=True) + "\n")
                    event_counts[event.event_type] += 1
                _annotate(frame, persons, weapons, events, frame_no, fps)
                writer.write(frame)
                if on_progress is not None and (frame_no % 10 == 0 or frame_no == total_frames):
                    on_progress(frame_no, total_frames)
    finally:
        capture.release()
        writer.release()

    if frame_no == 0:
        raise ValueError("Video contains no decodable frames")
    report: dict[str, Any] = {
        "application": "CrowdLens Open", "version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": source.name,
        "source_sha256": _sha256(source) if config.input_hash else None,
        "fps": fps, "fps_estimated": fps_estimated,
        "width": width, "height": height, "frames": frame_no,
        "duration_seconds": round(frame_no / fps, 3),
        "anonymous_tracks_count": len(stable_ids),
        "events_by_type": dict(event_counts),
        "weapon_detection_enabled": weapon_model is not None,
        "weapon_model_classes": sorted(model_names) if weapon_model is not None else [],
        "warning": "Automated outputs are hypotheses for human review, never proof of identity, intent or possession.",
        "settings": config.as_dict(),
        "artifacts": ["annotated.mp4", "events.jsonl", "tracks.csv", "report.json"],
    }
    (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

