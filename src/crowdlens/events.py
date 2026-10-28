"""Objective event heuristics for human review, not intent classification."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .models import AssignedPerson, BBox, Event, ObjectObservation


def _center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _contains(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon for normalized image coordinates."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass
class _MotionState:
    first_frame: int
    last_frame: int
    center: tuple[float, float]
    height: float
    rapid_streak: int = 0
    prolonged_reported: bool = False


class BehaviorMonitor:
    def __init__(self, zones: dict[str, list[list[float]]],
                 motion_threshold: float, min_motion_frames: int,
                 prolonged_seconds: float, absence_reset_seconds: float,
                 motion_cooldown_seconds: float, zone_cooldown_seconds: float):
        self.zones = zones
        self.motion_threshold = motion_threshold
        self.min_motion_frames = min_motion_frames
        self.prolonged_seconds = prolonged_seconds
        self.absence_reset_seconds = absence_reset_seconds
        self.motion_cooldown_seconds = motion_cooldown_seconds
        self.zone_cooldown_seconds = zone_cooldown_seconds
        self.states: dict[str, _MotionState] = {}
        self.zone_inside: dict[tuple[str, str], bool] = {}
        self.last_event: dict[tuple[str, str], int] = {}

    def observe(self, persons: list[AssignedPerson], frame: int, fps: float,
                width: int, height: int) -> list[Event]:
        emitted: list[Event] = []
        current = {p.stable_id for p in persons}
        # Clear zone state after absence so re-entry can trigger a fresh review.
        for sid in set(self.states) - current:
            state = self.states[sid]
            if frame - state.last_frame > int(fps * self.absence_reset_seconds):
                for key in list(self.zone_inside):
                    if key[0] == sid:
                        del self.zone_inside[key]
                del self.states[sid]

        for person in persons:
            sid = person.stable_id
            x, y = _center(person.bbox)
            box_height = max(1.0, float(person.bbox[3] - person.bbox[1]))
            state = self.states.get(sid)
            if state is None or frame - state.last_frame > int(fps * self.absence_reset_seconds):
                state = _MotionState(frame, frame, (x, y), box_height)
                self.states[sid] = state
            else:
                delta = frame - state.last_frame
                if 0 < delta <= fps and person.match_kind == "ongoing":
                    speed = hypot(x - state.center[0], y - state.center[1]) / max(1.0, (state.height + box_height) / 2) * fps / delta
                    state.rapid_streak = state.rapid_streak + 1 if speed > self.motion_threshold else 0
                    key = (sid, "rapid_motion")
                    if state.rapid_streak >= self.min_motion_frames and self._can_emit(key, frame, fps, self.motion_cooldown_seconds):
                        emitted.append(Event(frame, frame / fps, "rapid_motion", sid,
                                             details={"body_heights_per_second": round(speed, 2), "heuristic": True}))
                        self.last_event[key] = frame
                else:
                    state.rapid_streak = 0
                state.last_frame = frame
                state.center = (x, y)
                state.height = box_height

            if not state.prolonged_reported and (frame - state.first_frame) / fps >= self.prolonged_seconds:
                emitted.append(Event(frame, frame / fps, "prolonged_presence", sid,
                                     details={"seconds": round((frame - state.first_frame) / fps, 2), "heuristic": True}))
                state.prolonged_reported = True

            center_norm = (x / width, y / height)
            for zone_name, polygon in self.zones.items():
                key = (sid, zone_name)
                inside = _contains(center_norm, polygon)
                if inside and not self.zone_inside.get(key, False) and self._can_emit(key, frame, fps, self.zone_cooldown_seconds):
                    emitted.append(Event(frame, frame / fps, "zone_entry", sid,
                                         details={"zone": zone_name, "heuristic": True}))
                    self.last_event[key] = frame
                self.zone_inside[key] = inside
        return emitted

    def _can_emit(self, key: tuple[str, str], frame: int, fps: float, cooldown: float) -> bool:
        return frame - self.last_event.get(key, -10**12) >= fps * cooldown


class WeaponEventMemory:
    """Suppress repeated events for the same nearby detection in adjacent frames."""

    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self.recent: list[tuple[str, BBox, int]] = []

    def observe(self, weapons: list[ObjectObservation], persons: list[AssignedPerson],
                frame: int, fps: float) -> list[Event]:
        self.recent = [r for r in self.recent if frame - r[2] <= fps * self.cooldown_seconds]
        result: list[Event] = []
        for ob in weapons:
            # Image-space proximity is not evidence of ownership or intent.
            near = [p.stable_id for p in persons if _iou(ob.bbox, p.bbox) > 0.05 or
                    (p.bbox[0] <= _center(ob.bbox)[0] <= p.bbox[2] and
                     p.bbox[1] <= _center(ob.bbox)[1] <= p.bbox[3])]
            matched = next((idx for idx, (label, box, _) in enumerate(self.recent)
                            if label == ob.label and _iou(box, ob.bbox) >= 0.25), None)
            if matched is not None:
                self.recent[matched] = (ob.label, ob.bbox, frame)
                continue
            result.append(Event(frame, frame / fps, "possible_weapon", None, ob.confidence,
                                {"label": ob.label, "bbox_xyxy": list(ob.bbox),
                                 "nearby_anonymous_tracks": near, "proximity_is_not_possession": True}))
            self.recent.append((ob.label, ob.bbox, frame))
        return result

