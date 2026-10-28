"""Small dependency-light data structures used across the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class PersonObservation:
    local_id: int
    bbox: BBox
    confidence: float
    feature: "Any"  # 1-D numpy array; kept optional at import time


@dataclass(frozen=True)
class AssignedPerson:
    local_id: int
    stable_id: str
    bbox: BBox
    confidence: float
    match_kind: str  # ongoing | new | reidentified | ambiguous
    match_similarity: float | None = None


@dataclass(frozen=True)
class ObjectObservation:
    label: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class Event:
    frame: int
    timestamp_s: float
    event_type: str
    person_id: str | None
    confidence: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    review_status: str = "pending"

