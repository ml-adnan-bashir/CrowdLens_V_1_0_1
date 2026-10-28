"""Validated project settings. All policy-sensitive thresholds are explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    person_model: str = "yolov8n.pt"
    weapon_model: str | None = None
    person_confidence: float = 0.35
    weapon_confidence: float = 0.50
    weapon_labels: list[str] = field(default_factory=lambda: ["handgun", "pistol", "rifle", "firearm", "gun", "knife"])
    tracker: str = "botsort.yaml"
    reid_backend: str = "histogram"  # histogram or resnet18
    reid_similarity: float = 0.92
    reid_margin: float = 0.06
    reid_gallery_seconds: float = 45.0
    reid_ema: float = 0.8
    zone_names: dict[str, list[list[float]]] = field(default_factory=dict)
    zone_cooldown_seconds: float = 10.0
    rapid_motion_body_heights_per_second: float = 2.5
    rapid_motion_min_frames: int = 3
    rapid_motion_cooldown_seconds: float = 10.0
    prolonged_presence_seconds: float = 30.0
    absence_reset_seconds: float = 3.0
    event_cooldown_seconds: float = 4.0
    input_hash: bool = True

    @classmethod
    def load(cls, path: str | Path | None) -> "Settings":
        if path is None:
            return cls().validate()
        with open(path, "r", encoding="utf-8") as stream:
            data: Any = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError("Settings YAML must be a mapping")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        return cls(**data).validate()

    def validate(self) -> "Settings":
        for name in ("person_confidence", "weapon_confidence", "reid_similarity", "reid_margin", "reid_ema"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.reid_backend not in {"histogram", "resnet18"}:
            raise ValueError("reid_backend must be histogram or resnet18")
        for name in ("reid_gallery_seconds", "zone_cooldown_seconds", "rapid_motion_cooldown_seconds", "prolonged_presence_seconds", "absence_reset_seconds", "event_cooldown_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rapid_motion_min_frames < 1:
            raise ValueError("rapid_motion_min_frames must be >= 1")
        if self.rapid_motion_body_heights_per_second <= 0:
            raise ValueError("rapid_motion_body_heights_per_second must be > 0")
        if not isinstance(self.weapon_labels, list) or not all(isinstance(x, str) for x in self.weapon_labels):
            raise ValueError("weapon_labels must be a string list")
        for name, polygon in self.zone_names.items():
            if not isinstance(name, str) or len(polygon) < 3 or any(len(point) != 2 for point in polygon):
                raise ValueError("Each zone must have a name and >= 3 [x,y] polygon points")
            if any(not all(0 <= float(value) <= 1 for value in point) for point in polygon):
                raise ValueError("Zone points must use normalized coordinates in [0, 1]")
        return self

    def as_dict(self) -> dict:
        return asdict(self)

