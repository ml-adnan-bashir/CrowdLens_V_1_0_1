"""Conservative, anonymous within-video re-association of local tracker IDs.

This is appearance matching, not facial recognition or a determination of real
identity. Two people in similar attire may look alike; confidence can be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import AssignedPerson, PersonObservation


def _normalized(feature: np.ndarray) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(arr))
    if magnitude < 1e-9 or not np.isfinite(magnitude):
        raise ValueError("Appearance feature must have finite nonzero magnitude")
    return arr / magnitude


@dataclass
class _Identity:
    stable_id: str
    feature: np.ndarray
    last_seen: int


class IdentityGallery:
    """Use tracking continuity first, then one-to-one appearance matching.

    The gallery is in-memory and expires after max_gap_frames. Local tracking
    IDs are never exposed as persistent identities outside a single analysis.
    """

    def __init__(self, similarity: float = 0.92, ambiguity_margin: float = 0.06,
                 max_gap_frames: int = 1350, ema: float = 0.8):
        if not 0 <= similarity <= 1 or not 0 <= ambiguity_margin <= 1 or not 0 <= ema <= 1:
            raise ValueError("similarity, ambiguity_margin and ema must be in [0,1]")
        if max_gap_frames < 1:
            raise ValueError("max_gap_frames must be >= 1")
        self.similarity = similarity
        self.ambiguity_margin = ambiguity_margin
        self.max_gap_frames = max_gap_frames
        self.ema = ema
        self.identities: dict[str, _Identity] = {}
        self.local_to_stable: dict[int, str] = {}
        self.sequence = 0

    def _new(self, feature: np.ndarray, frame: int) -> str:
        self.sequence += 1
        stable_id = f"P{self.sequence:04d}"
        self.identities[stable_id] = _Identity(stable_id, feature, frame)
        return stable_id

    def update(self, observations: list[PersonObservation], frame: int) -> list[AssignedPerson]:
        if frame < 0:
            raise ValueError("Frame number cannot be negative")
        ids = [ob.local_id for ob in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate local tracker ID in a frame")

        # Drop expired appearances and their tracker bindings. Long-lived
        # identities are intentionally not kept as a biometric/person registry.
        expired = {sid for sid, rec in self.identities.items() if frame - rec.last_seen > self.max_gap_frames}
        for sid in expired:
            del self.identities[sid]
        self.local_to_stable = {lid: sid for lid, sid in self.local_to_stable.items() if sid in self.identities}

        features = [_normalized(ob.feature) for ob in observations]
        assignments: dict[int, tuple[str, str, float | None]] = {}
        occupied: set[str] = set()

        # Continuity of the live local tracker is stronger than a gallery guess.
        for index, ob in enumerate(observations):
            sid = self.local_to_stable.get(ob.local_id)
            if sid and sid not in occupied:
                assignments[index] = (sid, "ongoing", None)
                occupied.add(sid)

        unknown = [i for i in range(len(observations)) if i not in assignments]
        candidate_ids = [sid for sid, rec in self.identities.items() if sid not in occupied and rec.last_seen < frame]
        proposals: list[tuple[float, int, str]] = []
        ambiguous: set[int] = set()
        for index in unknown:
            matches = sorted(
                ((float(np.dot(features[index], self.identities[sid].feature)), sid) for sid in candidate_ids),
                reverse=True,
            )
            if not matches or matches[0][0] < self.similarity:
                continue
            # Reject close alternatives, so a generic black jacket does not
            # silently become a confident re-identification.
            if len(matches) > 1 and matches[0][0] - matches[1][0] < self.ambiguity_margin:
                ambiguous.add(index)
                continue
            proposals.append((matches[0][0], index, matches[0][1]))

        for score, index, sid in sorted(proposals, reverse=True):
            if sid not in occupied:
                assignments[index] = (sid, "reidentified", score)
                occupied.add(sid)
            else:
                ambiguous.add(index)

        result: list[AssignedPerson] = []
        for index, ob in enumerate(observations):
            if index in assignments:
                sid, kind, score = assignments[index]
                old = self.identities[sid]
                old.feature = _normalized(self.ema * old.feature + (1 - self.ema) * features[index])
                old.last_seen = frame
            else:
                sid = self._new(features[index], frame)
                kind = "ambiguous" if index in ambiguous else "new"
                score = None
            self.local_to_stable[ob.local_id] = sid
            result.append(AssignedPerson(ob.local_id, sid, ob.bbox, ob.confidence, kind, score))
        return result

