"""Synthetic end-to-end plumbing test, deliberately no camera/video/model weights."""

import json
import sys
import types

import numpy as np

from crowdlens.config import Settings
from crowdlens.pipeline import analyze


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def int(self):
        return Tensor(self.values.astype(int))

    def numpy(self):
        return self.values


class Boxes:
    def __init__(self, rows):
        self.rows = rows
        self.xyxy = Tensor([row[:4] for row in rows])
        self.conf = Tensor([row[4] for row in rows])
        self.cls = Tensor([row[5] for row in rows])
        self.id = Tensor([row[6] for row in rows]) if rows and rows[0][6] is not None else None

    def __len__(self):
        return len(self.rows)


class Result:
    def __init__(self, rows, names):
        self.boxes = Boxes(rows)
        self.names = names


def test_mocked_video_pipeline_creates_real_structured_artifacts(tmp_path, monkeypatch):
    class Video:
        def __init__(self):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, field):
            return {1: 10., 2: 100, 3: 100, 4: 3}[field]

        def read(self):
            self.index += 1
            if self.index > 3:
                return False, None
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

        def release(self):
            pass

    class Writer:
        def __init__(self, name):
            self.name = name
            self.frames = 0

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames += 1

        def release(self):
            from pathlib import Path

            Path(self.name).write_bytes(f"synthetic:{self.frames}".encode())

    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FPS = 1
    cv2.CAP_PROP_FRAME_WIDTH = 2
    cv2.CAP_PROP_FRAME_HEIGHT = 3
    cv2.CAP_PROP_FRAME_COUNT = 4
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.VideoCapture = lambda name: Video()
    cv2.VideoWriter_fourcc = lambda *args: 0
    cv2.VideoWriter = lambda name, *args: Writer(name)
    cv2.rectangle = lambda *args, **kwargs: None
    cv2.putText = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    class YOLO:
        def __init__(self, model):
            self.weapon = model == "weapons.pt"
            self.names = {0: "handgun" if self.weapon else "person"}
            self.count = 0

        def track(self, *args, **kwargs):
            self.count += 1
            row = (10, 10, 30, 60, .91, 0, 1 if self.count == 1 else 77)
            return [Result([row] if self.count != 2 else [], self.names)]

        def predict(self, *args, **kwargs):
            self.count += 1
            row = (12, 18, 20, 29, .85, 0, None)
            return [Result([row] if self.count == 1 else [], self.names)]

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    class StubEncoder:
        def encode(self, crop):
            return np.array([1., 0.], dtype=np.float32)

    monkeypatch.setattr("crowdlens.pipeline.make_encoder", lambda backend: StubEncoder())
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake fixture; VideoCapture is mocked")
    output = tmp_path / "output"
    result = analyze(source, output, Settings(weapon_model="weapons.pt", reid_similarity=.9))
    assert result["frames"] == 3
    assert result["weapon_detection_enabled"] is True
    assert result["anonymous_tracks_count"] == 1
    assert result["events_by_type"]["possible_weapon"] == 1
    assert result["events_by_type"]["appearance_reassociation"] == 1
    assert (output / "annotated.mp4").read_bytes() == b"synthetic:3"
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert all(event["review_status"] == "pending" for event in events)
    assert (output / "report.json").exists()

