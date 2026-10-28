"""Replaceable visual appearance encoders; no facial recognition."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AppearanceEncoder(Protocol):
    def encode(self, bgr_crop: np.ndarray) -> np.ndarray: ...


class HistogramEncoder:
    """Fast baseline: clothing/color histogram, weak for similar appearances."""

    def encode(self, bgr_crop: np.ndarray) -> np.ndarray:
        import cv2

        if bgr_crop.size == 0:
            return np.ones(96, dtype=np.float32) / np.sqrt(96)
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
        vec = hist.flatten().astype(np.float32)
        length = float(np.linalg.norm(vec))
        return vec / length if length else np.ones(96, dtype=np.float32) / np.sqrt(96)


class ResNet18Encoder:
    """Optional ImageNet embedding baseline, NOT a validated person-ReID model.

    First use downloads the pretrained ImageNet weights; offline deployments
    should pre-cache them. Camera/site-specific validation remains required.
    """

    def __init__(self) -> None:
        import torch
        from torchvision import models

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.network = torch.nn.Sequential(*list(backbone.children())[:-1]).eval()
        self.transform = weights.transforms()
        self.torch = torch

    def encode(self, bgr_crop: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        if bgr_crop.size == 0:
            return np.zeros(512, dtype=np.float32)  # caller discards invalid features
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        sample = self.transform(Image.fromarray(rgb)).unsqueeze(0)
        with self.torch.no_grad():
            vec = self.network(sample).flatten().cpu().numpy().astype(np.float32)
        length = float(np.linalg.norm(vec))
        return vec / length if length else vec


def make_encoder(backend: str) -> AppearanceEncoder:
    if backend == "histogram":
        return HistogramEncoder()
    if backend == "resnet18":
        return ResNet18Encoder()
    raise ValueError(f"Unsupported re-identification backend: {backend}")

