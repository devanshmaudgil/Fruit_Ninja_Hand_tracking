"""MediaPipe Tasks index-fingertip tracking — low-latency LIVE_STREAM mode."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

Point = Tuple[float, float]

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "hand_landmarker.task"

INDEX_TIP = 8

# Infer on a small frame for speed; map landmarks back to game size
INFER_W = 320
INFER_H = 240


class HandTracker:
    """Async hand tracking with low-res inference and tip prediction."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.45,
        tracking_confidence: float = 0.45,
        model_path: Optional[Path] = None,
    ):
        model = Path(model_path) if model_path else DEFAULT_MODEL
        if not model.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model}. "
                "Download hand_landmarker.task into the models/ folder."
            )

        self._lock = threading.Lock()
        self._raw_tip: Optional[Point] = None  # normalized 0–1 in infer space, then game coords
        self._prev_tip: Optional[Point] = None
        self._tip: Optional[Point] = None
        self._vx = 0.0
        self._vy = 0.0
        self._last_result_t = 0.0
        self._game_w = 960
        self._game_h = 540
        self._start_ms = int(time.time() * 1000)
        self._last_ts = -1

        base_options = mp_python.BaseOptions(model_asset_path=str(model))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            result_callback=self._on_result,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        tip: Optional[Point] = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0][INDEX_TIP]
            tip = (lm.x * self._game_w, lm.y * self._game_h)

        with self._lock:
            now = time.perf_counter()
            if tip is not None and self._raw_tip is not None:
                dt = max(1e-3, now - self._last_result_t)
                self._vx = (tip[0] - self._raw_tip[0]) / dt
                self._vy = (tip[1] - self._raw_tip[1]) / dt
            elif tip is None:
                self._vx *= 0.5
                self._vy *= 0.5
            self._raw_tip = tip
            self._last_result_t = now

    def update(self, frame_bgr: np.ndarray, width: int, height: int) -> Optional[Point]:
        """
        Submit a downscaled frame for async detection; return predicted tip immediately.
        frame_bgr should already be mirrored.
        """
        self._game_w = width
        self._game_h = height

        # Downscale for fast inference
        small = cv2.resize(frame_bgr, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000) - self._start_ms
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        try:
            self._landmarker.detect_async(mp_image, timestamp_ms)
        except Exception:
            # Drop frame if landmarker is busy — keeps game loop real-time
            pass

        with self._lock:
            raw = self._raw_tip
            vx, vy = self._vx, self._vy
            age = time.perf_counter() - self._last_result_t if self._last_result_t else 999.0

        self._prev_tip = self._tip

        if raw is None or age > 0.35:
            self._tip = None
            return None

        # Predict slightly ahead to hide model latency (~1 frame)
        lead = min(0.045, age + 0.02)
        pred = (raw[0] + vx * lead, raw[1] + vy * lead)
        # Light smoothing toward prediction
        if self._tip is not None:
            a = 0.65
            self._tip = (self._tip[0] * (1 - a) + pred[0] * a, self._tip[1] * (1 - a) + pred[1] * a)
        else:
            self._tip = pred
        return self._tip

    @property
    def tip(self) -> Optional[Point]:
        return self._tip

    @property
    def prev_tip(self) -> Optional[Point]:
        return self._prev_tip

    def speed(self) -> float:
        if self._tip is None or self._prev_tip is None:
            return 0.0
        dx = self._tip[0] - self._prev_tip[0]
        dy = self._tip[1] - self._prev_tip[1]
        return (dx * dx + dy * dy) ** 0.5

    def close(self) -> None:
        self._landmarker.close()
