"""Procedural SFX + pygame mixer playback for Fruit Ninja."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
SOUNDS_DIR = ROOT / "assets" / "sounds"
SAMPLE_RATE = 22050


def _write_wav(path: Path, samples: np.ndarray) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _env(n: int, attack: float = 0.01, release: float = 0.08) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    a = max(1, int(attack * n))
    r = max(1, int(release * n))
    env = np.ones(n, dtype=np.float32)
    env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    env[-r:] = np.linspace(1, 0, r, dtype=np.float32)
    return env


def _tone(freq: float, dur: float, vol: float = 0.4, kind: str = "sine") -> np.ndarray:
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    if kind == "square":
        wave_s = np.sign(np.sin(2 * np.pi * freq * t))
    elif kind == "noise":
        wave_s = np.random.uniform(-1, 1, n).astype(np.float32)
    else:
        wave_s = np.sin(2 * np.pi * freq * t)
    return wave_s * _env(n) * vol


def _sweep(f0: float, f1: float, dur: float, vol: float = 0.35) -> np.ndarray:
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / max(dur, 1e-6))
    return np.sin(phase).astype(np.float32) * _env(n, 0.005, 0.12) * vol


def _mix(*parts: np.ndarray) -> np.ndarray:
    if not parts:
        return np.zeros(1, dtype=np.float32)
    max_len = max(p.shape[0] for p in parts)
    out = np.zeros(max_len, dtype=np.float32)
    for p in parts:
        out[: p.shape[0]] += p
    return out


def generate_all_sounds(force: bool = False) -> Dict[str, Path]:
    """Create WAV files once under assets/sounds/."""
    names = ("start", "slash", "fruit", "bomb", "gameover")
    paths = {name: SOUNDS_DIR / f"{name}.wav" for name in names}
    if not force and all(p.exists() for p in paths.values()):
        return paths

    # Neon START — bright rising chime
    start = _mix(
        _sweep(520, 980, 0.18, 0.32),
        np.pad(_tone(1175, 0.12, 0.28), (int(0.08 * SAMPLE_RATE), 0)),
        np.pad(_tone(1568, 0.16, 0.22), (int(0.14 * SAMPLE_RATE), 0)),
    )
    _write_wav(paths["start"], start)

    # Sword whoosh — noise + falling pitch
    n = int(SAMPLE_RATE * 0.16)
    noise = np.random.uniform(-1, 1, n).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    whoosh = _mix(noise * np.exp(-t * 18) * 0.45, _sweep(900, 220, 0.14, 0.2))
    _write_wav(paths["slash"], whoosh)

    # Fruit cut — short juicy click + high tick
    crunch = np.random.uniform(-1, 1, int(0.05 * SAMPLE_RATE)).astype(np.float32)
    crunch *= _env(crunch.shape[0], 0.001, 0.4) * 0.35
    fruit = _mix(
        crunch,
        _tone(880, 0.04, 0.28, "square"),
        np.pad(_tone(1320, 0.05, 0.22), (int(0.02 * SAMPLE_RATE), 0)),
    )
    _write_wav(paths["fruit"], fruit)

    # Bomb — deep boom + crackle
    crack = np.random.uniform(-1, 1, int(0.25 * SAMPLE_RATE)).astype(np.float32)
    crack *= np.exp(-np.linspace(0, 8, crack.shape[0])).astype(np.float32) * 0.4
    bomb = _mix(
        _sweep(180, 55, 0.35, 0.55),
        crack,
        _tone(90, 0.3, 0.35),
    )
    _write_wav(paths["bomb"], bomb)

    # Game over — descending sad tones
    go = _mix(
        _tone(440, 0.22, 0.35),
        np.pad(_tone(349, 0.28, 0.32), (int(0.18 * SAMPLE_RATE), 0)),
        np.pad(_tone(262, 0.45, 0.38), (int(0.4 * SAMPLE_RATE), 0)),
    )
    _write_wav(paths["gameover"], go)

    return paths


class SoundManager:
    """Lazy pygame mixer wrapper; safe no-op if audio init fails."""

    def __init__(self) -> None:
        self._ok = False
        self._sounds: Dict[str, object] = {}
        self._last_slash = 0.0
        try:
            import pygame

            generate_all_sounds()
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
            for name, path in generate_all_sounds().items():
                self._sounds[name] = pygame.mixer.Sound(str(path))
            # Volumes
            self._sounds["slash"].set_volume(0.35)
            self._sounds["fruit"].set_volume(0.55)
            self._sounds["bomb"].set_volume(0.75)
            self._sounds["start"].set_volume(0.7)
            self._sounds["gameover"].set_volume(0.8)
            self._ok = True
        except Exception as exc:
            print(f"Audio disabled ({exc}). Install pygame for sound effects.")

    def play(self, name: str) -> None:
        if not self._ok:
            return
        snd = self._sounds.get(name)
        if snd is not None:
            snd.play()

    def play_slash(self, now: float, min_gap: float = 0.09) -> None:
        if now - self._last_slash < min_gap:
            return
        self._last_slash = now
        self.play("slash")

    def close(self) -> None:
        if not self._ok:
            return
        try:
            import pygame

            pygame.mixer.stop()
        except Exception:
            pass
