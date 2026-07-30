"""Hand Gesture Fruit Ninja — main entry and game loop."""

from __future__ import annotations

import sys
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from audio import SoundManager
from game_objects import BladeTrail, ObjectManager
from hand_tracker import HandTracker

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BACKGROUND_PATH = ASSETS / "background.jpg"
HIGHSCORE_PATH = ROOT / "highscore.txt"

WINDOW_W = 1280
WINDOW_H = 720
WINDOW_NAME = "Fruit Ninja"
MAX_LIVES = 3
POINTS_PER_FRUIT = 10

PIP_W = 200
PIP_H = 140
PIP_MARGIN = 24

WHITE = (255, 255, 255)
OFFWHITE = (236, 238, 240)
DIM = (140, 145, 150)
INK = (18, 18, 20)
SURFACE = (28, 28, 32)
LINE = (55, 55, 60)
ACCENT = (80, 220, 120)
NEON = (40, 255, 90)       # neon green (BGR)
NEON_CORE = (200, 255, 220)
WARN = (70, 85, 255)
CAM_OK = (80, 220, 120)
CAM_BAD = (70, 85, 255)

START_BANNER_SEC = 1.15
BOMB_FLASH_SEC = 0.45
LIFE_POP_SEC = 0.9


class State(Enum):
    TITLE = auto()
    PLAYING = auto()
    GAME_OVER = auto()


def load_highscore() -> int:
    try:
        return int(HIGHSCORE_PATH.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def save_highscore(score: int) -> None:
    try:
        HIGHSCORE_PATH.write_text(str(score), encoding="utf-8")
    except OSError:
        pass


def load_background(width: int, height: int) -> np.ndarray:
    path = BACKGROUND_PATH
    if not path.exists():
        alt = ROOT / "images (1).jpg"
        path = alt if alt.exists() else BACKGROUND_PATH
    img = cv2.imread(str(path))
    if img is None:
        img = np.full((height, width, 3), (45, 58, 72), dtype=np.uint8)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.convertScaleAbs(img, alpha=1.12, beta=6)
    return img


def put(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.6,
    color: tuple = WHITE,
    thick: int = 1,
    center: bool = False,
) -> tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    if center:
        x -= tw // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return tw, th


def fill_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)


def stroke_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, t: int = 1) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, t, lineType=cv2.LINE_AA)


def draw_life_pip(frame: np.ndarray, x: int, y: int, on: bool) -> None:
    color = WARN if on else LINE
    cv2.circle(frame, (x, y), 9, color, -1, lineType=cv2.LINE_AA)
    if on:
        cv2.circle(frame, (x, y), 9, WHITE, 1, lineType=cv2.LINE_AA)


def draw_hud(frame: np.ndarray, score: int, highscore: int, lives: int) -> None:
    fill_rect(frame, 0, 0, WINDOW_W, 72, INK)
    cv2.line(frame, (0, 72), (WINDOW_W, 72), LINE, 1, lineType=cv2.LINE_AA)

    put(frame, "SCORE", (28, 28), scale=0.42, color=DIM, thick=1)
    put(frame, str(score), (28, 58), scale=1.05, color=WHITE, thick=2)

    put(frame, "BEST", (200, 28), scale=0.42, color=DIM, thick=1)
    put(frame, str(highscore), (200, 58), scale=0.85, color=OFFWHITE, thick=2)

    cv2.line(frame, (175, 18), (175, 54), LINE, 1, lineType=cv2.LINE_AA)

    put(frame, "LIVES", (WINDOW_W - 168, 28), scale=0.42, color=DIM, thick=1)
    for i in range(MAX_LIVES):
        draw_life_pip(frame, WINDOW_W - 140 + i * 36, 50, on=(i < lives))


def draw_camera_pip(frame: np.ndarray, cam_bgr: np.ndarray, tip: tuple | None) -> None:
    pip = cv2.resize(cam_bgr, (PIP_W, PIP_H), interpolation=cv2.INTER_AREA)

    if tip is not None:
        px = int(np.clip(tip[0] / WINDOW_W * PIP_W, 3, PIP_W - 3))
        py = int(np.clip(tip[1] / WINDOW_H * PIP_H, 3, PIP_H - 3))
        cv2.circle(pip, (px, py), 6, ACCENT, 2, lineType=cv2.LINE_AA)
        cv2.circle(pip, (px, py), 2, WHITE, -1, lineType=cv2.LINE_AA)
        label, lc = "LIVE", CAM_OK
    else:
        label, lc = "NO HAND", CAM_BAD

    x1 = WINDOW_W - PIP_W - PIP_MARGIN
    y1 = WINDOW_H - PIP_H - PIP_MARGIN - 28
    x2, y2 = x1 + PIP_W, y1 + PIP_H

    fill_rect(frame, x1, y1 - 28, x2, y1, SURFACE)
    put(frame, label, (x1 + 10, y1 - 9), scale=0.45, color=lc, thick=1)
    put(frame, "CAMERA", (x2 - 78, y1 - 9), scale=0.4, color=DIM, thick=1)

    frame[y1:y2, x1:x2] = pip
    stroke_rect(frame, x1, y1 - 28, x2, y2, LINE, 1)


def draw_title(frame: np.ndarray, highscore: int) -> None:
    frame[:] = (frame.astype(np.float32) * 0.28).astype(np.uint8)

    cx = WINDOW_W // 2
    card_w, card_h = 520, 360
    x1, y1 = cx - card_w // 2, WINDOW_H // 2 - card_h // 2 - 10
    x2, y2 = x1 + card_w, y1 + card_h
    fill_rect(frame, x1, y1, x2, y2, SURFACE)
    stroke_rect(frame, x1, y1, x2, y2, LINE, 1)
    fill_rect(frame, x1, y1, x2, y1 + 3, ACCENT)

    put(frame, "FRUIT NINJA", (cx, y1 + 88), scale=1.45, color=WHITE, thick=2, center=True)
    put(frame, "INDEX FINGER TO SLICE", (cx, y1 + 130), scale=0.5, color=DIM, thick=1, center=True)

    cv2.line(frame, (x1 + 48, y1 + 160), (x2 - 48, y1 + 160), LINE, 1, lineType=cv2.LINE_AA)
    put(frame, "HIGH SCORE", (cx, y1 + 195), scale=0.4, color=DIM, thick=1, center=True)
    put(frame, str(highscore), (cx, y1 + 240), scale=1.2, color=ACCENT, thick=2, center=True)

    bx1, by1 = cx - 130, y1 + 275
    bx2, by2 = cx + 130, y1 + 325
    fill_rect(frame, bx1, by1, bx2, by2, ACCENT)
    put(frame, "SPACE  TO  START", (cx, by1 + 34), scale=0.55, color=INK, thick=2, center=True)


def draw_game_over(frame: np.ndarray, score: int, highscore: int, is_new: bool) -> None:
    frame[:] = (frame.astype(np.float32) * 0.25).astype(np.uint8)

    cx = WINDOW_W // 2
    card_w, card_h = 480, 340
    x1, y1 = cx - card_w // 2, WINDOW_H // 2 - card_h // 2
    x2, y2 = x1 + card_w, y1 + card_h
    fill_rect(frame, x1, y1, x2, y2, SURFACE)
    stroke_rect(frame, x1, y1, x2, y2, LINE, 1)
    fill_rect(frame, x1, y1, x2, y1 + 3, WARN)

    put(frame, "GAME OVER", (cx, y1 + 70), scale=1.1, color=WHITE, thick=2, center=True)
    put(frame, "SCORE", (cx, y1 + 120), scale=0.4, color=DIM, thick=1, center=True)
    put(frame, str(score), (cx, y1 + 175), scale=1.5, color=WHITE, thick=2, center=True)

    if is_new:
        put(frame, "NEW BEST", (cx, y1 + 220), scale=0.55, color=ACCENT, thick=2, center=True)
    else:
        put(frame, f"BEST  {highscore}", (cx, y1 + 220), scale=0.5, color=DIM, thick=1, center=True)

    put(frame, "R  RESTART      Q  QUIT", (cx, y1 + 280), scale=0.5, color=OFFWHITE, thick=1, center=True)


def draw_start_banner(frame: np.ndarray, age: float) -> None:
    """Neon green START! flash when the round begins."""
    if age < 0 or age > START_BANNER_SEC:
        return
    # Peak early, fade out
    if age < 0.15:
        strength = age / 0.15
    else:
        strength = max(0.0, 1.0 - (age - 0.15) / (START_BANNER_SEC - 0.15))

    # Soft green wash
    wash = np.zeros_like(frame)
    wash[:] = (20, 80, 30)
    cv2.addWeighted(wash, 0.18 * strength, frame, 1.0 - 0.18 * strength, 0, frame)

    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    scale = 2.4 + 0.35 * (1.0 - strength)
    text = "START!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 5
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = cx - tw // 2
    y = cy + th // 2

    # Neon layers (outer glow → core)
    glow_cols = [
        ((0, 90, 30), thick + 14),
        ((20, 180, 60), thick + 8),
        (NEON, thick + 3),
        (NEON_CORE, max(2, thick - 1)),
        (WHITE, 1),
    ]
    for col, tk in glow_cols:
        # Dim outer layers by strength
        c = tuple(int(v * (0.5 + 0.5 * strength)) for v in col)
        cv2.putText(frame, text, (x, y), font, scale, c, tk, cv2.LINE_AA)

    # Underline accent
    line_w = int(tw * 0.55 * strength)
    cv2.line(frame, (cx - line_w, y + 18), (cx + line_w, y + 18), NEON, 3, lineType=cv2.LINE_AA)


def apply_bomb_flash(frame: np.ndarray, age: float) -> None:
    """Full-screen destructive white flash after striking a bomb."""
    if age < 0 or age > BOMB_FLASH_SEC:
        return
    # Hard white peak then red-tinged fade
    if age < 0.08:
        a = 0.92 * (1.0 - age / 0.08 * 0.15)
        flash = np.full_like(frame, 255)
        cv2.addWeighted(flash, a, frame, 1.0 - a, 0, frame)
    else:
        t = (age - 0.08) / (BOMB_FLASH_SEC - 0.08)
        a = 0.55 * (1.0 - t)
        flash = np.zeros_like(frame)
        flash[:] = (220, 230, 255)  # white → slight cool
        cv2.addWeighted(flash, a, frame, 1.0 - a, 0, frame)
        # Edge vignette crack (dark corners)
        if t < 0.7:
            cv2.rectangle(frame, (0, 0), (WINDOW_W - 1, WINDOW_H - 1), (255, 255, 255), 4, lineType=cv2.LINE_AA)


def draw_life_lost(frame: np.ndarray, age: float, lives_left: int) -> None:
    if age < 0 or age > LIFE_POP_SEC:
        return
    strength = max(0.0, 1.0 - age / LIFE_POP_SEC)
    cy = 110
    put(
        frame,
        "-1 LIFE",
        (WINDOW_W - 120, cy),
        scale=0.75 + 0.2 * strength,
        color=tuple(int(c * strength + 40 * (1 - strength)) for c in WARN),
        thick=2,
        center=True,
    )
    put(
        frame,
        f"{lives_left} LEFT",
        (WINDOW_W - 120, cy + 32),
        scale=0.45,
        color=tuple(int(200 * strength) for _ in range(3)),
        thick=1,
        center=True,
    )


def open_camera() -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def read_latest_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    ok = cap.grab()
    if not ok:
        return False, None
    cap.grab()
    return cap.retrieve()


def begin_round(sfx: SoundManager) -> tuple:
    sfx.play("start")
    now = time.perf_counter()
    return (
        0,  # score
        MAX_LIVES,
        False,  # new_high
        now,  # start_banner_t
        -1.0,  # bomb_flash_t
        -1.0,  # life_lost_t
    )


def main() -> None:
    background = load_background(WINDOW_W, WINDOW_H)
    highscore = load_highscore()
    sfx = SoundManager()

    cap = open_camera()
    if not cap.isOpened():
        print("Error: could not open webcam. Connect a camera and try again.")
        return

    tracker = HandTracker(max_hands=1)
    objects = ObjectManager(WINDOW_W, WINDOW_H)
    blade = BladeTrail()

    state = State.TITLE
    score = 0
    lives = MAX_LIVES
    new_high = False
    start_banner_t = -1.0
    bomb_flash_t = -1.0
    life_lost_t = -1.0
    last_t = time.perf_counter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    try:
        while True:
            now = time.perf_counter()
            dt = min(0.05, now - last_t)
            last_t = now

            ok, cam = read_latest_frame(cap)
            if not ok or cam is None:
                print("Warning: failed to read webcam frame.")
                break

            cam = cv2.flip(cam, 1)
            tip = tracker.update(cam, WINDOW_W, WINDOW_H)
            speed = tracker.speed()
            prev = tracker.prev_tip

            frame = background.copy()

            if state == State.PLAYING:
                objects.update(dt, score)
                fruits, bombs, _ = objects.try_slice(tip, prev, speed)

                if speed >= 22:
                    sfx.play_slash(now)

                if fruits:
                    score += fruits * POINTS_PER_FRUIT
                    if score > highscore:
                        highscore = score
                    for _ in range(fruits):
                        sfx.play("fruit")

                if bombs:
                    lives -= bombs
                    bomb_flash_t = now
                    life_lost_t = now
                    sfx.play("bomb")
                    if lives <= 0:
                        lives = 0
                        saved = load_highscore()
                        if score > saved:
                            save_highscore(score)
                            highscore = score
                            new_high = True
                        else:
                            highscore = max(highscore, saved)
                            new_high = False
                        state = State.GAME_OVER
                        blade.clear()
                        sfx.play("gameover")

                blade.add(tip)
                objects.draw(frame)
                blade.draw(frame)
                draw_hud(frame, score, highscore, lives)

                if bomb_flash_t > 0:
                    apply_bomb_flash(frame, now - bomb_flash_t)
                if life_lost_t > 0 and lives >= 0:
                    draw_life_lost(frame, now - life_lost_t, max(0, lives))
                if start_banner_t > 0:
                    draw_start_banner(frame, now - start_banner_t)

            elif state == State.TITLE:
                blade.add(None)
                draw_title(frame, highscore)

            elif state == State.GAME_OVER:
                blade.add(None)
                objects.draw(frame)
                if bomb_flash_t > 0 and (now - bomb_flash_t) < BOMB_FLASH_SEC:
                    apply_bomb_flash(frame, now - bomb_flash_t)
                draw_game_over(frame, score, highscore, new_high)

            draw_camera_pip(frame, cam, tip)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if state == State.TITLE and key == ord(" "):
                objects.reset()
                blade.clear()
                score, lives, new_high, start_banner_t, bomb_flash_t, life_lost_t = begin_round(sfx)
                state = State.PLAYING
            if state == State.GAME_OVER and key in (ord("r"), ord("R")):
                objects.reset()
                blade.clear()
                highscore = load_highscore()
                score, lives, new_high, start_banner_t, bomb_flash_t, life_lost_t = begin_round(sfx)
                state = State.PLAYING

    finally:
        tracker.close()
        sfx.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
