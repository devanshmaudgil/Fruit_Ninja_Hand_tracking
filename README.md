# Hand Gesture Fruit Ninja

A webcam Fruit Ninja-style game controlled by your **index finger**. Fruits and bombs arc across a wooden playfield; a neon sword trail follows your fingertip. Slice fruit for points — hit a bomb and you lose a life.

Built with **Python**, **OpenCV**, **MediaPipe**, and **Pygame**.

---

## GitHub description (paste into repo About)

```
Play Fruit Ninja with your webcam — slice flying fruit with your index finger using MediaPipe hand tracking, OpenCV rendering, and custom SFX.
```

**Topics / tags:** `python` `opencv` `mediapipe` `hand-tracking` `computer-vision` `game` `fruit-ninja` `pygame` `webcam`

---

## Demo controls

| Input | Action |
|-------|--------|
| Index finger (fast swipe) | Slash |
| `SPACE` | Start (neon **START!** + chime) |
| `R` | Restart after game over |
| `Q` / `Esc` | Quit |

## Rules

- **+10** per fruit sliced  
- **3 lives** — lose one when you hit a bomb (white flash + blast)  
- Missing fruit does **not** cost a life  
- High score persists in `highscore.txt`

---

## How we built this

### Idea

Classic Fruit Ninja, but the “blade” is your real hand. The webcam is used only for tracking; the game draws on a custom wooden background so the playfield stays thematic and readable.

### Stack

| Piece | Role |
|-------|------|
| **OpenCV** | Window, sprites, HUD, camera PiP, effects |
| **MediaPipe Tasks (`HandLandmarker`)** | Index fingertip (landmark 8) in real time |
| **NumPy** | Physics helpers, procedural audio samples |
| **Pygame mixer** | Playback of generated WAV sound effects |

### Architecture

```
Webcam frame
    → mirror + downscale (320×240)
    → MediaPipe HandLandmarker (LIVE_STREAM, async)
    → index tip (x, y) + velocity prediction
    → blade trail + slice collision
    → OpenCV compose: wood background + fruits/bombs + HUD + PiP
```

```
Hand_gesture_fruit_Ninja/
├── main.py              # Game loop, screens, HUD, start/bomb FX
├── hand_tracker.py      # Low-latency MediaPipe tip tracking
├── game_objects.py      # Fruits, bombs, physics, blade, blasts
├── audio.py             # Procedural SFX + pygame playback
├── assets/
│   ├── background.jpg   # Wooden playfield
│   └── sounds/          # start, slash, fruit, bomb, gameover
├── models/
│   └── hand_landmarker.task
├── highscore.txt
├── requirements.txt
└── README.md
```

### 1. Hand tracking (low latency)

MediaPipe’s older `mp.solutions.hands` API is gone in current packages, so we use **`HandLandmarker` in `LIVE_STREAM` mode**:

- Inference runs **async** so the game loop is not blocked  
- Frames are **downscaled** before detection  
- Camera buffer is kept short (DirectShow on Windows)  
- Tip position is **lightly predicted** from velocity to hide model lag  
- **No skeleton** is drawn — only a thin sword trail and a tip marker on the camera PiP  

### 2. Gameplay loop

`main.py` owns states: **Title → Playing → Game Over**.

Each frame:

1. Grab the newest webcam frame  
2. Update fingertip from `HandTracker`  
3. Spawn / move fruits & bombs with gravity arcs (`game_objects.py`)  
4. Test slash segments against object radii (speed gate avoids idle “slices”)  
5. Draw background → objects → blade → HUD → overlays → camera PiP  

### 3. Fruits, bombs, and effects

- Fruits are **flat, procedural** circles (no external sprite packs)  
- Bombs trigger a **white screen flash**, shock rings, particle blast, and a **-1 LIFE** popup  
- Pressing **SPACE** shows a neon green **START!** banner  

### 4. Audio

`audio.py` **synthesizes WAV files** on first run (chime, whoosh, cut, boom, game-over) and plays them with Pygame — no large binary sound packs required in the repo.

### 5. Visual design choices

- Sharp wooden background (no heavy blur/vignette HUD)  
- Flat top bar for score / best / lives  
- Camera preview in the corner so you can confirm tracking without covering the playfield  

---

## Setup

**Requirements:** Python 3.9+, a webcam.

```bash
git clone https://github.com/devanshmaudgil/Fruit_Ninja_Hand_tracking.git
cd Fruit_Ninja_Hand_tracking
pip install -r requirements.txt
python main.py
```

### Dependencies

```
opencv-python
mediapipe
numpy
pygame
```

The hand model lives at `models/hand_landmarker.task`. If missing, download Google’s MediaPipe `hand_landmarker.task` into that folder.

Sounds appear under `assets/sounds/` after the first launch.

---

## Tips for better play

- Good lighting on your hand  
- Stand far enough that the full swipe fits in frame  
- Slash **quickly** across fruit — slow hovering won’t count  
- Avoid bombs; three hits end the run  

---

## What we learned / tradeoffs

- **Async + small inference size** mattered more for feel than raw accuracy  
- Drawing a custom background (instead of raw webcam) made fruit contrast and theming easier  
- Procedural art and SFX kept the project self-contained for a same-day build  

---

## License

Add your preferred license (e.g. MIT) when you publish the repo. Background art and the MediaPipe model follow their own upstream terms.
