"""Fruits, bombs, blade trail, blast particles, and slice collision."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

FRUIT_TYPES = {
    # Flat 2-tone fills — crisp, not glossy cartoon
    "apple": {"fill": (70, 70, 230), "shade": (50, 50, 180), "leaf": (70, 200, 90), "radius": 34},
    "orange": {"fill": (50, 165, 255), "shade": (30, 120, 210), "leaf": (70, 200, 90), "radius": 32},
    "watermelon": {"fill": (70, 190, 70), "shade": (40, 130, 45), "stripe": (35, 100, 35), "radius": 40},
    "lemon": {"fill": (55, 235, 255), "shade": (30, 185, 220), "leaf": (70, 200, 90), "radius": 30},
    "grape": {"fill": (190, 70, 170), "shade": (140, 40, 120), "leaf": (70, 200, 90), "radius": 28},
}

GRAVITY = 980.0
SLICE_SPEED_MIN = 12.0
TRAIL_LIFETIME = 0.14
BLAST_DURATION = 0.65


@dataclass
class FlyingObject:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    kind: str
    fruit_name: Optional[str] = None
    alive: bool = True
    sliced: bool = False
    halves: List[dict] = field(default_factory=list)
    fade: float = 1.0
    rot: float = 0.0
    spin: float = 0.0

    def update(self, dt: float) -> None:
        if self.halves:
            for h in self.halves:
                h["x"] += h["vx"] * dt
                h["y"] += h["vy"] * dt
                h["vy"] += GRAVITY * dt
                h["fade"] = max(0.0, h["fade"] - dt * 2.8)
            self.fade = max(h["fade"] for h in self.halves) if self.halves else 0.0
            if self.fade <= 0.0:
                self.alive = False
            return

        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.rot += self.spin * dt

    def off_screen(self, width: int, height: int) -> bool:
        if self.halves:
            return self.fade <= 0.0
        return self.y - self.radius > height + 40 or self.x < -80 or self.x > width + 80

    def slice_fruit(self, tip: Point, prev: Point) -> None:
        self.sliced = True
        dx = tip[0] - prev[0]
        dy = tip[1] - prev[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        speed = 240.0
        name = self.fruit_name or "apple"
        style = FRUIT_TYPES[name]
        self.halves = [
            {
                "x": self.x,
                "y": self.y,
                "vx": self.vx + nx * speed,
                "vy": self.vy - 90,
                "fade": 1.0,
                "side": -1,
                "style": style,
                "name": name,
                "radius": self.radius,
            },
            {
                "x": self.x,
                "y": self.y,
                "vx": self.vx - nx * speed,
                "vy": self.vy - 90,
                "fade": 1.0,
                "side": 1,
                "style": style,
                "name": name,
                "radius": self.radius,
            },
        ]


@dataclass
class BlastParticle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: Tuple[int, int, int]
    life: float
    max_life: float


@dataclass
class BlastEffect:
    """Destructive bomb blast — white-hot shards + expanding shock rings."""

    x: float
    y: float
    start: float
    particles: List[BlastParticle] = field(default_factory=list)

    @classmethod
    def create(cls, x: float, y: float) -> "BlastEffect":
        now = time.time()
        colors = [
            (255, 255, 255),
            (240, 250, 255),
            (180, 230, 255),
            (80, 180, 255),
            (40, 120, 255),
        ]
        particles: List[BlastParticle] = []
        for _ in range(42):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(220, 720)
            life = random.uniform(0.25, BLAST_DURATION)
            particles.append(
                BlastParticle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    radius=random.uniform(3, 16),
                    color=random.choice(colors),
                    life=life,
                    max_life=life,
                )
            )
        return cls(x=x, y=y, start=now, particles=particles)

    def update(self, dt: float) -> None:
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += 180 * dt
            p.life -= dt
            p.radius *= 0.96

    def alive(self) -> bool:
        return (time.time() - self.start) < BLAST_DURATION + 0.15

    def draw(self, frame: np.ndarray) -> None:
        t = time.time() - self.start
        if t < 0.35:
            for k, mul in enumerate((1.0, 0.65, 0.4)):
                ring_r = int(30 + t * 900 * mul)
                alpha_line = max(1, int(4 * (1.0 - t / 0.35)))
                cv2.circle(
                    frame,
                    (int(self.x), int(self.y)),
                    ring_r,
                    (255, 255, 255) if k == 0 else (200, 230, 255),
                    alpha_line,
                    lineType=cv2.LINE_AA,
                )
        for p in self.particles:
            if p.life <= 0:
                continue
            alpha = max(0.0, p.life / p.max_life)
            r = max(1, int(p.radius * (0.5 + alpha)))
            color = tuple(int(c * (0.55 + 0.45 * alpha)) for c in p.color)
            cv2.circle(frame, (int(p.x), int(p.y)), r, color, -1, lineType=cv2.LINE_AA)


@dataclass
class TrailPoint:
    x: float
    y: float
    t: float


class BladeTrail:
    """Thin, sharp slash stroke — no soft glow blobs."""

    def __init__(self, lifetime: float = TRAIL_LIFETIME):
        self.lifetime = lifetime
        self.points: Deque[TrailPoint] = deque(maxlen=20)

    def add(self, tip: Optional[Point]) -> None:
        now = time.time()
        if tip is not None:
            self.points.append(TrailPoint(tip[0], tip[1], now))
        while self.points and now - self.points[0].t > self.lifetime:
            self.points.popleft()

    def clear(self) -> None:
        self.points.clear()

    def draw(self, frame: np.ndarray) -> None:
        if len(self.points) < 2:
            return
        now = time.time()
        pts = list(self.points)
        for i in range(1, len(pts)):
            age = now - pts[i].t
            alpha = max(0.0, 1.0 - age / self.lifetime)
            thickness = max(1, int(5 * alpha))
            p1 = (int(pts[i - 1].x), int(pts[i - 1].y))
            p2 = (int(pts[i].x), int(pts[i].y))
            # Crisp white core only
            cv2.line(frame, p1, p2, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
            if thickness >= 3:
                cv2.line(frame, p1, p2, (200, 255, 220), 1, lineType=cv2.LINE_AA)


def _point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def segment_hits_circle(p1: Point, p2: Point, cx: float, cy: float, radius: float) -> bool:
    if math.hypot(p1[0] - cx, p1[1] - cy) <= radius:
        return True
    if math.hypot(p2[0] - cx, p2[1] - cy) <= radius:
        return True
    return _point_segment_distance(cx, cy, p1[0], p1[1], p2[0], p2[1]) <= radius


def spawn_object(width: int, height: int, prefer_bomb_chance: float = 0.2) -> FlyingObject:
    is_bomb = random.random() < prefer_bomb_chance
    x = random.uniform(width * 0.15, width * 0.85)
    y = height + 30
    target_x = random.uniform(width * 0.25, width * 0.75)
    flight_time = random.uniform(0.9, 1.3)
    vx = (target_x - x) / flight_time
    vy = -random.uniform(640, 840)
    spin = random.uniform(-4.0, 4.0)

    if is_bomb:
        return FlyingObject(x=x, y=y, vx=vx, vy=vy, radius=32, kind="bomb", spin=spin)

    name = random.choice(list(FRUIT_TYPES.keys()))
    style = FRUIT_TYPES[name]
    return FlyingObject(
        x=x, y=y, vx=vx, vy=vy, radius=float(style["radius"]), kind="fruit", fruit_name=name, spin=spin
    )


def draw_fruit_circle(frame: np.ndarray, x: int, y: int, radius: int, name: str, alpha: float = 1.0) -> None:
    style = FRUIT_TYPES[name]
    fill = style["fill"]
    shade = style["shade"]

    if alpha < 0.99:
        overlay = frame.copy()
        target = overlay
    else:
        target = frame

    # Flat modern fruit — solid body + small hard highlight (no soft shadow)
    cv2.circle(target, (x, y), radius, fill, -1, lineType=cv2.LINE_AA)
    # Bottom shade crescent (hard edge, not blur)
    cv2.ellipse(target, (x, y + radius // 5), (int(radius * 0.85), int(radius * 0.7)), 0, 20, 160, shade, -1, lineType=cv2.LINE_AA)
    # Tiny specular — sharp
    cv2.circle(target, (x - radius // 3, y - radius // 3), max(2, radius // 7), (255, 255, 255), -1, lineType=cv2.LINE_AA)

    if name == "watermelon":
        for i in range(-2, 3):
            ang = i * 0.32
            x1 = int(x + math.cos(ang) * radius * 0.85)
            y1 = int(y + math.sin(ang) * radius * 0.85)
            x2 = int(x + math.cos(ang + math.pi) * radius * 0.85)
            y2 = int(y + math.sin(ang + math.pi) * radius * 0.85)
            cv2.line(target, (x1, y1), (x2, y2), style["stripe"], 2, lineType=cv2.LINE_AA)
    else:
        cv2.line(target, (x, y - radius + 1), (x, y - radius - 8), (45, 100, 55), 2, lineType=cv2.LINE_AA)
        leaf = style.get("leaf", (70, 200, 90))
        cv2.ellipse(target, (x + 6, y - radius - 4), (8, 4), -30, 0, 360, leaf, -1, lineType=cv2.LINE_AA)

    if alpha < 0.99:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_fruit_half(frame: np.ndarray, h: dict) -> None:
    x, y = int(h["x"]), int(h["y"])
    r = int(h["radius"])
    alpha = float(h["fade"])
    name = h["name"]
    side = h["side"]
    overlay = frame.copy()
    draw_fruit_circle(overlay, x, y, r, name, alpha=1.0)
    if side < 0:
        pts = np.array([[x, y - r - 2], [x + r + 2, y - r - 2], [x + r + 2, y + r + 2], [x, y + r + 2]], np.int32)
    else:
        pts = np.array([[x, y - r - 2], [x - r - 2, y - r - 2], [x - r - 2, y + r + 2], [x, y + r + 2]], np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    overlay[mask == 255] = frame[mask == 255]
    cv2.line(overlay, (x, y - r + 2), (x, y + r - 2), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_bomb(frame: np.ndarray, x: int, y: int, radius: int) -> None:
    # Flat dark disc + thin red ring
    cv2.circle(frame, (x, y), radius, (32, 32, 36), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x, y), radius, (70, 85, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x - radius // 3, y - radius // 3), max(2, radius // 6), (90, 90, 98), -1, lineType=cv2.LINE_AA)
    fuse_end = (x + radius // 2 + 8, y - radius - 6)
    cv2.line(frame, (x + radius // 4, y - radius + 2), fuse_end, (120, 120, 128), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, fuse_end, 5, (70, 200, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, fuse_end, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def draw_object(frame: np.ndarray, obj: FlyingObject) -> None:
    if obj.halves:
        for h in obj.halves:
            if h["fade"] > 0:
                draw_fruit_half(frame, h)
        return
    x, y = int(obj.x), int(obj.y)
    if obj.kind == "bomb":
        draw_bomb(frame, x, y, int(obj.radius))
    else:
        draw_fruit_circle(frame, x, y, int(obj.radius), obj.fruit_name or "apple")


class ObjectManager:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.objects: List[FlyingObject] = []
        self.blasts: List[BlastEffect] = []
        self._spawn_timer = 0.0
        self._next_spawn = random.uniform(0.55, 0.95)

    def reset(self) -> None:
        self.objects.clear()
        self.blasts.clear()
        self._spawn_timer = 0.0
        self._next_spawn = random.uniform(0.55, 0.95)

    def spawn_interval(self, score: int) -> float:
        base = max(0.42, 1.05 - score * 0.004)
        return random.uniform(base * 0.75, base * 1.05)

    def update(self, dt: float, score: int) -> None:
        self._spawn_timer += dt
        if self._spawn_timer >= self._next_spawn:
            bomb_chance = min(0.28, 0.18 + score * 0.0005)
            count = 1 if random.random() < 0.72 else 2
            for _ in range(count):
                self.objects.append(spawn_object(self.width, self.height, bomb_chance))
            self._spawn_timer = 0.0
            self._next_spawn = self.spawn_interval(score)

        for obj in self.objects:
            obj.update(dt)
        self.objects = [o for o in self.objects if o.alive and not o.off_screen(self.width, self.height)]

        for blast in self.blasts:
            blast.update(dt)
        self.blasts = [b for b in self.blasts if b.alive()]

    def try_slice(
        self, tip: Optional[Point], prev: Optional[Point], speed: float
    ) -> Tuple[int, int, List[BlastEffect]]:
        fruits = 0
        bombs = 0
        new_blasts: List[BlastEffect] = []
        if tip is None or prev is None or speed < SLICE_SPEED_MIN:
            return 0, 0, new_blasts

        for obj in self.objects:
            if not obj.alive or obj.sliced or obj.halves:
                continue
            if segment_hits_circle(prev, tip, obj.x, obj.y, obj.radius + 8):
                if obj.kind == "fruit":
                    obj.slice_fruit(tip, prev)
                    fruits += 1
                else:
                    obj.alive = False
                    obj.sliced = True
                    blast = BlastEffect.create(obj.x, obj.y)
                    self.blasts.append(blast)
                    new_blasts.append(blast)
                    bombs += 1
        return fruits, bombs, new_blasts

    def draw(self, frame: np.ndarray) -> None:
        for obj in self.objects:
            draw_object(frame, obj)
        for blast in self.blasts:
            blast.draw(frame)
