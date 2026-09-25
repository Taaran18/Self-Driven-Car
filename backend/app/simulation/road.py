import math
import random

import numpy as np

from app.simulation.constants import (
    LOOKAHEAD,
    MAX_SLOPE,
    POINTS_PER_SEGMENT,
    RUNOUT,
    SEGMENT_LENGTH,
    TRAIL,
)
from app.simulation.geometry import hermite


class Road:
    def __init__(self, rng: random.Random, width: float, curviness: float, length: float):
        self.rng = rng
        self.width = width
        self.curviness = curviness
        self.finish_y = -length
        self.left: list[tuple[float, float]] = []
        self.right: list[tuple[float, float]] = []
        self.version = 0
        self._last_center: tuple[float, float] | None = None
        self._control = (0.0, 0.0, 0.0)
        self._arrays: tuple[np.ndarray, np.ndarray] | None = None

        for i in range(POINTS_PER_SEGMENT * 2 + 1):
            self._add_center_point(0.0, SEGMENT_LENGTH * 2 - i * SEGMENT_LENGTH / POINTS_PER_SEGMENT)
        self.extend_to(0.0)

    @property
    def end_y(self) -> float:
        return self._control[1]

    def extend_to(self, leader_y: float) -> None:
        changed = False
        while self._control[1] > max(leader_y - LOOKAHEAD, self.finish_y - RUNOUT):
            self._add_segment(straight=self._control[1] <= self.finish_y)
            changed = True
        if changed:
            self._changed()

    def trim_behind(self, rear_y: float) -> None:
        cutoff = rear_y + TRAIL
        drop = 0
        while drop < len(self.left) - 2 and self.left[drop][1] > cutoff and self.right[drop][1] > cutoff:
            drop += 1
        if drop > POINTS_PER_SEGMENT:
            del self.left[:drop]
            del self.right[:drop]
            self._changed()

    def walls_between(self, y_min: float, y_max: float) -> tuple[np.ndarray, np.ndarray]:
        left, right = self.arrays()
        starts, ends = [], []
        for side in (left, right):
            a, b = side[:-1], side[1:]
            near = (np.maximum(a[:, 1], b[:, 1]) >= y_min) & (np.minimum(a[:, 1], b[:, 1]) <= y_max)
            starts.append(a[near])
            ends.append(b[near])
        return np.concatenate(starts), np.concatenate(ends)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._arrays is None:
            self._arrays = (np.asarray(self.left, dtype=float), np.asarray(self.right, dtype=float))
        return self._arrays

    def snapshot(self) -> dict:
        left, right = self.arrays()
        return {
            "version": self.version,
            "left": np.round(left, 1).ravel().tolist(),
            "right": np.round(right, 1).ravel().tolist(),
            "finish_y": self.finish_y,
            "width": self.width,
        }

    def _changed(self) -> None:
        self.version += 1
        self._arrays = None

    def _add_segment(self, straight: bool) -> None:
        x1, y1, slope1 = self._control
        if straight:
            x2, slope2 = x1, 0.0
        else:
            x2 = x1 + (self.rng.random() - 0.5) * self.curviness
            slope2 = MAX_SLOPE * (self.rng.random() - 0.5)
        y2 = y1 - SEGMENT_LENGTH
        t = np.arange(1, POINTS_PER_SEGMENT + 1) / POINTS_PER_SEGMENT
        xs = hermite(t, x1, x2, slope1 * (y2 - y1), slope2 * (y2 - y1))
        ys = y1 + t * (y2 - y1)
        for x, y in zip(xs, ys, strict=False):
            self._add_center_point(float(x), float(y))
        self._control = (x2, y2, slope2)

    def _add_center_point(self, x: float, y: float) -> None:
        previous = self._last_center
        angle = 0.0 if previous is None else math.atan2(x - previous[0], previous[1] - y)
        dx = self.width / 2 * math.cos(angle)
        dy = self.width / 2 * math.sin(angle)
        left = (x - dx, y - dy)
        right = (x + dx, y + dy)
        if self.left and left[1] >= self.left[-1][1]:
            left = self.left[-1]
        if self.right and right[1] >= self.right[-1][1]:
            right = self.right[-1]
        self.left.append(left)
        self.right.append(right)
        self._last_center = (x, y)
