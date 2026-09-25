import numpy as np


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def intersect(a_start, a_end, b_start, b_end) -> tuple[np.ndarray, np.ndarray]:
    a_dir = a_end - a_start
    b_dir = b_end - b_start
    denominator = cross(a_dir, b_dir)
    offset = b_start - a_start
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross(offset, b_dir) / denominator
        u = cross(offset, a_dir) / denominator
    hit = (denominator != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return hit, t


def cast_rays(origins, angles_deg, max_range, wall_starts, wall_ends) -> np.ndarray:
    angles = np.radians(angles_deg)
    directions = np.stack([np.sin(angles), -np.cos(angles)], axis=-1)
    ray_starts = origins[:, None, None, :]
    ray_ends = (origins[:, None, :] + directions * max_range)[:, :, None, :]
    if len(wall_starts) == 0:
        return np.full(angles.shape, max_range)
    hit, t = intersect(ray_starts, ray_ends, wall_starts[None, None], wall_ends[None, None])
    nearest = np.where(hit, t, 1.0).min(axis=2)
    return nearest * max_range


def car_corners(x, y, rotation_deg, length, width) -> np.ndarray:
    heading = np.radians(rotation_deg)
    forward = np.stack([np.sin(heading), -np.cos(heading)], axis=-1) * (length / 2)
    side = np.stack([np.cos(heading), np.sin(heading)], axis=-1) * (width / 2)
    center = np.stack([x, y], axis=-1)
    return np.stack(
        [center + forward + side, center + forward - side, center - forward - side, center - forward + side],
        axis=1,
    )


def boxes_hit_walls(corners, wall_starts, wall_ends) -> np.ndarray:
    if len(wall_starts) == 0 or len(corners) == 0:
        return np.zeros(len(corners), dtype=bool)
    edge_starts = corners[:, :, None, :]
    edge_ends = np.roll(corners, -1, axis=1)[:, :, None, :]
    hit, _ = intersect(edge_starts, edge_ends, wall_starts[None, None], wall_ends[None, None])
    return hit.any(axis=(1, 2))


def hermite(t: np.ndarray, p0: float, p1: float, m0: float, m1: float) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1
