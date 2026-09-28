"""Shared polyline sampling utilities for route-constraint checks."""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple

from modules.config import LAT_TO_KM, SRTM_RESOLUTION


DEFAULT_SAMPLE_STEP_KM = SRTM_RESOLUTION / 1000.0


def horizontal_distance_km(a: Sequence[float], b: Sequence[float]) -> float:
    """Approximate horizontal distance between WGS-84 points in kilometres."""
    mean_lat_rad = math.radians((float(a[0]) + float(b[0])) * 0.5)
    dlat_km = (float(a[0]) - float(b[0])) * LAT_TO_KM
    dlon_km = (float(a[1]) - float(b[1])) * LAT_TO_KM * math.cos(mean_lat_rad)
    return math.hypot(dlat_km, dlon_km)


def resample_path(
    path: Iterable[Sequence[float]],
    max_horizontal_step_km: float = DEFAULT_SAMPLE_STEP_KM,
    max_vertical_step_m: float = 100.0,
) -> List[Tuple[float, ...]]:
    """Linearly resample every path segment at bounded horizontal/vertical gaps."""
    points = [tuple(float(value) for value in point) for point in path]
    if not points:
        return []
    if len(points) == 1:
        return points
    if max_horizontal_step_km <= 0.0 or max_vertical_step_m <= 0.0:
        raise ValueError("resampling steps must be positive")

    sampled: List[Tuple[float, ...]] = [points[0]]
    for start, end in zip(points, points[1:]):
        dimensions = min(len(start), len(end))
        if dimensions < 2:
            raise ValueError("path points must contain latitude and longitude")
        horizontal_steps = math.ceil(
            horizontal_distance_km(start, end) / max_horizontal_step_km
        )
        vertical_steps = 1
        if dimensions >= 3:
            vertical_steps = math.ceil(abs(end[2] - start[2]) / max_vertical_step_m)
        steps = max(1, horizontal_steps, vertical_steps)
        for index in range(1, steps + 1):
            fraction = index / steps
            sampled.append(
                tuple(
                    start[axis] + fraction * (end[axis] - start[axis])
                    for axis in range(dimensions)
                )
            )
    return sampled


def path_satisfies(
    path: Iterable[Sequence[float]],
    point_is_valid: Callable[[Sequence[float]], bool],
    max_horizontal_step_km: float = DEFAULT_SAMPLE_STEP_KM,
) -> bool:
    """Return true only when every resampled point satisfies the constraint."""
    return all(
        point_is_valid(point)
        for point in resample_path(path, max_horizontal_step_km=max_horizontal_step_km)
    )
