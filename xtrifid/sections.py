"""Metric coordinates of an elliptical loft in its centred local frame."""

import numpy as np

from .geometry import rotation_matrix
from .model import Part


def bounds(rows):
    z, rx, ry, x, y = rows.T
    return np.array([(x - rx).min(), (y - ry).min(), z[0]]), np.array(
        [(x + rx).max(), (y + ry).max(), z[-1]]
    )


def local_sections(part):
    rows = np.array(part.sections)
    lo, hi = bounds(rows)
    scale = np.asarray(part.size) / (hi - lo)
    rows[:, [3, 4, 0]] -= (lo + hi) / 2
    return rows * scale[[2, 0, 1, 0, 1]]


def move_section(part, index, coordinates):
    rows = local_sections(part)
    rows[index] = coordinates
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("Высота сечения должна находиться между соседними сечениями")
    if np.any(rows[:, 1:3] < 0):
        raise ValueError("Радиусы сечения должны быть положительными или нулевыми")
    lo, hi = bounds(rows)
    span, mid = hi - lo, (hi + lo) / 2
    if np.any(span <= 0):
        raise ValueError("Сечения должны образовывать объём")
    rows[:, 3:5] -= mid[:2]
    rows[:, 0] -= lo[2]
    rows /= span[[2, 0, 1, 0, 1]]
    rows[0, 0], rows[-1, 0] = 0, 1
    center = rotation_matrix(part.rotation) @ mid + part.center
    return Part.parse(
        part.data()
        | {"size": span.tolist(), "center": center.tolist(), "sections": rows.tolist()}
    )
