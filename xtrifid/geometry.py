"""Closed Z-up part meshes. All faces use outward counterclockwise winding."""

from dataclasses import dataclass
from functools import lru_cache
from math import cos, pi, radians, sin, tau

import numpy as np

from .model import MODES
from .polygon import triangulate, turn


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: tuple
    smooth: tuple


def triangles(faces):
    for face in faces:
        for i in range(1, len(face) - 1):
            yield (face[0], face[i], face[i + 1])


def bridge(lower, upper):
    """Connect two rings or terminate one at a single pole."""
    n = max(len(lower), len(upper))
    for i in range(n):
        j = (i + 1) % n
        if len(lower) == 1:
            yield (lower[0], upper[j], upper[i])
        elif len(upper) == 1:
            yield (lower[i], lower[j], upper[0])
        else:
            yield (lower[i], lower[j], upper[j], upper[i])


def radial(shape, n, outline):
    vertices, faces, smooth = [], [], []
    if shape == "sphere":
        steps = n // 2
        profile = (
            [(0, -0.5)]
            + [(0.5 * sin(pi * i / steps), -0.5 * cos(pi * i / steps)) for i in range(1, steps)]
            + [(0, 0.5)]
        )
    elif shape == "torus":
        steps = n // 2
        profile = [
            (0.34 + 0.16 * cos(tau * i / steps), 0.5 * sin(tau * i / steps))
            for i in range(steps)
        ]
    elif shape == "lathe":
        radii, heights = zip(*outline)
        bottom, top = min(heights), max(heights)
        profile = [
            (0.5 * r / max(radii), (z - bottom) / (top - bottom) - 0.5) for r, z in outline
        ]
    else:
        profile = [(0, -0.5), (0.5, -0.5)]
        if shape == "cylinder":
            profile.append((0.5, 0.5))
        profile.append((0, 0.5))
    rings = []
    for radius, z in profile:
        count = n if radius else 1
        rings.append(tuple(range(len(vertices), len(vertices) + count)))
        vertices.extend(
            (radius * cos(tau * j / n), radius * sin(tau * j / n), z) for j in range(count)
        )
    if shape in ("torus", "lathe"):
        rings.append(rings[0])
        profile.append(profile[0])
    for i, (a, b) in enumerate(zip(rings, rings[1:])):
        if len(a) == len(b) == 1:
            continue
        band = list(bridge(a, b))
        faces.extend(band)
        cap = profile[i][1] == profile[i + 1][1]
        smooth.extend([not cap] * len(band))
    return vertices, faces, smooth


def prism(outline):
    xy = np.array(outline, dtype=float)
    xy = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0) - 0.5
    n = len(xy)
    vertices = [(x, y, z) for z in (-0.5, 0.5) for x, y in xy]
    faces = list(bridge(tuple(range(n)), tuple(range(n, 2 * n))))
    # Keep a convex quadrilateral cap editable. Concave silhouettes need
    # constrained cap triangles so no face crosses the outline.
    cap = (
        [tuple(range(n))]
        if n == 4 and all(turn(xy[i - 1], xy[i], xy[(i + 1) % n]) > 0 for i in range(n))
        else triangulate(outline)
    )
    for face in cap:
        faces.extend((face[::-1], tuple(v + n for v in face)))
    return vertices, faces, [False] * len(faces)


@lru_cache(maxsize=144)
def build(part, mode):
    if mode not in MODES:
        raise ValueError("Неизвестный профиль сетки")
    if part.shape in ("box", "extrusion"):
        outline = part.outline or ((0, 0), (1, 0), (1, 1), (0, 1))
        vertices, faces, smooth = prism(outline)
    else:
        shape, outline = part.shape, part.outline
        if shape == "tube":
            hole = part.inner_diameter / part.size[0]
            shape, outline = "lathe", ((hole, 0), (1, 0), (1, 1), (hole, 1))
        vertices, faces, smooth = radial(shape, 16 if mode == "game" else 48, outline)
    if mode == "game":
        smooth = [s for face, s in zip(faces, smooth) for _ in range(len(face) - 2)]
        faces = list(triangles(faces))
    x, y, z = map(radians, part.rotation)
    rx = ((1, 0, 0), (0, cos(x), -sin(x)), (0, sin(x), cos(x)))
    ry = ((cos(y), 0, sin(y)), (0, 1, 0), (-sin(y), 0, cos(y)))
    rz = ((cos(z), -sin(z), 0), (sin(z), cos(z), 0), (0, 0, 1))
    world = (np.asarray(vertices) * part.size) @ (np.asarray(rz) @ ry @ rx).T + part.center
    world.setflags(write=False)
    return Mesh(world, tuple(faces), tuple(smooth))
