"""Closed Z-up part meshes. All faces use outward counterclockwise winding."""

from dataclasses import dataclass
from functools import lru_cache
from math import cos, pi, radians, sin, tau

import numpy as np

from .model import MODES
from .polygon import triangulate, turn

RADIAL_SEGMENTS = {"game": 16, "render": 48}


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


def unit_profile(part):
    """Control points in a unit bounding box: XY for prisms, radius/Z for turned solids."""
    if part.shape in ("box", "extrusion"):
        xy = np.array(part.outline or ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=float)
        return (xy - xy.min(axis=0)) / np.ptp(xy, axis=0) - 0.5
    if part.shape == "lathe":
        radii, heights = zip(*part.outline)
        bottom, top = min(heights), max(heights)
        return [
            (0.5 * r / max(radii), (z - bottom) / (top - bottom) - 0.5) for r, z in part.outline
        ]
    if part.shape == "tube":
        bore = 0.5 * (part.inner_diameter / part.size[0])
        return [(bore, -0.5), (0.5, -0.5), (0.5, 0.5), (bore, 0.5)]
    if part.shape == "cone":
        return [(0, -0.5), (0.5, -0.5), (0, 0.5)]
    if part.shape == "cylinder":
        return [(0, -0.5), (0.5, -0.5), (0.5, 0.5), (0, 0.5)]
    raise ValueError("У этой формы нет редактируемого профиля")


def ring_indices(profile, segments):
    rings = []
    first = 0
    for radius, _ in profile:
        count = segments if radius else 1
        rings.append(range(first, first + count))
        first += count
    return rings


def radial(part, n):
    vertices, faces, smooth = [], [], []
    if part.shape == "sphere":
        steps = n // 2
        profile = (
            [(0, -0.5)]
            + [(0.5 * sin(pi * i / steps), -0.5 * cos(pi * i / steps)) for i in range(1, steps)]
            + [(0, 0.5)]
        )
    elif part.shape == "torus":
        steps = n // 2
        profile = [
            (0.34 + 0.16 * cos(tau * i / steps), 0.5 * sin(tau * i / steps))
            for i in range(steps)
        ]
    else:
        profile = unit_profile(part)
    rings = ring_indices(profile, n)
    for (radius, z), ring in zip(profile, rings):
        count = len(ring)
        vertices.extend(
            (radius * cos(tau * j / n), radius * sin(tau * j / n), z) for j in range(count)
        )
    if part.shape in ("torus", "lathe", "tube"):
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


def prism(part):
    xy = unit_profile(part)
    n = len(xy)
    vertices = [(x, y, z) for z in (-0.5, 0.5) for x, y in xy]
    faces = list(bridge(tuple(range(n)), tuple(range(n, 2 * n))))
    # Keep a convex quadrilateral cap editable. Concave silhouettes need
    # constrained cap triangles so no face crosses the outline.
    cap = (
        [tuple(range(n))]
        if n == 4 and all(turn(xy[i - 1], xy[i], xy[(i + 1) % n]) > 0 for i in range(n))
        else triangulate(part.outline)
    )
    for face in cap:
        faces.extend((face[::-1], tuple(v + n for v in face)))
    return vertices, faces, [False] * len(faces)


def rotation_matrix(rotation):
    x, y, z = map(radians, rotation)
    rx = ((1, 0, 0), (0, cos(x), -sin(x)), (0, sin(x), cos(x)))
    ry = ((cos(y), 0, sin(y)), (0, 1, 0), (-sin(y), 0, cos(y)))
    rz = ((cos(z), -sin(z), 0), (sin(z), cos(z), 0), (0, 0, 1))
    return np.asarray(rz) @ ry @ rx


@lru_cache(maxsize=144)
def build(part, mode):
    if mode not in MODES:
        raise ValueError("Неизвестный профиль сетки")
    if part.shape in ("box", "extrusion"):
        vertices, faces, smooth = prism(part)
    else:
        vertices, faces, smooth = radial(part, RADIAL_SEGMENTS[mode])
    if mode == "game":
        smooth = [s for face, s in zip(faces, smooth) for _ in range(len(face) - 2)]
        faces = list(triangles(faces))
    world = (np.asarray(vertices) * part.size) @ rotation_matrix(part.rotation).T + part.center
    world.setflags(write=False)
    return Mesh(world, tuple(faces), tuple(smooth))
