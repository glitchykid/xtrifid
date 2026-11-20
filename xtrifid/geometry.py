"""Closed Z-up part meshes. All faces use outward counterclockwise winding."""

from dataclasses import dataclass
from functools import lru_cache
from math import cos, pi, radians, sin, tau

import numpy as np

from .model import DETAILS, MODES
from .organic import organic_mesh, organic_triangle_bound
from .polygon import triangulate, turn
from .resources import mesh_triangle_budget
from .rings import bridge

_SEGMENTS = {"game": (8, 16, 32), "render": (24, 48, 64)}


def radial_segments(mode, detail="medium"):
    if mode not in MODES or detail not in DETAILS:
        raise ValueError("Неизвестный профиль или уровень детализации")
    return _SEGMENTS[mode][DETAILS.index(detail)]


def triangle_bound(part, mode, detail):
    """Estimate the triangulated working set without allocating the mesh."""
    n = radial_segments(mode, detail)
    if part.shape == "box":
        return 12
    if part.shape == "extrusion":
        return 4 * len(part.outline) - 4
    if part.shape == "sphere":
        return n * (n - 2)
    if part.shape == "torus":
        return n * n
    if part.shape == "tube":
        return 8 * n
    if part.shape == "cylinder":
        return 4 * n
    if part.shape == "cone":
        return 2 * n
    if part.shape == "lathe":
        return 2 * n * sum(r > 0 for r, _ in part.outline)
    return organic_triangle_bound(part, n)


class MeshBudgetError(ValueError):
    pass


def check_scene_budget(scene, mode):
    budget = mesh_triangle_budget()
    if budget <= 0:
        raise MeshBudgetError("Недостаточно свободной памяти для построения сетки.")
    if scene is not None:
        needed = sum(triangle_bound(p, mode, scene.detail) for p in scene.parts)
        if needed > budget:
            raise MeshBudgetError(
                f"Для сетки требуется до {needed:_} треугольников, текущий лимит {budget:_}. "
                "Снизьте детализацию, выберите Game Ready или освободите память.".replace(
                    "_", " "
                )
            )


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: tuple
    smooth: tuple


def triangles(faces):
    for face in faces:
        for i in range(1, len(face) - 1):
            yield (face[0], face[i], face[i + 1])


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


def radial_profile(part, n):
    if part.shape == "sphere":
        steps = n // 2
        return (
            [(0, -0.5)]
            + [(0.5 * sin(pi * i / steps), -0.5 * cos(pi * i / steps)) for i in range(1, steps)]
            + [(0, 0.5)]
        )
    if part.shape == "torus":
        steps = n // 2
        return [
            (0.34 + 0.16 * cos(tau * i / steps), 0.5 * sin(tau * i / steps))
            for i in range(steps)
        ]
    return unit_profile(part)


def radial(part, n):
    vertices, faces, smooth = [], [], []
    profile = radial_profile(part, n)
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


def height_bounds(part, detail):
    """Evaluate only vertical extrema for support placement, without tessellating a scene."""
    axis = rotation_matrix(part.rotation)[2] * part.size
    if part.shape in ("box", "extrusion"):
        projected = unit_profile(part) @ axis[:2]
        half = abs(axis[2]) / 2
        lo, hi = projected.min() - half, projected.max() + half
    elif part.shape == "loft":
        # Every supported segment count includes all four extrema of an ellipse.
        # Project the normalized rings directly without building faces or a scene.
        z, rx, ry, cx, cy = np.asarray(part.sections).T
        xmin, xmax = (cx - rx).min(), (cx + rx).max()
        ymin, ymax = (cy - ry).min(), (cy + ry).max()
        xscale = axis[0] / (xmax - xmin)
        yscale = axis[1] / (ymax - ymin)
        centers = (cx - (xmax + xmin) / 2) * xscale + (cy - (ymax + ymin) / 2) * yscale
        centers += (z - 0.5) * axis[2]
        n = radial_segments("render", detail)
        angles = np.arange(n) * (tau / n)
        heights = centers[:, None] + np.outer(rx * xscale, np.sin(angles))
        heights -= np.outer(ry * yscale, np.cos(angles))
        lo, hi = heights.min(), heights.max()
    else:
        n = radial_segments("render", detail)
        angles = np.arange(n) * (tau / n)
        xy = axis[0] * np.cos(angles) + axis[1] * np.sin(angles)
        radius, z = np.asarray(radial_profile(part, n)).T
        lo = (radius * xy.min() + z * axis[2]).min()
        hi = (radius * xy.max() + z * axis[2]).max()
    return float(lo + part.center[2]), float(hi + part.center[2])


def build(part, mode, detail="medium"):
    if triangle_bound(part, mode, detail) >= 4096:
        return _build(part, mode, detail)
    return _small_mesh(part, mode, detail)


def _build(part, mode, detail):
    segments = radial_segments(mode, detail)
    if part.shape in ("box", "extrusion"):
        vertices, faces, smooth = prism(part)
    elif part.shape in ("sphere", "torus", "tube", "lathe", "cone", "cylinder"):
        vertices, faces, smooth = radial(part, segments)
    else:
        vertices, faces, smooth = organic_mesh(part, segments)
    if mode == "game":
        smooth = [s for face, s in zip(faces, smooth) for _ in range(len(face) - 2)]
        faces = list(triangles(faces))
    world = (np.asarray(vertices) * part.size) @ rotation_matrix(part.rotation).T + part.center
    world.setflags(write=False)
    return Mesh(world, tuple(faces), tuple(smooth))


# Keep repeated controls responsive without retaining dense meshes across edits.
_small_mesh = lru_cache(maxsize=48)(_build)
