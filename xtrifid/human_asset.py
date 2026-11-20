"""Bounded, baked levels of the hm08 anatomical control mesh in unit stature."""

from pathlib import Path

import numpy as np

# Exact counts of the distributed levels let the resource guard run before loading a mesh.
_COUNTS = {8: 6480, 16: 12640, 24: 17300, 32: 21960, 48: 28036, 64: 108944}


def rest_triangle_count(segments):
    if segments not in _COUNTS:
        raise ValueError("Неизвестная детализация анатомической сетки")
    return _COUNTS[segments]


def rest_mesh(segments):
    rest_triangle_count(segments)
    path = Path(__file__).with_name("assets") / f"human-{segments}.npz"
    with np.load(path, allow_pickle=False) as asset:
        vertices = asset["vertices"].astype(float)
        faces = [tuple(f) for field in ("triangles", "quads") for f in asset[field].tolist()]
    return vertices, faces
