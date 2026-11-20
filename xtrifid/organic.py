"""Anatomical cross-sections and closed garment meshes. The figure faces −Y."""

from math import cos, exp, pi, sin, tau

import numpy as np

from .human_asset import rest_triangle_count
from .human_body import posed_mesh
from .rings import bridge

ORGANIC_SHAPES = ("loft", "head", "cloak", "hood", "human")
MAX_SECTIONS = 24

# Height, half-width, half-depth, lateral offset and sagittal offset.
# Landmarks run from chin to occiput and crown.
HEAD = (
    (0.00, 0.00, 0.00, 0, -0.19),
    (0.04, 0.19, 0.23, 0, -0.18),
    (0.11, 0.30, 0.34, 0, -0.10),
    (0.22, 0.38, 0.39, 0, -0.04),
    (0.27, 0.39, 0.40, 0, -0.01),
    (0.30, 0.40, 0.41, 0, 0.00),
    (0.34, 0.42, 0.42, 0, 0.01),
    (0.42, 0.45, 0.44, 0, 0.02),
    (0.46, 0.47, 0.45, 0, 0.02),
    (0.51, 0.477, 0.455, 0, 0.02),
    (0.54, 0.48, 0.46, 0, 0.02),
    (0.555, 0.482, 0.462, 0, 0.02),
    (0.60, 0.49, 0.47, 0, 0.02),
    (0.65, 0.49, 0.48, 0, 0.02),
    (0.75, 0.48, 0.49, 0, 0.03),
    (0.85, 0.43, 0.45, 0, 0.04),
    (0.93, 0.31, 0.34, 0, 0.05),
    (0.98, 0.15, 0.17, 0, 0.06),
    (1.00, 0.00, 0.00, 0, 0.06),
)


def _face_depth(angle, z):
    # Displace the front of the same mesh at the brows, eye sockets, nose and lips.
    angle = min(angle, tau - angle)

    def feature(a, h, wa, wh):
        return exp(-(((angle - a) / wa) ** 2) - ((z - h) / wh) ** 2)

    return (
        0.20 * feature(0, 0.435, 0.21, 0.065)
        + 0.14 * feature(0, 0.52, 0.14, 0.14)
        + 0.065 * feature(0.37, 0.612, 0.29, 0.045)
        - 0.072 * feature(0.38, 0.539, 0.18, 0.046)
        + 0.068 * feature(0.40, 0.54, 0.16, 0.023)
        + 0.035 * feature(0.65, 0.47, 0.22, 0.07)
        + 0.039 * feature(0.15, 0.425, 0.11, 0.038)
        + 0.065 * feature(0, 0.332, 0.29, 0.028)
        + 0.053 * feature(0, 0.263, 0.23, 0.030)
        - 0.029 * feature(0, 0.296, 0.31, 0.021)
    )


def _loft(rows, n, face=False):
    vertices, rings = [], []
    for z, rx, ry, cx, cy in rows:
        first = len(vertices)
        if not rx:
            vertices.append((cx, cy, z))
        else:
            for i in range(n):
                angle = tau * i / n
                depth = _face_depth(angle, z) if face else 0
                vertices.append((cx + rx * sin(angle), cy - ry * cos(angle) - depth, z))
        rings.append(range(first, len(vertices)))
    faces = [f for lower, upper in zip(rings, rings[1:]) for f in bridge(lower, upper)]
    smooth = [True] * len(faces)
    for index, ring in ((0, rings[0]), (-1, rings[-1])):
        if len(ring) > 1:
            z, _, _, x, y = rows[index]
            center = (len(vertices),)
            vertices.append((x, y, z))
            caps = list(bridge(center, ring) if index == 0 else bridge(ring, center))
            faces.extend(caps)
            smooth.extend([False] * len(caps))
    return vertices, faces, smooth


def _inner_face(face):
    # Keep the first corner: both skins must triangulate along the same diagonal
    # or their non-planar fabric panels can intersect in a coarse mesh.
    return (face[0], *face[:0:-1])


# Hem to neck. The elliptical radii and rearward sweep are fractions of stature;
# the last two coordinates cut the front opening for relaxed and raised arms.
CLOAK = (
    (0.00, 0.253, 0.210, 0.058, 0.98, 1.18),
    (0.10, 0.245, 0.201, 0.052, 0.94, 1.22),
    (0.29, 0.234, 0.179, 0.046, 0.91, 1.30),
    (0.50, 0.220, 0.165, 0.034, 0.86, 1.46),
    (0.70, 0.204, 0.150, 0.021, 0.76, 1.86),
    (0.84, 0.194, 0.137, 0.012, 0.61, 2.10),
    (0.90, 0.185, 0.122, 0.006, 0.43, 2.10),
    (0.96, 0.143, 0.095, 0.004, 0.20, 2.05),
    (1.00, 0.057, 0.055, 0.002, 0.085, 0.085),
)


def _cloak(n, size, thrown_back):
    spans = n + 8
    outer, inner = [], []
    # Preserve the shoulder-to-neck distance as the hem length changes.
    yoke = min(0.45, 0.18 * size[0] / size[2])
    for z, rx, ry, cy, front_gap, back_gap in CLOAK:
        gap = back_gap if thrown_back else front_gap
        z = (z / 0.84) * (1 - yoke) if z <= 0.84 else 1 - yoke * (1 - z) / 0.16
        for j in range(spans + 1):
            u = j / spans
            angle = gap + (tau - 2 * gap) * u
            # The ripples grow from the shoulders. A small change in phase with height
            # keeps the ridges from running along uniformly straight generators.
            ripple = cos(7 * tau * u + 0.58 * z * sin(tau * u)) + 0.28 * cos(
                4 * tau * u - 0.4 * z
            )
            fold = 1 + (0.016 + 0.096 * (1 - z)) * ripple
            hem = (1 - z) ** 3 * (0.036 * cos(angle) + 0.010 * ripple)
            for points, scale in ((outer, fold), (inner, fold - 0.027)):
                points.append((rx * scale * sin(angle), cy - ry * scale * cos(angle), z + hem))
    count = len(outer)
    stride = spans + 1
    quads = []
    for row in range(len(CLOAK) - 1):
        for j in range(spans):
            a = row * stride + j
            quads.append((a, a + 1, a + stride + 1, a + stride))
    faces = quads + [tuple(i + count for i in _inner_face(f)) for f in quads]
    smooth = [True] * len(faces)
    top = count - stride
    perimeter = (
        list(range(stride))
        + list(range(2 * stride - 1, count, stride))
        + list(range(count - 2, top - 1, -1))
        + list(range(top - stride, 0, -stride))
    )
    for a, b in zip(perimeter, (*perimeter[1:], perimeter[0])):
        faces.append((b, a, a + count, b + count))
        smooth.append(False)
    return outer + inner, faces, smooth


def _hood(n):
    # A shell running from a face-sized open rim to a rounded, slightly peaked back.
    bands = max(4, n // 6)
    vertices, faces, smooth = [], [], []
    rims = []
    for inside in (False, True):
        rings = []
        for row in range(bands + 1):
            t = row / bands
            first = len(vertices)
            for i in range(1 if row == bands else n):
                a = tau * i / n
                crown = -sin(a)
                radius = cos(t * pi / 2)
                chin = max(-crown, 0)
                # Sides fall inward to the neck. A peak above the forehead and
                # loose fabric behind it give this shell a different fit from a helmet.
                fold = (0.025 + 0.045 * chin) * sin(6 * a + 1.2 * t) * (1 - crown * crown)
                x = radius * cos(a) * (0.46 - 0.035 * crown + fold) * (1 - 0.39 * chin**1.5)
                rim = -0.41 + 0.29 * chin**2 + 0.12 * (1 - crown * crown)
                y = rim + (0.53 - rim) * sin(t * pi / 2) + radius * 0.03 * fold
                z = radius * (0.48 * crown + 0.056 * max(crown, 0) ** 8) + 0.16 * t
                if inside:
                    x, y, z = 0.925 * x, 0.94 * y - 0.023, 0.93 * z
                vertices.append((x, y, z))
            rings.append(range(first, len(vertices)))
        for lower, upper in zip(rings, rings[1:]):
            strip = list(bridge(lower, upper))
            faces.extend([_inner_face(f) for f in strip] if inside else strip)
            smooth.extend([True] * len(strip))
        rims.append(rings[0])
    rims_faces = list(bridge(rims[1], rims[0]))
    faces.extend(rims_faces)
    smooth.extend([False] * len(rims_faces))
    return vertices, faces, smooth


def organic_triangle_bound(part, segments):
    if part.shape == "human":
        return rest_triangle_count(segments)
    if part.shape == "cloak":
        spans = segments + 8
        return 4 * (len(CLOAK) * spans + len(CLOAK) - 1)
    if part.shape == "hood":
        return 4 * max(4, segments // 6) * segments
    rows = HEAD if part.shape == "head" else part.sections
    return 2 * len(rows) * segments


def organic_mesh(part, segments):
    if part.shape == "human":
        return posed_mesh(part, segments)
    if part.shape == "cloak":
        points, faces, smooth = _cloak(segments, part.size, part.cloak_back)
    elif part.shape == "hood":
        points, faces, smooth = _hood(segments)
    elif part.shape == "head":
        points, faces, smooth = _loft(HEAD, segments, face=True)
    elif part.shape == "loft":
        points, faces, smooth = _loft(part.sections, segments)
    else:
        raise ValueError("Нет органической формы такого типа")
    points = np.asarray(points)
    normalized = (points - points.min(axis=0)) / np.ptp(points, axis=0) - 0.5
    return normalized, faces, smooth
