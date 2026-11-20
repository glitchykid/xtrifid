"""Bounded implicit surfaces for editable branch graphs and orthographic silhouettes."""

from itertools import permutations
from math import isfinite

import numpy as np

from . import surface_nets
from .resources import mesh_triangle_budget

MAX_NODES = 24
MAX_MASK = 96
MIN_RADIUS = 0.025
NODE_FIELDS = ("name", "parent", "x", "y", "z", "radius")
_RESOLUTIONS = {"game": (12, 18, 26), "render": (22, 32, 44)}
_DETAILS = ("low", "medium", "high")


def parse_nodes(raw):
    if not isinstance(raw, (tuple, list)) or not 2 <= len(raw) <= MAX_NODES:
        raise ValueError(f"Для объёмной формы нужно 2–{MAX_NODES} узла")
    rows = []
    names = {}
    for row in raw:
        if isinstance(row, dict) and row.keys() == set(NODE_FIELDS):
            row = tuple(row[key] for key in NODE_FIELDS)
        if not isinstance(row, (tuple, list)) or len(row) != 6:
            raise ValueError("Нужны имя, опора, три координаты и радиус узла")
        name, parent, *values = row
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 64
            or not name.isprintable()
            or name.strip().casefold() in names
            or name.strip() == "@root"
        ):
            raise ValueError("У каждого узла должно быть уникальное имя")
        name = name.strip()
        if (
            not all(type(v) in (int, float) and isfinite(v) and -1 <= v <= 1 for v in values)
            or values[3] < MIN_RADIUS
        ):
            raise ValueError(
                f"Координаты узла должны быть в пределах ±1, радиус — от {MIN_RADIUS} до 1"
            )
        names[name.casefold()] = len(rows)
        rows.append((name, parent, *map(float, values)))
    nodes = []
    for name, parent, *values in rows:
        if isinstance(parent, str):
            parent = -1 if parent == "@root" else names.get(parent.casefold())
        if type(parent) is not int or not -1 <= parent < len(rows):
            raise ValueError(f"Неизвестный родитель узла «{name}»")
        nodes.append((name, parent, *values))
    if sum(node[1] == -1 for node in nodes) != 1:
        raise ValueError("Нужен один корневой узел")
    for i in range(len(nodes)):
        chain = set()
        while i >= 0:
            if i in chain:
                raise ValueError("Узлы не могут образовывать цикл")
            chain.add(i)
            i = nodes[i][1]
    return tuple(nodes)


def parse_masks(raw):
    if not isinstance(raw, (tuple, list)) or len(raw) not in (1, 2):
        raise ValueError("Нужен один фронтальный и необязательный правый силуэт")
    result = []
    for view in raw:
        if not isinstance(view, (tuple, list)) or not 2 <= len(view) <= MAX_MASK:
            raise ValueError("Высота маски должна быть от 2 до 96 пикселей")
        if (
            not isinstance(view[0], str)
            or not 2 <= len(view[0]) <= MAX_MASK
            or any(
                not isinstance(row, str) or len(row) != len(view[0]) or set(row) - {"0", "1"}
                for row in view
            )
            or not any("1" in row for row in view)
        ):
            raise ValueError(
                "Для силуэта нужна непустая прямоугольная двоичная маска шириной 2–96 пикселей"
            )
        result.append(tuple(view))
    return tuple(result)


def _resolution(mode, detail, low_poly):
    if mode not in _RESOLUTIONS or detail not in _DETAILS:
        raise ValueError("Неизвестный профиль или детализация")
    n = _RESOLUTIONS[mode][_DETAILS.index(detail)]
    return max(10, n * 2 // 3) if low_poly else n


def _view_field(rows, coordinates):
    values = np.asarray([list(row.encode()) for row in rows], dtype=np.int16)
    values = np.pad((values - ord("0")) * 2 - 1, 1, constant_values=-1)
    h, w = len(rows), len(rows[0])
    x = np.clip((coordinates + 0.5) * w + 0.5, 0, w + 1)
    z = np.clip((0.5 - coordinates) * h + 0.5, 0, h + 1)
    i, j = x.astype(int), z.astype(int)
    ii, jj = np.minimum(i + 1, w + 1), np.minimum(j + 1, h + 1)
    dx, dz = (x - i)[:, None], (z - j)[None, :]
    return (1 - dx) * ((1 - dz) * values[j, i[:, None]] + dz * values[jj, i[:, None]]) + dx * (
        (1 - dz) * values[j, ii[:, None]] + dz * values[jj, ii[:, None]]
    )


def _section_intervals(profile, line):
    """Locate each continuous horizontal run, interpolating its two silhouette edges."""
    crossing = np.flatnonzero(np.diff(profile > 0))
    if not len(crossing):
        return None
    t = profile[crossing] / (profile[crossing] - profile[crossing + 1])
    limits = line[crossing] + t * (line[crossing + 1] - line[crossing])
    low, high = limits[::2], limits[1::2]
    centers, radii = (low + high) / 2, (high - low) / 2
    offset = line[:, None] - centers
    nearest = np.argmin(abs(offset) - radii, axis=1)
    radius = radii[nearest]
    return offset[np.arange(len(line)), nearest], radius


def _projection_profile(rows, sampled, line, width, height):
    """Squared half-thickness in metres, plus the visible span of each row."""
    span = np.zeros(len(line))
    ellipses = np.zeros_like(sampled)
    for z in range(1, len(line) - 1):
        intervals = _section_intervals(sampled[:, z], line)
        if intervals is not None:
            offset, radius = intervals
            span[z] = radius.max() * width
            ellipses[:, z] = np.maximum(0, 1 - (offset / radius) ** 2)
    if not any("0" in row.strip("0") for row in rows):
        return ellipses, span, None

    # Every inscribed disk of a silhouette is the projection of a sphere. Lift
    # the whole family together so the centre of a body does not jump when its
    # outline meets a neighbouring appendage. Both distance passes are separable.
    m = max(64, len(rows), len(rows[0]))
    fine = np.arange(-1, m + 2) / m - 0.5
    inside = _view_field(rows, fine) > 0
    separation = (fine[:, None] - fine) ** 2
    distance = np.where(inside, np.inf, 0.0)
    distance = np.min(distance[None, :, :] + separation[:, :, None] * width**2, axis=1)
    distance = np.min(distance[:, None, :] + separation[None, :, :] * height**2, axis=2)
    radii = np.maximum(0, np.sqrt(distance) - min(width, height) / (2 * m))
    offset = (line[:, None] - fine) ** 2
    envelope = np.max(radii[None, :, :] ** 2 - offset[:, :, None] * width**2, axis=1)
    envelope = np.maximum(
        0, np.max(envelope[:, None, :] - offset[None, :, :] * height**2, axis=2)
    )
    maximum = envelope.max(axis=0)
    relative = envelope / np.maximum(maximum, 1.0e-30)
    return relative, span, maximum


def _cross_section(profile, partner_span=None):
    relative, span, maximum = profile
    if maximum is None:
        if partner_span is None:
            return relative * (span / max(span)) ** 2 if np.any(span) else relative
        return relative
    if partner_span is None:
        return relative * maximum / max(maximum) if np.any(maximum) else relative
    # Scale the main body to match the second projection. Thin lateral elements
    # retain their measured depth when the body behind them is much broader.
    physical = np.minimum(1, maximum / np.maximum(partner_span**2, 1.0e-30))
    return relative * (physical + (1 - physical) * relative**2)


def _rounded_sections(part, front, side, line):
    width, depth, height = part.size
    first = _projection_profile(part.masks[0], front, line, width, height)
    if side is None:
        a = _cross_section(first)
        b = np.maximum(0, 1 - (2 * line[:, None]) ** 2)
    else:
        second = _projection_profile(part.masks[1], side, line, depth, height)
        a = _cross_section(first, second[1])
        b = _cross_section(second, first[1])
    # Two elliptical support functions give 1 - (x/rx)^2 - (y/ry)^2.
    return (a[:, None, :] + b[None, :, :] - 1) * (len(line) - 3) / 4


def _smooth_sections(field, hull):
    # Filter the continuous scalar field before extracting polygons; each pass
    # has compact support and the exterior of the two measured masks stays empty.
    original = np.clip(field, -1, 1)
    filtered = original
    for weight in (0.24, 0.18):
        for axis in range(3):
            filtered = (1 - 2 * weight) * filtered + weight * (
                np.roll(filtered, -1, axis) + np.roll(filtered, 1, axis)
            )
    filtered = np.where(hull < -0.85, np.minimum(filtered, -0.02), filtered)
    # Keep at least one interior sample along each existing projection ray.
    # This protects thin attachments when the filter spans both opposing walls.
    ridge = (original > 0) & (
        (original == original.max(axis=0, keepdims=True))
        | (original == original.max(axis=1, keepdims=True))
    )
    filtered = np.where(ridge, np.maximum(filtered, 0.5 * original), filtered)
    euler = _tetra_euler(original)
    for strength in (1.0, 0.5, 0.25, 0.125):
        candidate = original + strength * (filtered - original)
        if _tetra_euler(candidate) == euler:
            return candidate
    return original


def _field(part, n):
    line = np.arange(-1, n + 2) / n - 0.5
    if part.shape == "reference":
        epsilon = 0.002
        front = _view_field(part.masks[0], line)[:, None, :]
        if len(part.masks) == 2:
            side = _view_field(part.masks[1], line)[None, :, :]
        else:
            # A single orthographic view leaves the object depth unconstrained.
            # Use a slab at the supplied depth, preserving all front-view openings.
            side = ((0.5 - abs(line)) * n)[None, :, None]
        field = np.minimum(front, side)
        if part.rounded:
            right = side[0] if len(part.masks) == 2 else None
            field = _smooth_sections(
                np.minimum(field, _rounded_sections(part, front[:, 0, :], right, line)), field
            )
    elif part.shape == "sculpt":
        points = np.asarray([node[2:5] for node in part.nodes])
        radii = np.asarray([node[5] for node in part.nodes])
        lo = np.min(points - radii[:, None], axis=0)
        hi = np.max(points + radii[:, None], axis=0)
        epsilon = 0.002 * np.max(hi - lo) / n
        # Sample sub-cell twigs as a connected tube at coarser resolutions.
        radii = np.maximum(radii, np.max(hi - lo) * 0.85 / n)
        lo = np.min(points - radii[:, None], axis=0)
        hi = np.max(points + radii[:, None], axis=0)
        axes = [lo[i] + (line + 0.5) * (hi[i] - lo[i]) for i in range(3)]
        xyz = np.meshgrid(*axes, indexing="ij", sparse=True)
        field = np.full((len(line),) * 3, -np.inf)
        for i, node in enumerate(part.nodes):
            parent = node[1]
            if parent < 0:
                continue
            delta = points[i] - points[parent]
            offset = [a - p for a, p in zip(xyz, points[parent])]
            length2 = delta @ delta
            t = (
                np.clip(sum(d * a for d, a in zip(delta, offset)) / length2, 0, 1)
                if length2 > 1e-12
                else 1.0
            )
            distance = np.sqrt(sum((a - t * d) ** 2 for a, d in zip(offset, delta)))
            edge = radii[parent] + t * (radii[i] - radii[parent]) - distance
            field = np.maximum(field, edge)
    else:
        raise ValueError("Неизвестная объёмная форма")
    # Never cut off an exterior surface; place the isovalue strictly between samples.
    field[[0, -1], :, :] = -1
    field[:, [0, -1], :] = -1
    field[:, :, [0, -1]] = -1
    # Keep shared crossings distinguishable when the GPU converts vertices to float32.
    near = np.abs(field) < epsilon
    field[near] = np.where(field[near] > 0, epsilon, -epsilon)
    return field


# Split each Cartesian cell about the same body diagonal. Adjacent cell faces then
# have identical diagonals. The isosurface cannot acquire cracks or four-way edges.
_CORNERS = surface_nets.CORNERS
_TETS = []
for order in permutations(range(3)):
    step = np.zeros(3, dtype=int)
    corners = [0]
    for axis in order:
        step[axis] = 1
        corners.append(int(step @ [4, 2, 1]))
    _TETS.append(corners)


def _cuts(tet):
    xyz = _CORNERS[tet]
    cuts = {}
    for mask in range(1, 15):
        inside = [i for i in range(4) if mask & (1 << i)]
        outside = [i for i in range(4) if not mask & (1 << i)]
        if len(inside) == 2:
            a, b = inside
            c, d = outside
            edges = [(a, c), (a, d), (b, d), (b, c)]
        else:
            edges = [(a, b) for a in inside for b in outside]
        midpoint = np.asarray([(xyz[a] + xyz[b]) * 0.5 for a, b in edges])
        normal = np.cross(midpoint[1] - midpoint[0], midpoint[2] - midpoint[0])
        if normal @ (xyz[outside].mean(0) - xyz[inside].mean(0)) < 0:
            edges.reverse()
        cuts[mask] = [(tet[a], tet[b]) for a, b in edges]
    return cuts


_CUTS = tuple(_cuts(tet) for tet in _TETS)


def _signs(field):
    inside = field > 0
    return [
        inside[
            x : x + field.shape[0] - 1, y : y + field.shape[1] - 1, z : z + field.shape[2] - 1
        ].astype(np.uint8)
        for x, y, z in _CORNERS
    ]


def _cases(signs, tet):
    return sum(signs[corner] << i for i, corner in enumerate(tet))


def _rectangular_hull(part):
    return (
        part.shape == "reference"
        and not part.rounded
        and all("0" not in row for mask in part.masks for row in mask)
    )


def _tetra_euler(field):
    # Each unique interpolation edge contributes one vertex. For a closed triangle
    # surface E = 3F/2, giving Euler characteristic V - F/2 before mesh allocation.
    inside = field > 0
    vertices = 0
    for offset in _CORNERS[1:]:
        low = tuple(slice(0, length - step) for length, step in zip(field.shape, offset))
        high = tuple(slice(step, None) for step in offset)
        vertices += np.count_nonzero(inside[low] != inside[high])
    return vertices - _triangle_count(_signs(field)) // 2


def _prepare(part, mode, detail, low_poly):
    n = _resolution(mode, detail, low_poly)
    if (
        part.shape == "reference"
        and part.rounded
        and (mode, detail, low_poly) == ("render", "high", False)
    ):
        # A narrow attachment can be only a few pixels wide even in a 96 px mask.
        # Use the source resolution up to a fixed high-detail CPU limit.
        source_size = max(max(len(mask), len(mask[0])) for mask in part.masks)
        n = max(n, min(60, source_size))
    field = _field(part, n)
    cells = surface_nets.cells(field)
    if surface_nets.unambiguous(cells):
        return field, cells
    if part.shape == "sculpt" or part.rounded:
        # A nearly tangent branch can cross diagonally at one resolution. Try
        # neighbouring samples, retaining topology. Low Poly favours the coarser one.
        euler = _tetra_euler(field)
        offsets = (-1, -2, -3, 1, 2, 3) if low_poly else (1, 2, 3)
        if part.shape == "reference" and not low_poly:
            offsets += (4, -1, -2)
        for offset in offsets:
            count = n + offset
            if count < 10:
                continue
            finer = _field(part, count)
            cells = surface_nets.cells(finer)
            if surface_nets.unambiguous(cells):
                vertices = np.count_nonzero((cells != 0) & (cells != 255))
                if vertices - surface_nets.edge_count(finer) == euler:
                    return finer, cells
    return field, None


def triangle_bound(part, mode, detail, low_poly=False):
    """Count the proposed triangles on a bounded sign grid, before mesh allocation."""
    if _rectangular_hull(part):
        return 12
    field, cells = _prepare(part, mode, detail, low_poly)
    if cells is None:
        return isosurface_triangle_count(field)
    return 2 * surface_nets.edge_count(field)


def _validate_field(field):
    if (
        not isinstance(field, np.ndarray)
        or field.ndim != 3
        or min(field.shape) < 2
        or max(field.shape) > 72
        or field.dtype.kind not in "fiu"
        or not np.isfinite(field).all()
    ):
        raise ValueError("Нужна конечная трёхмерная решётка размером от 2 до 72 по каждой оси")
    if any(np.any(np.take(field, (0, -1), axis=axis) > 0) for axis in range(3)):
        raise ValueError("Поверхность должна целиком помещаться внутри расчётной решётки")


def _triangle_count(signs):
    count = 0
    for tet in _TETS:
        cases = _cases(signs, tet)
        frequency = np.bincount(cases.ravel(), minlength=16)
        count += sum(int(frequency[i]) * (2 if i.bit_count() == 2 else 1) for i in range(1, 15))
    return count


def isosurface_triangle_count(field):
    _validate_field(field)
    return _triangle_count(_signs(field))


def isosurface(field, *, max_triangles=None):
    """Extract a closed zero level in fractional grid indices, positive values inside."""
    _validate_field(field)
    signs = _signs(field)
    count = _triangle_count(signs)
    if count == 0:
        return np.empty((0, 3)), []
    limit = mesh_triangle_budget()
    if max_triangles is not None:
        limit = min(limit, max_triangles)
    if count > limit:
        raise ValueError(
            f"Для объёмной поверхности требуется {count} треугольников, лимит {limit}"
        )
    if np.any(field == 0):
        # Coincident lattice vertices need a consistent side of the level set.
        field = np.where(field == 0, -max(float(np.max(np.abs(field))), 1e-10) * 1e-6, field)
    stride = np.asarray([field.shape[1] * field.shape[2], field.shape[2], 1])
    offsets = _CORNERS @ stride
    blocks = []
    for tet, cuts in zip(_TETS, _CUTS):
        cases = _cases(signs, tet)
        for mask, edges in cuts.items():
            cells = np.argwhere(cases == mask) @ stride
            if not cells.size:
                continue
            keys = []
            for a, b in edges:
                low, high = sorted((offsets[a], offsets[b]))
                keys.append((cells + low) * field.size + cells + high)
            blocks.append(np.column_stack(keys))
    # Vertex identity is the lattice edge, independent of which cell produced the face.
    ids, indices = np.unique(
        np.concatenate([block.ravel() for block in blocks]), return_inverse=True
    )
    low, high = np.divmod(ids, field.size)
    a, b = field.flat[low], field.flat[high]
    t = (a / (a - b))[:, None]
    start = np.column_stack(np.unravel_index(low, field.shape))
    end = np.column_stack(np.unravel_index(high, field.shape))
    vertices = start + t * (end - start)
    faces = []
    start = 0
    for block in blocks:
        rows = indices[start : start + block.size].reshape(block.shape)
        faces.extend(map(tuple, rows.tolist()))
        start += block.size
    return vertices, faces


def _fair_sections(vertices, faces):
    # Shared indexed vertices make it possible to filter the actual surface
    # without changing face connectivity or independently shrinking its panels.
    directed = np.asarray(
        [(a, b) for face in faces for a, b in zip(face, (*face[1:], face[0]))]
    )
    source, target = directed.T
    degree = np.bincount(source, minlength=len(vertices))
    result = vertices
    for weight in (0.55, -0.45) * 4:
        average = (
            np.column_stack(
                [
                    np.bincount(source, weights=result[target, axis], minlength=len(vertices))
                    for axis in range(3)
                ]
            )
            / degree[:, None]
        )
        result = result + weight * (average - result)
    delta = result - vertices
    length = np.linalg.norm(delta, axis=1)
    # Displacement in lattice units. Very thin silhouettes retain their local features.
    return vertices + delta * np.minimum(1, 0.65 / np.maximum(length, 1.0e-12))[:, None]


def mesh(part, mode, detail, low_poly=False):
    if _rectangular_hull(part):
        faces = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]
        return _CORNERS - 0.5, faces, [False] * 6
    field, cells = _prepare(part, mode, detail, low_poly)
    if cells is None:
        vertices, faces = isosurface(field)
    else:
        count = 2 * surface_nets.edge_count(field)
        if count > mesh_triangle_budget():
            raise ValueError("Снизьте детализацию объёмной модели или освободите память")
        vertices, faces = surface_nets.quads(field, cells)
    if not faces:
        raise ValueError("На выбранном уровне детализации силуэты не образовали объём")
    if part.shape == "reference" and part.rounded:
        vertices = _fair_sections(vertices, faces)
    span = np.ptp(vertices, axis=0)
    if np.any(span < 1e-8):
        raise ValueError("Силуэт имеет нулевой объём")
    vertices = (vertices - vertices.min(axis=0)) / span - 0.5
    return vertices, faces, [not low_poly] * len(faces)
