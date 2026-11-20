"""Resolve the model's typed surface and orientation parameters into editable geometry."""

from dataclasses import replace
from math import cos, hypot, isclose, pi, sin

from .geometry import rotation_matrix
from .model import MAX_LENGTH, MIN_SIZE, loft_sections, vector

LOFT_ENDS = ("rounded", "flat", "pointed")
LOFT_LEANS = {"right": (0, 1), "left": (0, -1), "back": (1, 1), "front": (1, -1)}
CAP_STEPS = 3
CYLINDER_AXES = {
    "vertical_z": (2, (0, 0, 0)),
    "width_x": (0, (0, 90, 0)),
    "depth_y": (1, (90, 0, 0)),
}


def finish_loft(part, recipe):
    """Complete the requested ends inside the part's dimensioned envelope."""
    ends = (recipe.get("bottom_end"), recipe.get("top_end"))
    if ends == (None, None):
        return part
    if any(end not in LOFT_ENDS for end in ends):
        raise ValueError("Specify both loft ends as rounded, flat or pointed")
    rows = list(part.sections)
    widths = [max(r[i] * 2 for r in rows) for i in (1, 2)]
    rises = []
    for end, row in zip(ends, (rows[0], rows[-1])):
        _, rx, ry, _, _ = row
        if not rx and end == "flat":
            raise ValueError("A flat end needs two positive section radii")
        radius = min(rx * part.size[0] / widths[0], ry * part.size[1] / widths[1])
        rises.append(min(0.45, radius / part.size[2]) if end != "flat" else 0)
    lower, upper = rises
    rows = [(lower + z * (1 - lower - upper), *tail) for z, *tail in rows]
    caps = [[], []]
    for cap, end, index, tip_z in zip(caps, ends, (0, -1), (0, 1)):
        _, tip_rx, _, tip_x, tip_y = rows[index]
        if end == "flat" or (end == "pointed" and not tip_rx):
            continue
        if not tip_rx:
            rows.pop(index)
        z, rx, ry, cx, cy = rows[index]
        if end == "rounded":
            for step in range(1, CAP_STEPS):
                angle = step * pi / (2 * CAP_STEPS)
                t, radius = sin(angle), cos(angle)
                cap.append(
                    (
                        z + t * (tip_z - z),
                        rx * radius,
                        ry * radius,
                        cx + t * (tip_x - cx),
                        cy + t * (tip_y - cy),
                    )
                )
        cap.append((tip_z, 0, 0, tip_x, tip_y))
    sections = [*reversed(caps[0]), *rows, *caps[1]]
    lean = recipe.get("top_lean", "as_drawn")
    if lean != "as_drawn":
        if not isinstance(lean, str) or lean not in LOFT_LEANS:
            raise ValueError("Unknown top displacement direction")
        axis, direction = LOFT_LEANS[lean]
        radius, center = 1 + axis, 3 + axis
        low = min(r[center] - r[radius] for r in sections)
        high = max(r[center] + r[radius] for r in sections)
        displacement = direction * (sections[-1][center] - (low + high) / 2)
        tolerance = (1e-9 if recipe.get("measured_lean") else 0.02) * (high - low)
        if displacement <= tolerance:
            sections = [list(r) for r in sections]
            for row in sections:
                if displacement < -tolerance:
                    row[center] *= -1
                else:
                    row[center] += direction * 0.18 * (high - low) * row[0]
            # Uniform axis normalization preserves the envelope and [-1,1] controls.
            scale = max(1, *(abs(r[center]) for r in sections))
            for row in sections:
                row[center] /= scale
                row[radius] /= scale
    return replace(part, sections=loft_sections(sections))


def orient_cylinder(part, recipe, diameters):
    for diameter in diameters:
        if part.size.count(diameter) == 2:
            length = next(value for value in part.size if value != diameter)
            part = replace(part, size=(diameter, diameter, length))
            break
    axis = recipe.get("axis", "custom")
    if axis == "custom":
        return part
    if not isinstance(axis, str) or axis not in CYLINDER_AXES:
        raise ValueError("Choose the world axis of the cylinder")
    dim, canonical = CYLINDER_AXES[axis]
    preferred = (
        part.rotation if abs(rotation_matrix(part.rotation)[dim, 2]) > 1 - 1e-9 else canonical
    )
    if "world_size" in recipe:
        expected = vector(recipe["world_size"], 3, MIN_SIZE, MAX_LENGTH)
        sx, sy, sz = part.size
        candidates = (preferred, part.rotation, *(r for _, r in CYLINDER_AXES.values()))
        for rotation in candidates:
            extents = [
                hypot(x * sx, y * sy) + abs(z * sz) for x, y, z in rotation_matrix(rotation)
            ]
            if all(
                isclose(a, b, rel_tol=0.001, abs_tol=1e-7) for a, b in zip(extents, expected)
            ):
                return replace(part, rotation=rotation)
    return replace(part, rotation=preferred)
