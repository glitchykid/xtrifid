"""Edit a single control point in the part's pre-edit, centred local coordinate system."""

from .geometry import build, radial_segments, ring_indices, rotation_matrix, unit_profile
from .model import MAX_LENGTH, MIN_SIZE, Part, vector

PROFILE_SHAPES = ("box", "extrusion", "cylinder", "cone", "tube", "lathe")


def profile_points(part):
    """Physical XY metres, or (radius along local +X, height along local Z) metres."""
    width, depth, height = part.size
    vertical_scale = depth if part.shape in ("box", "extrusion") else height
    return tuple((float(x) * width, float(y) * vertical_scale) for x, y in unit_profile(part))


def _check_index(index, points):
    if type(index) is not int or not 0 <= index < len(points):
        raise ValueError("Выберите существующую точку профиля")


def profile_guide(part, index, mode, detail="medium"):
    """World-space polyline on this mesh: a vertical edge, a closed ring, or a pole."""
    section = unit_profile(part)
    _check_index(index, section)
    vertices = build(part, mode, detail).vertices
    if part.shape in ("box", "extrusion"):
        return vertices[[index, index + len(section)]]
    ring = ring_indices(section, radial_segments(mode, detail))[index]
    if len(ring) == 1:
        return vertices[ring.start : ring.stop]
    return vertices[[*ring, ring.start]]


def move_profile_point(part, index, point):
    """Move one edge or ring, preserving the world positions of the rest of the profile."""
    points = list(profile_points(part))
    _check_index(index, points)
    # Coordinates refer to the old local centre. A valid expanded profile may
    # extend beyond MAX_LENGTH on one side before it is recentered.
    target = vector(point, 2, -2 * MAX_LENGTH, 2 * MAX_LENGTH)
    if target == points[index]:
        return part
    points[index] = target

    xs, ys = zip(*points)
    left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
    span = top - bottom
    mid = (top + bottom) / 2
    if part.shape in ("box", "extrusion"):
        shape = "extrusion"
        size = vector((right - left, span, part.size[2]), 3, MIN_SIZE, MAX_LENGTH)
        offset = ((right + left) / 2, mid, 0)
        outline = tuple(((x - left) / size[0], (y - bottom) / span) for x, y in points)
    else:
        if left < 0:
            raise ValueError("Радиус точки не может быть отрицательным")
        shape = "lathe"
        width = 2 * right
        size = vector(
            (width, part.size[1] * (width / part.size[0]), span), 3, MIN_SIZE, MAX_LENGTH
        )
        offset = (0, 0, mid)
        outline = tuple((r / right, (z - bottom) / span) for r, z in points)

    # The stored part origin is the centre of its bounds; express its shift
    # in world axes before replacing it, so untouched points stay anchored.
    center = (rotation_matrix(part.rotation) @ offset + part.center).tolist()
    result = Part.parse(
        part.data() | {"shape": shape, "size": size, "center": center, "outline": outline}
    )
    if result.outline != outline:
        raise ValueError("Перемещение меняет направление обхода профиля")
    return result
