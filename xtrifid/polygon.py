"""Validate and triangulate a simple XY outline, including concave silhouettes."""

from math import isclose

EPS = 1e-9


def turn(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, c):
    return all(min(x, y) - EPS <= z <= max(x, y) + EPS for x, y, z in zip(a, b, c))


def intersects(a, b, c, d):
    ab = turn(a, b, c), turn(a, b, d)
    cd = turn(c, d, a), turn(c, d, b)
    if ab[0] * ab[1] < 0 and cd[0] * cd[1] < 0:
        return True
    return any(
        abs(v) <= EPS and on_segment(p, q, r)
        for v, p, q, r in (
            (ab[0], a, b, c),
            (ab[1], a, b, d),
            (cd[0], c, d, a),
            (cd[1], c, d, b),
        )
    )


def simplify_outline(points):
    """Drop only redundant samples on the same straight segment."""
    points = list(points)
    while len(points) > 3:
        for i, middle in enumerate(points):
            previous, following = points[i - 1], points[(i + 1) % len(points)]
            vx, vy = middle[0] - previous[0], middle[1] - previous[1]
            wx, wy = following[0] - previous[0], following[1] - previous[1]
            if (
                previous != following
                and isclose(vx * wy, vy * wx, rel_tol=1e-12)
                and on_segment(previous, following, middle)
            ):
                points.pop(i)
                break
        else:
            break
    return tuple(points)


def close_meridian(points, *, named_surface=False):
    """Close an outside meridian with radial end faces; leave returning wall paths intact."""
    radii = [r for r, _ in points]
    heights = [z for _, z in points]
    if named_surface and all(r > 0 for r in radii):
        steps = [b - a for a, b in zip(heights, heights[1:])]
        if all(s > 0 for s in steps) or all(s < 0 for s in steps):
            return (*points, (0, heights[-1]), (0, heights[0]))
    if radii.count(0) != 1 or 0 not in (radii[0], radii[-1]):
        return points
    if heights[0] == heights[-1] or not (
        heights == sorted(heights) or heights == sorted(heights, reverse=True)
    ):
        return points
    flat_z = heights[0] if radii[-1] == 0 else heights[-1]
    return (*points, (0, flat_z))


def normalize_outline(points):
    count = len(points)
    if len(set(points)) != count:
        raise ValueError("Контур содержит повторяющиеся вершины")
    for i, a in enumerate(points):
        b = points[(i + 1) % count]
        if abs(turn(points[i - 1], a, b)) <= EPS:
            raise ValueError("Соседние рёбра контура лежат на одной прямой")
        for j in range(i + 2, count):
            if (j + 1) % count != i and intersects(a, b, points[j], points[(j + 1) % count]):
                raise ValueError("Контур пересекает сам себя")
    area = sum(turn((0, 0), a, b) for a, b in zip(points, points[1:] + points[:1]))
    if abs(area) <= EPS:
        raise ValueError("Площадь контура слишком мала")
    return points if area > 0 else points[::-1]


def triangulate(points):
    remaining = list(range(len(points)))
    result = []
    while len(remaining) > 3:
        for i, b in enumerate(remaining):
            a, c = remaining[i - 1], remaining[(i + 1) % len(remaining)]
            if turn(points[a], points[b], points[c]) <= EPS:
                continue
            others = (points[k] for k in remaining if k not in (a, b, c))
            if any(
                min(
                    turn(points[a], points[b], p),
                    turn(points[b], points[c], p),
                    turn(points[c], points[a], p),
                )
                >= -EPS
                for p in others
            ):
                continue
            result.append((a, b, c))
            remaining.pop(i)
            break
        else:
            raise ValueError("Не удалось разбить контур на треугольники")
    result.append(tuple(remaining))
    return result
