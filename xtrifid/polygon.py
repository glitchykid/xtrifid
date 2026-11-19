"""Validate and triangulate a simple XY outline, including concave silhouettes."""

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
