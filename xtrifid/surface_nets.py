"""Dual quads for cells whose boundary contains just one unambiguous contour."""

from itertools import product

import numpy as np

CORNERS = np.asarray(tuple(product((0, 1), repeat=3)))
EDGES = tuple((a, a | bit) for a in range(8) for bit in (1, 2, 4) if not a & bit)


def _connected(mask):
    """The positive (or negative) corners must form one edge-connected component."""
    remaining = {i for i in range(8) if mask & (1 << i)}
    if not remaining:
        return True
    stack = [remaining.pop()]
    while stack:
        corner = stack.pop()
        for bit in (1, 2, 4):
            other = corner ^ bit
            if other in remaining:
                remaining.remove(other)
                stack.append(other)
    return not remaining


_AMBIGUOUS = np.asarray(
    [not (_connected(mask) and _connected(255 ^ mask)) for mask in range(256)]
)


def cells(field):
    inside = field > 0
    shape = np.subtract(field.shape, 1)
    codes = np.zeros(tuple(shape), dtype=np.uint8)
    for i, corner in enumerate(CORNERS):
        slices = tuple(slice(start, start + length) for start, length in zip(corner, shape))
        codes |= inside[slices].astype(np.uint8) << i
    return codes


def unambiguous(codes):
    return not np.any(_AMBIGUOUS[codes])


def edge_count(field):
    inside = field > 0
    return sum(np.count_nonzero(np.diff(inside, axis=axis)) for axis in range(3))


def quads(field, codes):
    active = np.argwhere((codes != 0) & (codes != 255))
    if not len(active):
        return np.empty((0, 3)), []
    ids = np.full(codes.shape, -1, dtype=np.int32)
    ids[tuple(active.T)] = np.arange(len(active))
    vertices = np.zeros_like(active, dtype=float)
    counts = np.zeros((len(active), 1), dtype=np.int16)
    for a, b in EDGES:
        first, last = CORNERS[a], CORNERS[b]
        f = field[tuple((active + first).T)]
        g = field[tuple((active + last).T)]
        crossed = (f > 0) != (g > 0)
        t = (f[crossed] / (f[crossed] - g[crossed]))[:, None]
        vertices[crossed] += first + t * (last - first)
        counts[crossed] += 1
    vertices = active + vertices / counts
    blocks = []
    directions = np.eye(3, dtype=int)
    inside = field > 0
    for axis in range(3):
        crossings = np.argwhere(np.diff(inside, axis=axis))
        b, c = directions[(axis + 1) % 3], directions[(axis + 2) % 3]
        ring = np.column_stack(
            [ids[tuple((crossings - offset).T)] for offset in (b + c, c, 0, b)]
        )
        reverse = ~inside[tuple(crossings.T)]
        ring[reverse] = ring[reverse, ::-1]
        blocks.append(ring)
    return vertices, list(map(tuple, np.concatenate(blocks).tolist()))
