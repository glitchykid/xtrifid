"""Bake the six shipped meshes from MakeHuman hm08 and the meshoptimizer v1.3 C ABI.

See xtrifid/assets/README.md for versions, inputs and reproduction instructions.
"""

import ctypes as ct
import hashlib
import sys
from pathlib import Path

import numpy as np

SOURCE_SHA = "8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c"
TARGETS = {8: 6000, 16: 12000, 24: 16500, 32: 21000}


def read_body(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("Unexpected MakeHuman hm08 revision")
    vertices, quads, group = [], [], ""
    for line in data.decode().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append(tuple(map(float, fields[1:4])))
        elif fields[0] == "g":
            group = fields[1]
        elif fields[0] == "f" and group == "body":
            quads.append(tuple(int(i.split("/")[0]) - 1 for i in fields[1:]))
    quads = np.asarray(quads)
    return np.asarray(vertices)[: quads.max() + 1], quads


def subdivide(v, q):
    """One Catmull–Clark step on the closed, all-quad body control cage."""
    corners = q.reshape(-1)
    half_edges = np.stack((q, np.roll(q, -1, axis=1)), axis=-1).reshape(-1, 2)
    edges, inverse = np.unique(np.sort(half_edges, axis=1), axis=0, return_inverse=True)
    rings = inverse.reshape(-1, 4)
    centers = v[q].mean(axis=1)
    midpoints = v[edges].mean(axis=1)
    edge_points = 2 * midpoints
    np.add.at(edge_points, rings, centers[:, None, :])
    valence = np.bincount(corners)[:, None]
    # Vertex mask: (mean surrounding face + 2 * mean edge midpoint + (n-3)*V) / n.
    neighbors = np.zeros_like(v)
    np.add.at(neighbors, q, centers[:, None, :])
    np.add.at(neighbors, edges, 2 * midpoints[:, None, :])
    moved = (neighbors / valence + (valence - 3) * v) / valence
    front = rings + len(v)
    center_ids = np.broadcast_to(np.arange(len(q))[:, None] + len(v) + len(edges), q.shape)
    quads = np.stack((q, front, center_ids, np.roll(front, 1, axis=1)), axis=2)
    return np.concatenate((moved, edge_points / 4, centers)), quads.reshape(-1, 4)


def simplify(library, vertices, quads, target):
    # The neutral control cage is exactly bilateral. Reducing one connected
    # half and reflecting its winding preserves the same features on both sides.
    quads = quads[(vertices[quads, 0] >= 0).all(axis=1)]
    lookup = {tuple(point): i for i, point in enumerate(vertices)}
    reflected = np.array([lookup[(-x, y, z)] for x, y, z in vertices])
    indices = np.ascontiguousarray(quads[:, (0, 1, 2, 0, 2, 3)], dtype=np.uint32)
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    result = np.empty(indices.size, dtype=np.uint32)
    # Express fine features in the error metric even where their relief is tiny
    # relative to body height. Weight vertex-normal deviations most at the face.
    tris = indices.reshape(-1, 3)
    normals = np.zeros_like(v)
    a, b, c = v[tris].transpose(1, 0, 2)
    cross = np.cross(b - a, c - a)
    for corners in tris.T:
        np.add.at(normals, corners, cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-16)
    face_weight = np.clip((v[:, 1] - 5.8) * 2, 0, 1) * np.clip((v[:, 2] + 0.4) * 2, 0, 1)
    normals *= 0.5 + 3.0 * face_weight[:, None]
    # Increase the spatial error scale above the jaw, keeping the rest-space
    # vertices and authored topology unchanged. This assigns more of a fixed
    # triangle budget to sub-millimetre facial and auricular landmarks.
    importance = np.clip((v[:, 1] - 5.5) / 0.8, 0, 1)
    metric = np.ascontiguousarray(
        v + (v - [0, 5.5, 0.5]) * (2 * importance[:, None]), dtype=np.float32
    )
    weights = np.full(3, 0.045, dtype=np.float32)
    fn = library.meshopt_simplifyWithAttributes
    fn.restype = ct.c_size_t
    pointer, size = ct.c_void_p, ct.c_size_t
    fn.argtypes = [
        pointer,
        pointer,
        size,
        pointer,
        size,
        size,
        pointer,
        size,
        pointer,
        size,
        pointer,
        size,
        ct.c_float,
        ct.c_uint,
        pointer,
    ]
    locked = np.ascontiguousarray(vertices[:, 0] == 0, dtype=np.uint8)
    for _ in range(8):
        n = fn(
            result.ctypes.data,
            indices.ctypes.data,
            indices.size,
            metric.ctypes.data,
            len(v),
            12,
            normals.ctypes.data,
            12,
            weights.ctypes.data,
            3,
            locked.ctypes.data,
            target * 3 // 2,
            1.0,
            64,
            None,
        )
        half = result[:n].reshape(-1, 3)
        reduced = np.concatenate((half, reflected[half[:, (0, 2, 1)]]))
        directed = np.stack((reduced, np.roll(reduced, -1, axis=1)), axis=-1).reshape(-1, 2)
        edges, times = np.unique(directed, return_counts=True, axis=0)
        if times.max() == 1:
            return reduced
        # A collapse can pinch two adjacent folds together in the unweighted
        # rest space. Keep the original one-ring at these specific landmarks.
        bad = edges[times > 1].ravel()
        locked[quads[np.isin(quads, bad).any(axis=1)].ravel()] = 1
    raise ValueError("Simplification produced a non-manifold edge")


def recover_quads(tris, quads):
    """Keep surviving original control quads editable in the reduced render mesh."""
    remaining = {tuple(sorted(t)): t for t in tris.tolist()}
    survived = []
    for a, b, c, d in quads.tolist():
        first, second = tuple(sorted((a, b, c))), tuple(sorted((a, c, d)))
        if first in remaining and second in remaining:
            survived.append((a, b, c, d))
            del remaining[first], remaining[second]
    return np.array(list(remaining.values())), np.array(survived)


def eyes(n):
    """Closed clay eyes in the two hm08 orbital helpers, in source decimetres."""
    # Back pole through sclera to limbus, iris and pupil. A shallow sculpted
    # pupil cup and limbal rim remain legible in untextured clay previews.
    profile = [
        (0, -1),
        (0.71, -0.71),
        (1, 0),
        (0.86, 0.51),
        (0.67, 0.74),
        (0.48, 0.88),
        (0.45, 0.906),
        (0.425, 0.891),
        (0.22, 0.944),
        (0.158, 0.964),
        (0.125, 0.929),
        (0, 0.91),
    ]
    vertices, tris, quads = [], [], []
    for side in (-1, 1):
        previous = []
        for radius, depth in profile:
            count = n if radius else 1
            ring = list(range(len(vertices), len(vertices) + count))
            for i in range(count):
                phi = 2 * np.pi * i / n
                vertices.append(
                    (
                        side * 0.30775 + 0.13815 * radius * np.cos(phi),
                        7.28415 + 0.13815 * radius * np.sin(phi),
                        1.24535 + 0.14665 * depth,
                    )
                )
            if previous:
                for i in range(n):
                    j = (i + 1) % n
                    if len(previous) == 1:
                        tris.append((previous[0], ring[j], ring[i]))
                    elif count == 1:
                        tris.append((previous[i], previous[j], ring[0]))
                    else:
                        quads.append((previous[i], previous[j], ring[j], ring[i]))
            previous = ring
    return np.array(vertices), np.array(tris), np.array(quads)


def bake(source, library, output):
    original, cage = read_body(source)
    for n, spokes in ((8, 12), (16, 16), (24, 20), (32, 24), (48, 32), (64, 48)):
        v, q = original, cage
        tris = np.empty((0, 3), dtype=int)
        if n == 64:
            v, q = subdivide(v, q)
        elif n in TARGETS:
            tris = simplify(library, v, q, TARGETS[n])
            if n == 24:
                tris, q = recover_quads(tris, q)
            else:
                q = np.empty((0, 4), dtype=int)
            used = np.unique(np.concatenate((tris.ravel(), q.ravel())))
            tris, q = np.searchsorted(used, tris), np.searchsorted(used, q)
            v = v[used]
        eyeballs, caps, rings = eyes(spokes)
        tris, q = np.vstack((tris, caps + len(v))), np.vstack((q, rings + len(v)))
        v = np.concatenate((v, eyeballs))
        # Right-handed, Z-up; the identical rest-space transform serves all LODs and rig weights.
        v = (v[:, (0, 2, 1)] - [0, 1.09965, -8.1676]) * [1, -1, 1] / 16.6589
        assert len(v) < 65536
        np.savez_compressed(
            output / f"human-{n}.npz",
            vertices=v.astype("<f4"),
            triangles=tris.astype("<u2"),
            quads=q.astype("<u2"),
        )
        print(n, len(v), len(tris), len(q), len(tris) + 2 * len(q))


if __name__ == "__main__":
    bake(Path(sys.argv[1]), ct.CDLL(sys.argv[2]), Path(sys.argv[3]))
