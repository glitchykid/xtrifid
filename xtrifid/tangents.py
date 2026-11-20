"""Per-corner shading frames shared by the viewport and the game exporter."""

import numpy as np

from .geometry import triangles
from .textures import uv_layout


def _indices(faces):
    if all(len(face) == 3 for face in faces):
        return np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    return np.asarray(list(triangles(faces)), dtype=np.int32).reshape(-1, 3)


def _normalized(vectors, fallback=0):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(lengths > 1e-24, vectors / np.maximum(lengths, 1e-24), fallback)


def texture_chart(mesh, part):
    """Rest-space UV values and chart corner indices in triangle order."""
    if not mesh.faces or np.ptp(mesh.vertices, axis=0).max() < 1e-20:
        return np.zeros((1, 2)), np.zeros_like(_indices(mesh.faces))
    uv, faces = uv_layout(mesh, part)
    return uv, _indices(faces)


def _fallback_tangent(normals):
    # Project a coordinate axis that cannot be parallel to the normal.
    axis = np.zeros_like(normals)
    along_x = np.abs(normals[..., 0]) < 0.8
    axis[..., 0] = along_x
    axis[..., 1] = ~along_x
    return _normalized(axis - (normals * axis).sum(axis=-1, keepdims=True) * normals)


def _tangents(points, corners, normals, smooth, area, chart):
    uv, uv_indices = chart
    coords = uv[uv_indices]
    duv1, duv2 = coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    sign = np.where(det < 0, -1.0, 1.0)
    edge1, edge2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    raw = edge1 * duv2[:, 1:] - edge2 * duv1[:, 1:]
    raw = _normalized(raw) * sign[:, None]
    raw[(np.abs(det) < 1e-18) | (area < 1e-24)] = 0
    at_corner = np.repeat(raw[:, None, :], 3, axis=1)
    if smooth.any():
        # Angle weighting is independent of triangle size. At a UV seam or a
        # mirror, accumulate only tangents belonging to the same parameterization.
        p = points[smooth]
        e1 = np.roll(p, -1, axis=1) - p
        e2 = np.roll(p, 1, axis=1) - p
        angles = np.arctan2(area[smooth, None], (e1 * e2).sum(axis=2))
        key = corners[smooth].astype(np.int64) * len(uv) + uv_indices[smooth]
        key = 2 * key + (sign[smooth, None] < 0)
        groups, inverse = np.unique(key.reshape(-1), return_inverse=True)
        summed = np.zeros((len(groups), 3))
        np.add.at(summed, inverse, (at_corner[smooth] * angles[:, :, None]).reshape(-1, 3))
        at_corner[smooth] = summed[inverse].reshape(-1, 3, 3)
    projected = at_corner - (at_corner * normals).sum(axis=2, keepdims=True) * normals
    tangent = np.empty((*corners.shape, 4), dtype=np.float32)
    tangent[..., :3] = _normalized(projected, _fallback_tangent(normals))
    tangent[..., 3] = sign[:, None]
    return coords.astype(np.float32), tangent


def _normal_data(mesh):
    corners = _indices(mesh.faces)
    points = mesh.vertices[corners]
    face = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    area = np.linalg.norm(face, axis=1)
    normal = np.repeat(face[:, None, :], 3, axis=1)
    smooth = np.repeat(mesh.smooth, [len(f) - 2 for f in mesh.faces]).astype(bool)
    if smooth.any():
        averages = np.zeros((len(mesh.vertices), 3))
        for column in range(3):
            np.add.at(averages, corners[smooth, column], face[smooth])
        normal[smooth] = averages[corners[smooth]]
    return corners, points, _normalized(normal, (0, 0, 1)).astype(np.float32), smooth, area


def corner_normals(mesh):
    """Triangle indices and their area-weighted smooth or flat corner normals."""
    corners, _, normals, _, _ = _normal_data(mesh)
    return corners, normals


def corner_frames(mesh, part, *, rest_mesh=None, chart=None):
    """Return triangles, unit normals, lower-left UVs and signed unit tangents.

    A positive normal-map green value points up the image, along
    cross(normal, tangent.xyz) * tangent.w. A deforming surface can reuse the
    rest mesh or its previously computed texture_chart while its frame moves.
    """
    if rest_mesh is not None and (
        mesh.faces != rest_mesh.faces or len(mesh.vertices) != len(rest_mesh.vertices)
    ):
        raise ValueError("Исходная и деформированная сетки должны иметь общую топологию")
    corners, points, normal, smooth, area = _normal_data(mesh)
    if part is None or part.texture is None:
        uv = np.zeros((*corners.shape, 2), dtype=np.float32)
    else:
        if chart is None:
            chart = texture_chart(mesh if rest_mesh is None else rest_mesh, part)
        if part.texture.pbr is not None:
            uv, tangent = _tangents(points, corners, normal, smooth, area, chart)
            return corners, normal, uv, tangent
        uv = chart[0][chart[1]].astype(np.float32)
    tangent = np.empty((*corners.shape, 4), dtype=np.float32)
    tangent[..., :3] = _fallback_tangent(normal)
    tangent[..., 3] = 1
    return corners, normal, uv, tangent
