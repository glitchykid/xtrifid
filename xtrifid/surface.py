"""A baked simulation frame, independent of the solver and the source scene."""

import base64
import binascii
from dataclasses import dataclass

import numpy as np

MAX_VERTICES = 60_000
MAX_FACES = 75_000


def _decode(encoded, limit):
    if not isinstance(encoded, str) or len(encoded) > 4 * ((limit + 2) // 3):
        raise ValueError("Превышен размер запечённой поверхности")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Повреждены данные запечённой поверхности") from exc


@dataclass(frozen=True)
class Surface:
    vertices: bytes  # local XYZ float64
    faces: bytes  # four little-endian int32 indices; -1 terminates triangles
    smooth: bytes
    triangles: int

    @classmethod
    def _checked(cls, vertices, faces, smooth):
        if (
            len(vertices) % 24
            or not 3 <= len(vertices) // 24 <= MAX_VERTICES
            or not 1 <= len(smooth) <= MAX_FACES
            or len(faces) != 16 * len(smooth)
        ):
            raise ValueError("Некорректные размеры запечённой сетки")
        xyz = np.frombuffer(vertices, dtype="<f8").reshape(-1, 3)
        indices = np.frombuffer(faces, dtype="<i4").reshape(-1, 4)
        quads = indices[:, 3] != -1
        if (
            not np.isfinite(xyz).all()
            or np.max(np.abs(xyz)) > 0.500001
            or np.min(indices[:, :3]) < 0
            or np.max(indices) >= len(xyz)
            or np.any(indices[:, 3] < -1)
            or np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0)
            or max(smooth) > 1
        ):
            raise ValueError("Некорректные вершины или грани запечённой сетки")
        for corners in (indices[:, :3], indices[quads][:, (0, 2, 3)]):
            a, b, c = (xyz[corners[:, i]] for i in range(3))
            if np.any(np.all(np.cross(b - a, c - a) == 0, axis=1)):
                raise ValueError("Запечённая поверхность содержит грань нулевой площади")
        return cls(vertices, faces, smooth, len(indices) + int(quads.sum()))

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or value.keys() != {"vertices", "faces", "smooth"}:
            raise ValueError("Отсутствуют данные запечённой сетки")
        return cls._checked(
            _decode(value["vertices"], MAX_VERTICES * 24),
            _decode(value["faces"], MAX_FACES * 16),
            _decode(value["smooth"], MAX_FACES),
        )

    def data(self):
        return {
            key: base64.b64encode(getattr(self, key)).decode("ascii")
            for key in ("vertices", "faces", "smooth")
        }

    def mesh(self):
        xyz = np.frombuffer(self.vertices, dtype="<f8").reshape(-1, 3)
        indices = np.frombuffer(self.faces, dtype="<i4").reshape(-1, 4).tolist()
        return (
            xyz,
            tuple(tuple(row[:3] if row[3] == -1 else row) for row in indices),
            tuple(map(bool, self.smooth)),
        )


def baked_part(name, mesh):
    from .model import MAX_LENGTH, MIN_SIZE, Part, vector

    if not 3 <= len(mesh.vertices) <= MAX_VERTICES or not 1 <= len(mesh.faces) <= MAX_FACES:
        raise ValueError("Кадр превышает допустимый размер запечённой сетки")
    xyz = np.asarray(mesh.vertices, dtype=float)
    if xyz.shape != (len(xyz), 3) or not np.isfinite(xyz).all():
        raise ValueError("Кадр содержит некорректные вершины")
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = vector(((lo + hi) / 2).tolist(), 3, -MAX_LENGTH, MAX_LENGTH)
    size = vector(np.maximum(hi - lo, MIN_SIZE).tolist(), 3, MIN_SIZE, MAX_LENGTH)
    local = np.asarray((xyz - center) / size, dtype="<f8")
    if len(mesh.smooth) != len(mesh.faces):
        raise ValueError("Неполные данные сглаживания")
    indices = np.full((len(mesh.faces), 4), -1, dtype="<i4")
    for i, face in enumerate(mesh.faces):
        if len(face) not in (3, 4) or any(
            not isinstance(v, (int, np.integer)) or not 0 <= v < len(xyz) for v in face
        ):
            raise ValueError("Неверная грань в кадре симуляции")
        indices[i, : len(face)] = face
    surface = Surface._checked(local.tobytes(), indices.tobytes(), bytes(mesh.smooth))
    return Part(name, "surface", size, center, (0.0, 0.0, 0.0), surface=surface)
