"""A baked simulation frame, independent of the solver and the source scene."""

import base64
import binascii
import struct
from dataclasses import dataclass, replace

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


def _polygons(packed):
    for a, b, c, d in struct.iter_unpack("<4i", packed):
        yield (a, b, c) if d == -1 else (a, b, c, d)


@dataclass(frozen=True)
class SurfaceUV:
    coords: bytes  # little-endian float32 UV pairs; keep coincident chart seams distinct
    faces: bytes  # same face order and triangle sentinel as the surface mesh

    @classmethod
    def checked(cls, coords, faces, geometry):
        quads = geometry[:, 3] != -1
        count = 3 * len(geometry) + int(quads.sum())
        if (
            len(coords) % 8
            or not 1 <= len(coords) // 8 <= count
            or len(faces) != geometry.nbytes
        ):
            raise ValueError("Некорректный размер UV-развёртки")
        points = np.frombuffer(coords, "<f4")
        ids = np.frombuffer(faces, "<i4").reshape(-1, 4)
        # Generated charts fit a tile and may extend across its periodic seam.
        if (
            not np.isfinite(points).all()
            or np.max(np.abs(points)) > 2.000001
            or (ids[:, :3] < 0).any()
            or (ids[quads, 3] < 0).any()
            or (ids[~quads, 3] != -1).any()
            or ids.max() >= len(coords) // 8
        ):
            raise ValueError("Некорректные координаты или углы UV-развёртки")
        return cls(coords, faces)

    @classmethod
    def from_layout(cls, chart, geometry):
        coords, faces = chart
        if len(faces) != len(geometry) or not 1 <= len(coords) <= 4 * len(geometry):
            raise ValueError("UV-развёртка не соответствует исходной сетке")
        points = np.asarray(coords)
        if points.shape != (len(coords), 2) or points.dtype.kind not in "ifu":
            raise ValueError("Неверная таблица UV-координат")
        ids = np.full_like(geometry, -1)
        for index, face in enumerate(faces):
            size = 3 if geometry[index, 3] == -1 else 4
            if len(face) != size or any(
                (type(i) is not int and not isinstance(i, np.integer))
                or not 0 <= i < len(coords)
                for i in face
            ):
                raise ValueError("UV-углы не соответствуют грани исходной сетки")
            ids[index, :size] = face
        return cls.checked(points.astype("<f4", copy=False).tobytes(), ids.tobytes(), geometry)


@dataclass(frozen=True)
class Surface:
    vertices: bytes  # local XYZ float64
    faces: bytes  # four little-endian int32 indices; -1 terminates triangles
    smooth: bytes
    triangles: int
    uv: SurfaceUV | None = None

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
        fields = {"vertices", "faces", "smooth"}
        if not isinstance(value, dict) or value.keys() not in (fields, fields | {"uv"}):
            raise ValueError("Отсутствуют данные запечённой сетки")
        surface = cls._checked(
            _decode(value["vertices"], MAX_VERTICES * 24),
            _decode(value["faces"], MAX_FACES * 16),
            _decode(value["smooth"], MAX_FACES),
        )
        if "uv" in value:
            chart = value["uv"]
            if not isinstance(chart, dict) or chart.keys() != {"coords", "faces"}:
                raise ValueError("Некорректная запечённая UV-развёртка")
            geometry = np.frombuffer(surface.faces, "<i4").reshape(-1, 4)
            corners = 3 * len(geometry) + int((geometry[:, 3] != -1).sum())
            surface = replace(
                surface,
                uv=SurfaceUV.checked(
                    _decode(chart["coords"], 8 * corners),
                    _decode(chart["faces"], geometry.nbytes),
                    geometry,
                ),
            )
        return surface

    def data(self):
        data = {
            key: base64.b64encode(getattr(self, key)).decode("ascii")
            for key in ("vertices", "faces", "smooth")
        }
        if self.uv is not None:
            data["uv"] = {
                key: base64.b64encode(getattr(self.uv, key)).decode("ascii")
                for key in ("coords", "faces")
            }
        return data

    def mesh(self):
        xyz = np.frombuffer(self.vertices, dtype="<f8").reshape(-1, 3)
        return xyz, tuple(_polygons(self.faces)), tuple(map(bool, self.smooth))

    def uv_layout(self, mesh_faces):
        from .geometry import triangles

        geometry, uv_faces = _polygons(self.faces), _polygons(self.uv.faces)
        if len(mesh_faces) != len(self.smooth):
            if len(mesh_faces) != self.triangles:
                raise ValueError("Топология запечённой UV-развёртки изменилась")
            geometry, uv_faces = triangles(geometry), triangles(uv_faces)
        if any(tuple(face) != expected for face, expected in zip(mesh_faces, geometry)):
            raise ValueError("Порядок граней запечённой UV-развёртки изменился")
        return np.frombuffer(self.uv.coords, "<f4").reshape(-1, 2), tuple(uv_faces)


def baked_part(name, mesh, *, uv_chart=None):
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
    if uv_chart is not None:
        surface = replace(surface, uv=SurfaceUV.from_layout(uv_chart, indices))
    return Part(name, "surface", size, center, (0.0, 0.0, 0.0), surface=surface)
