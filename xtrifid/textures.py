"""Portable material recipes, colour maps and UV charts shared by export and preview."""

import re
import struct
import zlib
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from .material_maps import (
    PBRMaterial,
    SurfaceTile,
)
from .material_maps import metallic_roughness_pixels as metallic_roughness_pixels
from .material_maps import normal_pixels as normal_pixels
from .material_maps import pbr_preset as pbr_preset

PALETTES = {
    "wood": ("#55301d", "#dba674"),
    "stone": ("#545963", "#c9c6bd"),
    "fabric": ("#414c61", "#b7c6d3"),
    "skin": ("#a76e59", "#e7bdaa"),
    "metal": ("#49535c", "#cad0d1"),
    "paint": ("#868f98", "#e4e6e5"),
}
RESOLUTIONS = (128, 256, 512, 1024)
MOTIFS = ("none", "spots", "stripes", "scales")


@dataclass(frozen=True)
class TextureSpec:
    material: str
    seed: int
    resolution: int
    color_a: str
    color_b: str
    scale: float
    motif: str = "none"
    accent: str | None = None
    pbr: PBRMaterial | None = None

    def __post_init__(self):
        if self.pbr is not None and not isinstance(self.pbr, PBRMaterial):
            raise ValueError("Некорректные параметры физического материала")
        if not isinstance(self.material, str) or self.material not in PALETTES:
            raise ValueError("Неизвестный материал текстуры")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("Семя текстуры должно быть целым от 0 до 2147483647")
        if type(self.resolution) is not int or self.resolution not in RESOLUTIONS:
            raise ValueError("Разрешение текстуры должно быть 128, 256, 512 или 1024")
        if (
            type(self.scale) not in (float, int)
            or not isfinite(self.scale)
            or not 0.25 <= self.scale <= 8
        ):
            raise ValueError("Масштаб рисунка должен быть от 0,25 до 8")
        if self.motif not in MOTIFS:
            raise ValueError("Неизвестный узор текстуры")
        for color in (self.color_a, self.color_b) + (
            () if self.accent is None else (self.accent,)
        ):
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError("Укажите цвет текстуры в формате #RRGGBB")

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or not data.keys() <= cls.__dataclass_fields__.keys():
            raise ValueError("Неверные параметры текстуры")
        material = data.get("material")
        if not isinstance(material, str) or material not in PALETTES:
            raise ValueError("Выберите материал текстуры")
        first, second = PALETTES[material]
        return cls(
            material,
            data.get("seed", 0),
            data.get("resolution", 256),
            data.get("color_a", first),
            data.get("color_b", second),
            data.get("scale", 1.0),
            data.get("motif", "none"),
            data.get("accent"),
            None if data.get("pbr") is None else PBRMaterial.parse(data["pbr"]),
        )

    def data(self):
        data = asdict(self)
        if self.pbr is None:
            del data["pbr"]
        return data


_MATERIAL_WORDS = {
    "wood": r"дерев|древес|дуб|сосн|орех|берёз|wood|oak|pine|walnut",
    "stone": r"кам[не]|каме|гранит|мрамор|сланец|бетон|stone|marble|granite|concrete",
    "fabric": r"ткан|ткач|текстил|шерст|льнян|джинс|деним|хлоп|cloth|fabric|linen|wool|denim|cotton",
    "skin": r"кож|порис|skin|leather",
    "metal": r"металл|сталь|стальн|бронз|медн|metal|steel|copper|bronze|brass",
    "paint": r"керамик|фарфор|глазур|окраш|краск|ceramic|porcelain|glaze|paint",
}
_COLORS = {
    r"син(?:яя|ий|юю|его|ей|ем|ие)|синев|blue": ("#18344e", "#719fc4"),
    r"красн|red": ("#5e1d20", "#d87063"),
    r"зел[её]н|green": ("#244a35", "#88ad78"),
    r"бел(?:ая|ый|ую|ого|ой)|white": ("#a7a39a", "#fbf9ec"),
    r"ч[её]рн|black": ("#13161a", "#5b6370"),
    r"золот|желт|жёлт|gold|yellow": ("#966224", "#edce72"),
    r"фиолет|пурпур|violet|purple": ("#42315a", "#b79ac7"),
}


def texture_from_prompt(text, *, seed=0, resolution=256):
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500:
        raise ValueError("Опишите материал в пределах 500 символов")
    text = text.casefold()
    found = next(
        (m for m, pattern in _MATERIAL_WORDS.items() if re.search(pattern, text)), None
    )
    if found is None:
        raise ValueError("Укажите материал: дерево, камень, ткань, кожа или металл")
    matches = sorted(
        (m.start(), palette)
        for pattern, palette in _COLORS.items()
        if (m := re.search(pattern, text))
    )
    colors = matches[0][1] if matches else PALETTES[found]
    accent = matches[1][1][1] if len(matches) > 1 else None
    motifs = {
        "spots": r"пятн|крап|горох|spots|speckl|polka",
        "stripes": r"полос|полоск|strip(e|ed)",
        "scales": r"чешу|scales|scallop",
    }
    motif = next((name for name, pattern in motifs.items() if re.search(pattern, text)), "none")
    specified = re.findall(r"#[0-9a-f]{6}(?![0-9a-f])", text)
    a, b = (
        specified[:2]
        if len(specified) >= 2
        else (specified[0], colors[1])
        if specified
        else colors
    )
    scale = (
        2.0
        if re.search("мелк|тонк|fine|small", text)
        else 0.65
        if re.search("крупн|coarse|large", text)
        else 1.0
    )
    return TextureSpec(
        found,
        seed,
        resolution,
        a,
        b,
        scale,
        motif,
        specified[2] if len(specified) > 2 else accent,
    )


def texture_working_bytes(parts):
    specs = {p.texture for p in parts if p.texture is not None}
    if not specs:
        return 0
    # Each resident RGB8 tile needs mip levels and driver staging. Generators
    # and exporters process one material at a time, so share one scratch budget.
    resident = sum((24 if s.pbr else 8) * s.resolution**2 for s in specs)
    scratch = max((96 if s.pbr else 64) * s.resolution**2 for s in specs)
    return resident + scratch


def pixels(spec):
    """Top-to-bottom RGB8 image. All pattern frequencies close at the tile boundary."""
    n = spec.resolution
    tile = SurfaceTile(spec)
    u, v = tile.u, tile.v
    broad, grain, phase = tile.broad, tile.grain, tile.phase
    freq, noise = tile.freq, tile.noise
    tau = 2 * np.pi
    if spec.material == "wood":
        wave = tile.tree_rings(freq(12))
        rings = ((1 + np.cos(wave)) / 2) ** 5
        values = 0.28 + 0.40 * noise(67, 5) + 0.27 * rings + 0.12 * broad
    elif spec.material == "stone":
        strata = tile.mineral_bands(freq(2), freq(3))
        veins = np.exp(-26 * np.sin(strata) ** 2)
        values = 0.13 + 0.44 * broad + 0.23 * grain + 0.31 * veins
    elif spec.material == "fabric":
        count = min(n // 6, freq(16)) * 2
        warp = np.cos(tau * count * u + phase)
        weft = np.cos(tau * count * v + phase)
        over = np.sin(np.pi * count * u + phase / 2) * np.sin(np.pi * count * v + phase / 2) > 0
        values = (
            0.48
            + 0.20 * np.where(over, warp, weft)
            + 0.12 * np.where(over, weft, warp)
            + 0.16 * (grain - 0.5)
            + 0.10 * broad
        )
    elif spec.material == "skin":
        pores = np.maximum(0, 0.68 - noise(54, 54)) ** 2
        values = 0.55 + 0.38 * (broad - 0.5) + 0.16 * (grain - 0.5) - 0.93 * pores
    elif spec.material == "paint":
        values = 0.58 + 0.18 * (broad - 0.5) + 0.07 * (grain - 0.5)
    else:
        scratches = noise(3, 86)
        values = (
            0.25 + 0.49 * scratches + 0.19 * broad + 0.08 * np.cos(tau * freq(46) * v + phase)
        )
    a, b = (
        np.frombuffer(bytes.fromhex(s[1:]), np.uint8).astype(np.float32)
        for s in (spec.color_a, spec.color_b)
    )
    rgb = a + (b - a) * np.clip(values, 0, 1)[:, :, None]
    if spec.motif != "none":
        count = 2 * min(n // 8, freq(4))
        work_u = count * u + 0.18 * (broad - 0.5) + phase / tau
        work_v = count * v + 0.16 * (broad - 0.5)
        if spec.motif == "stripes":
            mask = np.clip(
                0.5 + (np.cos(tau * (work_u + work_v)) - 0.32) * n / (8 * count), 0, 1
            )
        elif spec.motif == "spots":
            dx = ((work_u + 0.5 * (np.floor(work_v) % 2)) % 1) - 0.5
            dy = (work_v % 1) - 0.5
            mask = np.clip(
                0.5 + (0.26 + 0.08 * (broad - 0.5) - np.hypot(dx, dy)) * n / count, 0, 1
            )
        else:
            dx = 2 * (((work_u + 0.5 * (np.floor(work_v) % 2)) % 1) - 0.5)
            arc = 0.18 + 0.73 * np.sqrt(np.maximum(0, 1 - dx * dx))
            mask = np.clip(0.5 + (0.082 - abs(work_v % 1 - arc)) * n / count, 0, 1)
        ink = np.frombuffer(bytes.fromhex((spec.accent or spec.color_a)[1:]), np.uint8)
        rgb += (ink - rgb) * mask[:, :, None]
    return np.rint(rgb).astype(np.uint8)


def png_bytes(image):
    if image.dtype != np.uint8 or not (
        image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError("Для PNG нужна 8-битная одноканальная или RGB-карта")
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else 3

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    rows = np.zeros((height, width * channels + 1), np.uint8)
    rows[:, 1:] = image.reshape(height, width * channels)
    header = struct.pack(">IIBBBBB", width, height, 8, 0 if channels == 1 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def _radial_uv(mesh, part, local):
    unit = local / part.size
    radius = np.hypot(unit[:, 0], unit[:, 1])
    azimuth = np.arctan2(unit[:, 1], unit[:, 0]) / (2 * np.pi)
    # Each meridian seam duplicates UVs only. Both geometric vertex and normal
    # remain continuous. Planar end caps get their own disk-shaped charts.
    u = np.round(azimuth, 12) % 1
    torus = part.shape == "torus"
    v = (
        np.round(np.arctan2(unit[:, 2] * 0.32, radius - 0.34) / (2 * np.pi), 12) % 1
        if torus
        else unit[:, 2] + 0.5
    )
    corners = []
    for face in mesh.faces:
        ids = list(face)
        if not torus and np.ptp(unit[ids, 2]) <= 1e-10:
            chart = unit[ids, :2] + 0.5
        else:
            chart = np.column_stack((u[ids], v[ids]))
            poles = radius[ids] < 1e-10
            ring = ~poles
            if np.ptp(chart[ring, 0]) > 0.5:
                chart[:, 0] += chart[:, 0] < 0.5
            if poles.any():
                chart[poles, 0] = chart[ring, 0].mean()
            if torus and np.ptp(chart[:, 1]) > 0.5:
                chart[:, 1] += chart[:, 1] < 0.5
        corners.extend(chart)
    uv, inverse = np.unique(np.round(corners, 12), axis=0, return_inverse=True)
    indices = inverse.tolist()
    offsets = np.r_[0, np.cumsum([len(f) for f in mesh.faces])]
    return uv, [tuple(indices[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]


def uv_layout(mesh, part):
    """Per-corner box projection of a seamless colour tile, in the part's own frame."""
    # Local import avoids making the model/texture specification depend on meshing.
    from .geometry import rotation_matrix

    vertices = (mesh.vertices - part.center) @ rotation_matrix(part.rotation)
    if part.shape in ("lathe", "cone", "sphere", "torus", "cylinder", "tube"):
        return _radial_uv(mesh, part, vertices)
    # Use a common texel density even across a long, thin limb. Remove rigid-frame
    # roundoff before choosing the chart of a face equally inclined to two axes.
    vertices = (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0).max()
    faces = mesh.faces
    sizes = np.fromiter(map(len, faces), dtype=int, count=len(faces))
    offsets = np.r_[0, np.cumsum(sizes)]
    uv = np.empty((offsets[-1], 2), dtype=np.float64)
    chart_keys = np.empty(offsets[-1], dtype=np.int32)
    for size in np.unique(sizes):
        which = np.flatnonzero(sizes == size)
        ids = np.array([faces[i] for i in which])
        p = vertices[ids]
        cross = np.cross(p[:, 1:-1] - p[:, :1], p[:, 2:] - p[:, :1])
        lengths = np.linalg.norm(cross, axis=2, keepdims=True)
        normals = cross / np.maximum(lengths, 1e-30)
        score = np.abs(normals).min(axis=1)
        axis = np.round(score, 9).argmax(axis=1)
        # Orient opposing charts consistently from outside. Texture coordinates can
        # be discontinuous at seams without splitting any geometric vertex.
        group = np.arange(len(which))
        sign = cross.sum(axis=1)[group, axis] >= 0
        x = (axis + 1) % 3
        y = (axis + 2) % 3
        rows = offsets[which, None] + np.arange(size)
        chart_keys[rows] = ids * 6 + axis[:, None] * 2 + sign[:, None]
        u = p[group, :, x]
        uv[rows, 0] = np.where(sign[:, None], u, 1 - u)
        uv[rows, 1] = p[group, :, y]
        if size == 4:
            signed = normals[group, :, axis]
            folded = (signed[:, 0] * signed[:, 1] <= 0) | (score[group, axis] < 0.05)
            # Flatten highly bent quads about their triangulation diagonal. This
            # preserves both triangular areas even when their normals are orthogonal.
            for face, indices in zip(p[folded], rows[folded]):
                delta = face - face[0]
                diagonal = delta[2] / np.linalg.norm(delta[2])
                along = delta @ diagonal
                across = np.linalg.norm(np.cross(diagonal, delta), axis=1)
                across[3] *= -1
                uv[indices] = 0.5 + np.column_stack((along, across)) / (2 * np.sqrt(3))
                chart_keys[indices] = -1 - indices
    _, first, inverse = np.unique(chart_keys, return_index=True, return_inverse=True)
    corners = inverse.tolist()
    return uv[first], [tuple(corners[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]
