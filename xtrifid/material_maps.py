"""Periodic surface relief and linear metallic-roughness maps for a material tile."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PBRMaterial:
    roughness: float
    metallic: float
    normal_strength: float

    def __post_init__(self):
        for value, limit, label in (
            (self.roughness, 1, "Шероховатость"),
            (self.metallic, 1, "Металличность"),
            (self.normal_strength, 2, "Сила рельефа"),
        ):
            if type(value) not in (float, int) or not 0 <= value <= limit:
                raise ValueError(f"{label} должна быть числом от 0 до {limit}")

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or data.keys() != cls.__dataclass_fields__.keys():
            raise ValueError("Укажите шероховатость, металличность и силу рельефа материала")
        return cls(**data)


_PRESETS = {
    "wood": PBRMaterial(0.53, 0, 0.85),
    "stone": PBRMaterial(0.76, 0, 1.1),
    "fabric": PBRMaterial(0.88, 0, 1.0),
    "skin": PBRMaterial(0.48, 0, 0.8),
    "metal": PBRMaterial(0.30, 1, 0.72),
    "paint": PBRMaterial(0.35, 0, 0.55),
}


def pbr_preset(material):
    if not isinstance(material, str) or material not in _PRESETS:
        raise ValueError("Неизвестный материал поверхности")
    return _PRESETS[material]


class SurfaceTile:
    """Sample the same seeded substrate below the colour and physical maps."""

    def __init__(self, spec):
        self.spec = spec
        self.rng = np.random.Generator(np.random.PCG64(spec.seed))
        axis = np.arange(spec.resolution, dtype=np.float32) / spec.resolution
        self.u, self.v = axis[None, :], axis[:, None]
        self.broad = self.noise(5, 5)
        self.grain = self.noise(47, 47)
        self.phase = self.rng.random() * np.pi * 2

    def freq(self, count):
        return max(1, min(self.spec.resolution // 2, round(count * self.spec.scale)))

    def noise(self, across, down):
        cols, rows = self.freq(across), self.freq(down)
        table = self.rng.random((rows, cols), dtype=np.float32)
        u, v = self.u * cols, self.v * rows
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        a, b = u - i, v - j
        a = (a * a * (3 - 2 * a)).astype(np.float32)
        b = (b * b * (3 - 2 * b)).astype(np.float32)
        i2, j2 = (i + 1) % cols, (j + 1) % rows
        i, j = i % cols, j % rows
        return (1 - b) * ((1 - a) * table[j, i] + a * table[j, i2]) + b * (
            (1 - a) * table[j2, i] + a * table[j2, i2]
        )

    def tree_rings(self, count):
        return (
            2 * np.pi * (count * self.u + 0.84 * self.broad + 0.22 * self.noise(16, 3))
            + self.phase
        )

    def mineral_bands(self, across, down):
        return (
            2
            * np.pi
            * (across * self.u + down * self.v + 1.2 * self.broad + 0.28 * self.noise(19, 17))
            + self.phase
        )


def _surface_relief(spec):
    """Return a dimensionless profile and its physical depth in units of one UV repeat."""
    tile = SurfaceTile(spec)
    n = spec.resolution
    grain = tile.grain - 0.5
    broad = tile.broad - 0.5
    if spec.material == "wood":
        count = min(tile.freq(12), n // 8)
        rings = (0.5 + 0.5 * np.cos(tile.tree_rings(count))) ** 3
        vessels = tile.noise(67, 5)
        profile = 0.72 * rings + 0.87 * (vessels - 0.5) + 0.16 * broad
        depth = 0.011 * 12 / count
    elif spec.material == "stone":
        across, down = min(tile.freq(2), n // 10), min(tile.freq(3), n // 10)
        strata = tile.mineral_bands(across, down)
        veins = np.exp(-6 * np.sin(strata) ** 2)
        profile = broad + 0.72 * grain + 0.24 * veins + 0.3 * (tile.noise(21, 28) - 0.5)
        depth = 0.014 * 3 / down
    elif spec.material == "fabric":
        # Adjacent crossings alternate which yarn passes over the other. The
        # smooth maximum rounds each exposed thread where warp and weft meet.
        count = 2 * min(n // 8, tile.freq(16))
        warp_phase, weft_phase = (
            np.pi * count * axis + tile.phase / 2 for axis in (tile.u, tile.v)
        )
        crossing = np.cos(warp_phase) * np.cos(weft_phase)
        warp = (0.5 + 0.5 * np.cos(2 * warp_phase)) * (0.64 + 0.36 * crossing)
        weft = (0.5 + 0.5 * np.cos(2 * weft_phase)) * (0.64 - 0.36 * crossing)
        profile = 0.5 * (warp + weft + np.sqrt((warp - weft) ** 2 + 0.015)) + 0.12 * grain
        depth = 0.009 * 32 / count
    elif spec.material == "skin":
        pores = np.maximum(0, 0.68 - tile.noise(54, 54)) ** 2
        profile = 0.33 * broad + 0.22 * grain - 1.6 * pores
        depth = 0.015 * 54 / tile.freq(54)
    elif spec.material == "metal":
        scratches = tile.noise(3, 86)
        count = min(tile.freq(46), n // 5)
        tool_marks = np.cos(2 * np.pi * count * tile.v + tile.phase)
        profile = 0.86 * (scratches - 0.5) + 0.14 * tool_marks + 0.12 * broad
        depth = 0.005 * 46 / count
    else:
        # Shallow isotropic orange peel in a sprayed coating.
        profile = 0.85 * grain + 0.15 * broad
        depth = 0.0035 * 47 / tile.freq(47)
    # Repeating binomial footprint. Reject the Nyquist frequency in both axes
    # before differentiating pores, fine veins and thread crests.
    for axis in (0, 1):
        profile = (np.roll(profile, 1, axis) + 2 * profile + np.roll(profile, -1, axis)) * 0.25
    return profile, depth


def normals_from_height(height, strength):
    """Bake a periodic height field in UV units to a +Y tangent normal, PNG row zero at top."""
    dx = (np.roll(height, 1, axis=1) - np.roll(height, -1, axis=1)) * (
        height.shape[1] * 0.5 * strength
    )
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * (
        height.shape[0] * 0.5 * strength
    )
    inverse_length = dx * dx + dy * dy + 1
    np.sqrt(inverse_length, out=inverse_length)
    np.reciprocal(inverse_length, out=inverse_length)
    dx *= inverse_length
    dy *= inverse_length
    rgb = np.empty((*height.shape, 3), np.uint8)
    for index, component in enumerate((dx, dy, inverse_length)):
        component += 1
        component *= 127.5
        np.rint(component, out=component)
        rgb[:, :, index] = component
    return rgb


def normal_pixels(spec):
    if spec.pbr is None:
        raise ValueError("Для карты нормалей назначьте PBR-материал")
    if spec.pbr.normal_strength == 0:
        return np.full((spec.resolution, spec.resolution, 3), (128, 128, 255), dtype=np.uint8)
    relief, depth = _surface_relief(spec)
    relief *= depth
    return normals_from_height(relief, spec.pbr.normal_strength)


def metallic_roughness_pixels(spec):
    """glTF linear channels: unused red at one, final roughness in green, metal in blue."""
    if spec.pbr is None:
        raise ValueError("Для карты шероховатости и металличности назначьте PBR-материал")
    roughness = spec.pbr.roughness
    packed = np.empty((spec.resolution, spec.resolution, 3), np.uint8)
    packed[:, :, 0] = 255
    packed[:, :, 2] = round(255 * spec.pbr.metallic)
    if roughness in (0, 1):
        packed[:, :, 1] = round(255 * roughness)
    else:
        # Exposed crests are smoother. Keep the requested mean finish and reduce
        # variation continuously as the user approaches a perfectly rough or polished surface.
        profile, _ = _surface_relief(spec)
        profile -= profile.mean(dtype=np.float64)
        profile *= -0.72 * min(roughness, 1 - roughness)
        profile += roughness
        np.clip(profile, 0, 1, out=profile)
        packed[:, :, 1] = np.rint(255 * profile)
    return packed
