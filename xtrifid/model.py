from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING

from .organic import MAX_SECTIONS, ORGANIC_SHAPES
from .polygon import close_meridian, normalize_outline, simplify_outline, triangulate

if TYPE_CHECKING:
    from .surface import Surface
    from .textures import TextureSpec

PRIMITIVES = ("box", "cylinder", "sphere", "cone", "torus", "extrusion", "lathe", "tube")
SHAPES = PRIMITIVES + ORGANIC_SHAPES + ("surface", "sculpt", "reference")
STYLES = ("standard", "low_poly")
MODES = ("game", "render")
DETAILS = ("low", "medium", "high")
POSES = ("standing", "walking", "arms_out")
PHYSIQUES = ("neutral", "male", "female")
BODY_REGIONS = ("head", "ears", "hands", "feet", "shoulders", "waist", "nose", "lips", "chin")
MIN_BODY_SCALE, MAX_BODY_SCALE = 0.75, 1.25
MAX_PARTS = 48
MAX_OUTLINE = 32
MAX_LENGTH = 1000
MIN_SIZE = 0.001


def vector(value, length, low, high):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Ожидался вектор из {length} чисел")
    for n in value:
        if type(n) not in (int, float) or not low <= n <= high or not isfinite(n):
            raise ValueError(f"Координата вне диапазона {low} … {high}")
    return tuple(float(n) for n in value)


LOFT_COORDS = ("height", "radius_x", "radius_y", "offset_right", "offset_back")


def loft_sections(raw):
    if not isinstance(raw, (list, tuple)) or not 3 <= len(raw) <= MAX_SECTIONS:
        raise ValueError(f"Нужно от 3 до {MAX_SECTIONS} поперечных сечений")
    rows = []
    for row in raw:
        if isinstance(row, dict):
            if row.keys() != set(LOFT_COORDS):
                raise ValueError("Нужны высота, два радиуса и два смещения сечения")
            row = [row[key] for key in LOFT_COORDS]
        rows.append(vector(row, 5, -1, 1))
    if (
        rows[0][0] != 0
        or rows[-1][0] != 1
        or any(b[0] - a[0] < 1e-6 for a, b in zip(rows, rows[1:]))
    ):
        raise ValueError("Высоты сечений должны возрастать от 0 до 1")
    for i, (_, rx, ry, _, _) in enumerate(rows):
        if rx >= 1e-6 and ry >= 1e-6:
            continue
        if rx == ry == 0 and i in (0, len(rows) - 1):
            continue
        raise ValueError("Внутренним сечениям нужны два положительных радиуса")
    return tuple(rows)


@dataclass(frozen=True)
class Part:
    name: str
    shape: str
    size: tuple
    center: tuple
    rotation: tuple
    outline: tuple = ()
    inner_diameter: float | None = None
    sections: tuple = ()
    cloak_back: bool = False
    pose: str = "standing"
    body_scales: tuple = ()
    physique: str = "neutral"

    masks: tuple = ()
    nodes: tuple = ()
    surface: "Surface | None" = None
    texture: "TextureSpec | None" = None
    rounded: bool = False

    @classmethod
    def parse(cls, data, *, simplify_profiles=False):
        if not isinstance(data, dict):
            raise ValueError("Ожидалось описание детали")
        name = data.get("name")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not name.isprintable()
            or len(name) > 80
        ):
            raise ValueError("Некорректное имя детали")
        shape = data.get("shape")
        if shape not in SHAPES:
            raise ValueError("Неизвестная форма детали")
        outline = ()
        if shape in ("extrusion", "lathe"):
            raw = data.get("outline")
            if not isinstance(raw, (list, tuple)) or not 3 <= len(raw) <= MAX_OUTLINE:
                raise ValueError("Для этой формы нужен контур из 3–32 точек")
            points = []
            for point in raw:
                if simplify_profiles and shape == "lathe" and isinstance(point, dict):
                    if point.keys() != {"radius", "height"}:
                        raise ValueError("Нужны радиус и высота профиля вращения")
                    points.append(vector([point["radius"], point["height"]], 2, 0, 1))
                else:
                    points.append(vector(point, 2, -1, 1))
            points = tuple(points)
            if simplify_profiles:
                if shape == "lathe":
                    points = close_meridian(
                        points, named_surface=all(isinstance(p, dict) for p in raw)
                    )
                points = simplify_outline(points)
                if len(points) > MAX_OUTLINE:
                    raise ValueError("Профилю нужно свободное место для вершины на оси")
            outline = normalize_outline(points)
            if shape == "lathe":
                axis = [i for i, (r, _) in enumerate(outline) if r == 0]
                if any(r < 0 for r, z in outline) or (
                    axis
                    and (len(axis) != 2 or (axis[1] - axis[0]) not in (1, len(outline) - 1))
                ):
                    raise ValueError("Профиль вращения может касаться оси только одним ребром")
            else:
                triangulate(outline)
        size = vector(data.get("size"), 3, MIN_SIZE, MAX_LENGTH)
        bore = data.get("inner_diameter") if shape == "tube" else None
        cloak_back = data.get("cloak_back", False) if shape == "cloak" else False
        if type(cloak_back) is not bool:
            raise ValueError("Неверное положение пол плаща")
        if shape == "tube" and (
            type(bore) not in (int, float) or not 0.000001 <= bore < size[0]
        ):
            raise ValueError(
                "Внутренний диаметр должен быть положительным и меньше наружного размера X"
            )
        pose, scales, physique = "standing", (), "neutral"
        if shape == "human":
            pose = data.get("pose", "standing")
            if pose not in POSES:
                raise ValueError("Неизвестная поза человека")
            physique = data.get("physique", "neutral")
            if physique not in PHYSIQUES:
                raise ValueError("Неизвестный анатомический вариант фигуры")
            raw = data.get("body_scales", {})
            if not isinstance(raw, dict) or not raw.keys() <= set(BODY_REGIONS):
                raise ValueError("Неизвестная область тела")
            values = vector(
                [raw.get(region, 1) for region in BODY_REGIONS],
                len(BODY_REGIONS),
                MIN_BODY_SCALE,
                MAX_BODY_SCALE,
            )
            scales = tuple(
                (region, value) for region, value in zip(BODY_REGIONS, values) if value != 1
            )
        nodes, masks, surface, texture = (), (), None, None
        rounded = data.get("rounded", False)
        if type(rounded) is not bool or (rounded and shape != "reference"):
            raise ValueError("Округление сечений доступно для силуэтов")
        if shape in ("sculpt", "reference"):
            from .sculpt import parse_masks, parse_nodes

            if shape == "sculpt":
                nodes = parse_nodes(data.get("nodes"))
            else:
                masks = parse_masks(data.get("masks"))
        if shape == "surface":
            from .surface import Surface

            surface = Surface.parse(data.get("surface"))
        if "texture" in data:
            from .textures import TextureSpec

            texture = TextureSpec.parse(data["texture"])
        return cls(
            name.strip(),
            shape,
            size,
            vector(data.get("center"), 3, -MAX_LENGTH, MAX_LENGTH),
            vector(data.get("rotation"), 3, -360, 360),
            outline,
            bore,
            loft_sections(data.get("sections")) if shape == "loft" else (),
            cloak_back,
            pose,
            scales,
            physique,
            masks,
            nodes,
            surface,
            texture,
            rounded,
        )

    def data(self, *, include_geometry=True):
        data = {k: getattr(self, k) for k in ("name", "shape", "size", "center", "rotation")}
        if self.outline:
            data["outline"] = self.outline
        if self.inner_diameter is not None:
            data["inner_diameter"] = self.inner_diameter
        if self.sections:
            data["sections"] = self.sections
        if self.cloak_back:
            data["cloak_back"] = True
        if self.shape == "human":
            data["pose"] = self.pose
            if self.physique != "neutral":
                data["physique"] = self.physique
            if self.body_scales:
                data["body_scales"] = dict(self.body_scales)
        if self.nodes:
            data["nodes"] = self.nodes
        if self.rounded:
            data["rounded"] = True
        if include_geometry:
            if self.masks:
                data["masks"] = self.masks
            if self.surface:
                data["surface"] = self.surface.data()
            if self.texture:
                data["texture"] = self.texture.data()
        return data


@dataclass(frozen=True)
class Scene:
    parts: tuple
    detail: str = "medium"
    style: str = "standard"

    def __post_init__(self):
        if self.style not in STYLES:
            raise ValueError("Неизвестный стиль сетки")
        if self.detail not in DETAILS:
            raise ValueError("Неизвестный уровень детализации")

    def with_part(self, index, part):
        parts = list(self.parts)
        parts[index] = part
        return replace(self, parts=tuple(parts))

    @classmethod
    def parse(cls, data, *, simplify_profiles=False):
        raw = data.get("parts") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PARTS:
            raise ValueError(f"Ожидалось от 1 до {MAX_PARTS} деталей")
        parts = tuple(Part.parse(p, simplify_profiles=simplify_profiles) for p in raw)
        return cls(parts, data.get("detail", "medium"), data.get("style", "standard"))

    def data(self):
        return {
            "parts": [p.data() for p in self.parts],
            "detail": self.detail,
            "style": self.style,
        }


class Document:
    def __init__(self, scene=None, mode="game"):
        self.scene = scene
        self.mode = mode
        self.history = []

    def apply(self, scene, mode):
        if mode not in MODES:
            raise ValueError("Неизвестный профиль")
        if (scene, mode) != (self.scene, self.mode):
            self.history = (self.history + [(self.scene, self.mode)])[-20:]
            self.scene, self.mode = scene, mode

    def replace(self, index, part):
        self.apply(self.scene.with_part(index, part), self.mode)

    def undo(self):
        if self.history:
            self.scene, self.mode = self.history.pop()
