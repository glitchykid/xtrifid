from dataclasses import asdict, dataclass, replace
from math import isfinite

from .organic import MAX_SECTIONS, ORGANIC_SHAPES
from .polygon import normalize_outline, triangulate

PRIMITIVES = ("box", "cylinder", "sphere", "cone", "torus", "extrusion", "lathe", "tube")
SHAPES = PRIMITIVES + ORGANIC_SHAPES
MODES = ("game", "render")
DETAILS = ("low", "medium", "high")
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


def loft_sections(raw):
    if not isinstance(raw, (list, tuple)) or not 3 <= len(raw) <= MAX_SECTIONS:
        raise ValueError(f"Нужно от 3 до {MAX_SECTIONS} поперечных сечений")
    rows = tuple(vector(r, 5, -1, 1) for r in raw)
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
    return rows


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

    @classmethod
    def parse(cls, data):
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
            outline = normalize_outline(tuple(vector(p, 2, -1, 1) for p in raw))
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
        )

    def data(self):
        data = asdict(self)
        if not self.outline:
            del data["outline"]
        if self.inner_diameter is None:
            del data["inner_diameter"]
        if not self.sections:
            del data["sections"]
        if not self.cloak_back:
            del data["cloak_back"]
        return data


@dataclass(frozen=True)
class Scene:
    parts: tuple
    detail: str = "medium"

    def __post_init__(self):
        if self.detail not in DETAILS:
            raise ValueError("Неизвестный уровень детализации")

    def with_part(self, index, part):
        parts = list(self.parts)
        parts[index] = part
        return replace(self, parts=tuple(parts))

    @classmethod
    def parse(cls, data):
        raw = data.get("parts") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PARTS:
            raise ValueError(f"Ожидалось от 1 до {MAX_PARTS} деталей")
        return cls(tuple(Part.parse(p) for p in raw), data.get("detail", "medium"))

    def data(self):
        return {"parts": [p.data() for p in self.parts], "detail": self.detail}


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
