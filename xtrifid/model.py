from dataclasses import asdict, dataclass, replace
from math import isfinite

from .polygon import normalize_outline, triangulate

SHAPES = ("box", "cylinder", "sphere", "cone", "torus", "extrusion")
MODES = ("game", "render")
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


@dataclass(frozen=True)
class Part:
    name: str
    shape: str
    size: tuple
    center: tuple
    rotation: tuple
    outline: tuple = ()

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
        if shape == "extrusion":
            raw = data.get("outline")
            if not isinstance(raw, (list, tuple)) or not 3 <= len(raw) <= MAX_OUTLINE:
                raise ValueError("Для выдавливания нужен контур из 3–32 точек")
            outline = normalize_outline(tuple(vector(p, 2, -1, 1) for p in raw))
            triangulate(outline)
        return cls(
            name.strip(),
            shape,
            vector(data.get("size"), 3, MIN_SIZE, MAX_LENGTH),
            vector(data.get("center"), 3, -MAX_LENGTH, MAX_LENGTH),
            vector(data.get("rotation"), 3, -360, 360),
            outline,
        )

    def data(self):
        data = asdict(self)
        if not self.outline:
            del data["outline"]
        return data


@dataclass(frozen=True)
class Scene:
    parts: tuple

    @classmethod
    def parse(cls, data):
        raw = data.get("parts") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PARTS:
            raise ValueError(f"Ожидалось от 1 до {MAX_PARTS} деталей")
        return cls(tuple(Part.parse(p) for p in raw))

    def data(self):
        return {"parts": [p.data() for p in self.parts]}


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
        parts = list(self.scene.parts)
        parts[index] = part
        self.apply(replace(self.scene, parts=tuple(parts)), self.mode)

    def undo(self):
        if self.history:
            self.scene, self.mode = self.history.pop()
