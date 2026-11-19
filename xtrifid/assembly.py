"""Resolve a generated support graph, then freeze it as independent editable parts."""

from dataclasses import replace

from .geometry import build
from .model import MAX_LENGTH, Scene, vector


def assemble(data):
    parts = Scene.parse(data).parts
    supports = [p.get("support") for p in data["parts"]]
    if any(type(s) is not int or not -1 <= s <= len(parts) for s in supports):
        raise ValueError("support must be -1 (free), 0 (floor) or a 1-based part index")
    placed = {}
    tops = {}
    visiting = set()

    def place(i):
        if i in placed:
            return
        if i in visiting:
            raise ValueError("Cyclic support graph: parts cannot support each other")
        visiting.add(i)
        part = parts[i]
        z = build(part, "render").vertices[:, 2]
        bottom, top = float(z.min()), float(z.max())
        s = supports[i]
        if s >= 0:
            surface = 0.0
            if s:
                place(s - 1)
                surface = tops[s - 1]
            delta = surface - bottom
            x, y, cz = part.center
            part = replace(part, center=vector((x, y, cz + delta), 3, -MAX_LENGTH, MAX_LENGTH))
            top += delta
        placed[i] = part
        tops[i] = top
        visiting.remove(i)

    for i in range(len(parts)):
        place(i)
    return Scene(tuple(placed[i] for i in range(len(parts))))
