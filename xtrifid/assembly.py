"""Resolve a generated support graph, then freeze it as independent editable parts."""

from dataclasses import replace

from .components import fit_width, move_onto_support, ordered_components
from .construction import finish_loft, orient_cylinder
from .geometry import height_bounds
from .model import MAX_LENGTH, Scene, vector


def assemble(data, detail="medium", *, diameters=(), roles=()):
    if roles:
        data = ordered_components(data, roles)
    parts = Scene.parse(data, simplify_profiles=True).parts
    # Apply literal diameters only when a single cylindrical part fits the measurement.
    named_diameters = {
        d
        for d in diameters
        if sum(p.shape == "cylinder" and p.size.count(d) == 2 for p in parts) == 1
    }
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
        recipe = data["parts"][i]
        if part.shape == "loft":
            part = finish_loft(part, recipe)
        elif part.shape == "cylinder":
            part = orient_cylinder(part, recipe, named_diameters)
        s = supports[i]
        if s > 0:
            place(s - 1)
            if roles:
                part = fit_width(part, placed[s - 1], roles[i]["width"], detail)
                part = move_onto_support(part, placed[s - 1], detail)
        bottom, top = height_bounds(part, detail)
        if s >= 0:
            surface = 0.0
            if s:
                surface = tops[s - 1]
                if roles and roles[i]["joint"] == "joined":
                    support_floor, _ = height_bounds(placed[s - 1], detail)
                    surface -= 0.08 * min(top - bottom, surface - support_floor)
            delta = surface - bottom
            x, y, cz = part.center
            part = replace(part, center=vector((x, y, cz + delta), 3, -MAX_LENGTH, MAX_LENGTH))
            top += delta
        placed[i] = part
        tops[i] = top
        visiting.remove(i)

    for i in range(len(parts)):
        place(i)
    return Scene(tuple(placed[i] for i in range(len(parts))), detail)
