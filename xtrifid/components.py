"""Keep the named components and relative proportions of a text description."""

from dataclasses import replace

from .geometry import axis_bounds, rotation_matrix
from .model import MAX_LENGTH, MAX_PARTS, MIN_SIZE, vector

WIDTH_RELATIONS = ("unspecified", "wider", "narrower", "equal")
JOINTS = ("resting", "joined")
COMPONENTS_SCHEMA = {
    "type": "array",
    "maxItems": MAX_PARTS,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 80},
            "support": {"type": "string", "minLength": 1, "maxLength": 80},
            "width": {"type": "string", "enum": list(WIDTH_RELATIONS)},
            "joint": {"type": "string", "enum": list(JOINTS)},
        },
        "required": ["name", "support", "width", "joint"],
    },
}


def read_components(raw):
    if not isinstance(raw, list) or len(raw) > MAX_PARTS:
        raise ValueError("List the named components of the requested object")
    raw = list(raw)
    names = {"@floor": 0, "@free": -1}
    for i, role in enumerate(raw, 1):
        if not isinstance(role, dict) or role.keys() != {"name", "support", "width", "joint"}:
            raise ValueError("Each component needs name, support, width and joint")
        name = role["name"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or not name.isprintable()
            or len(name) > 80
            or name.casefold() in names
        ):
            raise ValueError("Give every component a unique readable name")
        names[name.casefold()] = i
    for i, role in enumerate(raw, 1):
        name = role["name"]
        support = role["support"]
        if isinstance(support, str):
            if support.casefold() not in names:
                raise ValueError(
                    f"Choose the exact supporting component name for {name}, or @floor or @free"
                )
            support = names[support.casefold()]
            raw[i - 1] = role | {"support": support}
        if (
            type(support) is int
            and support == i
            and len(raw) == 2
            and role["joint"] == "joined"
        ):
            other = raw[2 - i]
            if (
                isinstance(other, dict)
                and other.get("support") == 0
                and other.get("joint") == "resting"
            ):
                # A joined component cannot support itself. The other, grounded component
                # is the sole possible attachment in this two-piece assembly.
                support = 3 - i
                raw[i - 1] = role | {"support": support}
        if (
            type(support) is not int
            or not -1 <= support <= len(raw)
            or support == i
            or role["width"] not in WIDTH_RELATIONS
            or role["joint"] not in JOINTS
        ):
            raise ValueError("Invalid component relation")
        if support <= 0 and (role["width"] != "unspecified" or role["joint"] == "joined"):
            raise ValueError(
                f"Component {i} ({name}) has a width or joint relative to a missing support. "
                "Use @floor and width=unspecified for the grounded piece. "
                "Set the upper piece support to the exact name of the piece beneath it."
            )
    for i in range(len(raw)):
        chain = set()
        while i >= 0:
            if i in chain:
                raise ValueError("Component supports form a cycle")
            chain.add(i)
            i = raw[i]["support"] - 1
    return raw


def ordered_components(data, roles):
    parts = data.get("parts") if isinstance(data, dict) else None
    if (
        not isinstance(parts, list)
        or len(parts) != len(roles)
        or not all(isinstance(p, dict) and isinstance(p.get("name"), str) for p in parts)
    ):
        raise ValueError("Return exactly one geometry for each named component")
    by_name = {part["name"]: part for part in parts}
    if by_name.keys() != {role["name"] for role in roles}:
        raise ValueError("Every component name must occur exactly once")
    return {"parts": [by_name[r["name"]] | {"support": r["support"]} for r in roles]}


def fit_width(part, support, relation, detail):
    if relation == "unspecified":
        return part
    low, high = axis_bounds(part, detail, 0)
    support_low, support_high = axis_bounds(support, detail, 0)
    ratio = (high - low) / (support_high - support_low)
    if relation == "wider" and ratio <= 1 + 1e-9:
        target = 1.55
    elif relation == "narrower" and ratio >= 1 - 1e-9:
        target = 1 / 1.55
    elif relation == "equal":
        target = 1.0
    else:
        return part
    up = abs(rotation_matrix(part.rotation)[2])
    if max(up) < 1 - 1e-9:
        raise ValueError(
            "Adjust the local dimensions of the tilted component to satisfy its required world width"
        )
    size = [
        value if vertical > 0.5 else value * target / ratio
        for value, vertical in zip(part.size, up)
    ]
    return replace(part, size=vector(size, 3, MIN_SIZE, MAX_LENGTH))


def move_onto_support(part, support, detail):
    center = list(part.center)
    for dimension in (0, 1):
        lo, hi = axis_bounds(part, detail, dimension)
        base_lo, base_hi = axis_bounds(support, detail, dimension)
        if hi <= base_lo or base_hi <= lo:
            center[dimension] += (base_lo + base_hi - lo - hi) / 2
    return replace(part, center=vector(center, 3, -MAX_LENGTH, MAX_LENGTH))
