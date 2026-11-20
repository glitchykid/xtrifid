"""A bounded graph is the editable construction recipe for a continuous organic form."""

from .model import MAX_LENGTH, MIN_SIZE, Part, Scene, vector
from .sculpt import MAX_NODES, MIN_RADIUS, parse_nodes

NAME = {"type": "string", "minLength": 1, "maxLength": 64}
COORDS = ("right", "back", "up", "radius")
ROOT_FIELDS = {
    "name": NAME,
    **dict.fromkeys(COORDS[:3], {"type": "number", "minimum": -1, "maximum": 1}),
    "radius": {"type": "number", "minimum": MIN_RADIUS, "maximum": 1},
}
NODE_FIELDS = {"name": NAME, "parent": NAME, **{k: ROOT_FIELDS[k] for k in COORDS}}
SIZE_KEYS = ("width_m", "depth_m", "height_m")


def object_schema(fields):
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


SCULPT_RECIPE = object_schema(
    {
        "name": NAME,
        **dict.fromkeys(
            SIZE_KEYS, {"type": "number", "minimum": MIN_SIZE, "maximum": MAX_LENGTH}
        ),
        "root": object_schema(ROOT_FIELDS),
        "nodes": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_NODES - 1,
            "items": object_schema(NODE_FIELDS),
        },
    }
)

SCULPT_INSTRUCTIONS = """Build one connected branching skeleton with rounded tapered limbs.
Use a meaningful short name for the object and each anatomical joint in the user's own language.
The sole root is the base of the body. Every additional node has a unique name and specifies
the exact name of its parent (root or any junction). Each link forms a tapered tube.
A trunk starts at the base and ends at a junction; create a node for this junction.
A fork needs a separate child extending to each tip. Bends can have intermediate nodes.
Keep all requested counts, thicknesses, directions, and proportions. Do not add unrelated parts.
width_m = full left-right dimension, depth_m = front-back dimension, height_m = vertical dimension in metres.
Convert the stated height to metres and put it in height_m. Keep stated sizes for other dimensions too.
Node coordinates range from -1 to 1; up increases toward the top, right increases toward the RIGHT,
back increases toward the BACK. A twig going backwards ends at a LARGER back coordinate than its parent.
A vertical stalk starts low (e.g. up=-0.8) and grows up toward positive values.
Radius is measured in the same coordinate space. Use most of the coordinate range along the longest dimension.
The final connected surface fits the requested physical size exactly."""


def sculpt_scene(data, detail):
    if not isinstance(data, dict) or data.keys() != SCULPT_RECIPE["properties"].keys():
        raise ValueError("Return name, width_m, depth_m, height_m, root, and nodes")
    size = vector([data[key] for key in SIZE_KEYS], 3, MIN_SIZE, MAX_LENGTH)
    root, children = data["root"], data["nodes"]
    if (
        not isinstance(root, dict)
        or root.keys() != ROOT_FIELDS.keys()
        or not isinstance(children, list)
        or not 1 <= len(children) < MAX_NODES
    ):
        raise ValueError("Return one named root and a list of child nodes")
    nodes = [(root["name"], -1, *(root[key] for key in COORDS))]
    for child in children:
        if not isinstance(child, dict) or child.keys() != NODE_FIELDS.keys():
            raise ValueError("A child needs name, parent, right, back, up and radius")
        nodes.append((child["name"], child["parent"], *(child[key] for key in COORDS)))
    part = Part.parse(
        {
            "shape": "sculpt",
            "name": data["name"],
            "size": size,
            "center": (0, 0, size[2] / 2),
            "rotation": (0, 0, 0),
            "nodes": parse_nodes(nodes),
        }
    )
    return Scene((part,), detail)
