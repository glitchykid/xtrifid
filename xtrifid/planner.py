"""Translate a natural-language request into bounded data using local Ollama."""

import json
import os
import re
from dataclasses import replace
from math import ulp

from .assembly import assemble
from .character import (
    BODY_SCALES_SCHEMA,
    CHARACTER_SCHEMA,
    build_character,
    character_anchor,
    with_character_pose,
)
from .components import COMPONENTS_SCHEMA, read_components
from .construction import CAP_STEPS, CYLINDER_AXES, LOFT_ENDS, LOFT_LEANS
from .model import (
    DETAILS,
    LOFT_COORDS,
    MAX_LENGTH,
    MAX_OUTLINE,
    MAX_PARTS,
    MIN_SIZE,
    PRIMITIVES,
    Part,
)
from .ollama import DEFAULT_URL, PlannerError, post_chat
from .ollama import ModelMissing as ModelMissing
from .organic import MAX_SECTIONS
from .profile import move_profile_point, profile_points
from .sculpt_plan import SCULPT_INSTRUCTIONS, SCULPT_RECIPE, sculpt_scene

MAX_PROMPT = 2000
MAX_ACCESSORY_DESCRIPTION = 160
OBJECT_SHAPES = (*PRIMITIVES, "loft")


def numbers(length, low, high):
    return {
        "type": "array",
        "minItems": length,
        "maxItems": length,
        "items": {"type": "number", "minimum": low, "maximum": high},
    }


PART = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 80},
        "shape": {"type": "string", "enum": list(OBJECT_SHAPES)},
        "size": numbers(3, MIN_SIZE, MAX_LENGTH),
        "center": numbers(3, -MAX_LENGTH, MAX_LENGTH),
        "rotation": numbers(3, -360, 360),
        "inner_diameter": {"type": "number", "minimum": 0.000001, "maximum": MAX_LENGTH},
        "sections": {
            "type": "array",
            "minItems": 3,
            "maxItems": MAX_SECTIONS,
            "items": {
                "type": "object",
                "properties": {
                    key: {
                        "type": "number",
                        "minimum": -1 if key.startswith("offset") else 0,
                        "maximum": 1,
                    }
                    for key in LOFT_COORDS
                },
                "required": list(LOFT_COORDS),
                "additionalProperties": False,
            },
        },
        "outline": {
            "type": "array",
            "minItems": 3,
            "maxItems": MAX_OUTLINE,
            "items": numbers(2, -1, 1),
        },
    },
    "required": ["name", "shape", "size", "center", "rotation"],
}


def placed_schema(shapes, extra=()):
    keys = PART["required"] + list(extra)
    properties = {key: PART["properties"][key] for key in keys}
    properties["shape"] = {"type": "string", "enum": shapes}
    if shapes == ["lathe"]:
        meridian = {
            "type": "object",
            "additionalProperties": False,
            "properties": dict.fromkeys(
                ("radius", "height"), {"type": "number", "minimum": 0, "maximum": 1}
            ),
            "required": ["radius", "height"],
        }
        properties["outline"] = properties["outline"] | {"items": meridian}
    if shapes == ["cylinder"]:
        # Establish the world direction before the model fills in local angles.
        properties = {
            "name": properties["name"],
            "shape": properties["shape"],
            "world_size": numbers(3, MIN_SIZE, MAX_LENGTH),
            "axis": {"type": "string", "enum": [*CYLINDER_AXES, "custom"]},
            **properties,
        }
    if shapes == ["loft"]:
        for end in ("bottom_end", "top_end"):
            properties[end] = {"type": "string", "enum": list(LOFT_ENDS)}
        properties["sections"] = properties["sections"] | {
            "maxItems": MAX_SECTIONS - 2 * CAP_STEPS
        }
    properties["support"] = {"type": "integer", "minimum": -1, "maximum": MAX_PARTS}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


PROFILE_SHAPES = ("extrusion", "lathe")
LOFT_PART = placed_schema(["loft"], ("sections",))
PLACED_PART = {
    "oneOf": [
        placed_schema(
            [s for s in PRIMITIVES if s not in (*PROFILE_SHAPES, "tube", "cylinder")]
        ),
        placed_schema(["cylinder"]),
        placed_schema(["extrusion"], ("outline",)),
        placed_schema(["lathe"], ("outline",)),
        placed_schema(["tube"], ("inner_diameter",)),
        LOFT_PART,
    ]
}
SUBJECT = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["subject", "accessories", "character"],
            "properties": {
                "subject": {"type": "string", "enum": ["character"]},
                "accessories": {
                    "type": "array",
                    "maxItems": MAX_PARTS,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_ACCESSORY_DESCRIPTION,
                    },
                },
                "character": CHARACTER_SCHEMA,
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["subject", "components", "form"],
            "properties": {
                "subject": {"type": "string", "enum": ["object"]},
                "components": COMPONENTS_SCHEMA,
                "form": {"type": "string", "enum": ["general"]},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["subject", "components", "form", "bottom_end", "top_end", "top_lean"],
            "properties": {
                "subject": {"type": "string", "enum": ["object"]},
                "components": COMPONENTS_SCHEMA,
                "form": {"type": "string", "enum": ["varying_sections"]},
                "bottom_end": LOFT_PART["properties"]["bottom_end"],
                "top_end": LOFT_PART["properties"]["top_end"],
                "top_lean": {"type": "string", "enum": ["as_drawn", *LOFT_LEANS]},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["subject", "body", "growth"],
            "properties": {
                "subject": {"type": "string", "enum": ["branched_organic"]},
                "body": {"type": "string", "minLength": 1, "maxLength": 80},
                "growth": {"type": "string", "minLength": 1, "maxLength": 160},
            },
        },
    ]
}


def parts_schema(limit, part_schema=PLACED_PART):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["parts"],
        "properties": {
            "parts": {"type": "array", "minItems": 1, "maxItems": limit, "items": part_schema}
        },
    }


def component_geometry_schema(roles):
    names = {"type": "string", "enum": [r["name"] for r in roles]}
    shapes = [
        shape | {"properties": shape["properties"] | {"name": names}}
        for shape in PLACED_PART["oneOf"]
    ]
    result = parts_schema(len(roles), {"oneOf": shapes})
    result["properties"]["parts"]["minItems"] = len(roles)
    return result


INSTRUCTIONS = """You are a precision 3D modeller. Output only JSON following the schema.
Use metres, Z up, X width, Y depth. size = full local bounding dimensions [X,Y,Z].
center = world centre of the part. rotation = degrees about X, then Y, then Z.
Respect quantities, dimensions, relative placement, negative space and symmetry.
Model the requested object in recognisable separate named pieces, with real dimensions
and sensible connections. Put the base on z=0 unless asked otherwise. Name each piece
uniquely in the language of the user. Use box for a cuboid, a rectangular block or a
rectangular-section beam. It has six planar rectangular faces and square corners unless
the user requests a bevel or rounding. Choose a rod's shape by its cross-section.
Sphere can be an ellipsoid. Cone points along +Z.
For any solid circular or elliptical disk use cylinder, size [diameter X, diameter Y,
thickness] in metres. Its disk starts in the XY plane; rotate 90 degrees about X for XZ.
For a cylinder world_size is [world width X, world depth Y, world height Z] of the
finished piece. Copy these dimensions directly from the request when given. The
cylinder axis is its central rod direction or its disk's face normal. Set axis
to vertical_z for a standing rod or a horizontal plate, depth_y for a front-facing
disk in XZ, width_x for a side-facing disk in YZ. For an oblique axis use custom and
supply the Euler rotation. The same shape generates rods by increasing its local Z dimension.
For a bushing, hollow straight pipe, sleeve, or flat washer use tube. For tube,
inner_diameter is the full width of the opening in metres. Torus hole faces +Z. For custom silhouettes use extrusion: outline is an ordered
simple XY polygon of 3-32 points in [-1,1], implicitly closed, with no duplicate or
collinear vertices. Its extents stretch to size X and Y, and it extrudes along Z.
For a vase, bottle or other turned solid, use lathe. Its outline
is a wall cross-section with named radius and height, both in [0,1], before rotation
around Z. Trace only the nonnegative radial half-plane. For a solid, a monotonic outside
meridian ending on radius zero can be closed automatically to a flat base. For a hollow
vessel include both its outside and inside wall. For a sculpted asymmetrical body use loft. Specify the bottom_end and top_end
separately: flat (level end face), pointed (sharp tip), or rounded (elliptical dome).
The mesher completes each rounded or pointed end at the last ring. Design elliptical sections from
height 0 (bottom) to height 1 (top), strictly increasing. In section coordinates
radius_x and radius_y are half-width and half-depth, normalized (about 0.5 at full size).
offset_right and offset_back move a ring towards +X and +Y, respectively; negative values
move left and forward. Each value is a fraction of the part's envelope, not metres.
For a flat end use positive radii. For a rounded end, taper the last few rings into
one point with radius_x=0 and radius_y=0. Only endpoints can be points.
The completed loft stretches to the specified size. Rotate the finished loft
if its main axis isn't vertical. Use sections only for loft, outline only for lathe or
extrusion. Preserve every requested feature, count and dimension at all detail levels.
No materials, ground plane or surrounding scenery."""

DETAIL_BRIEFS = {
    "low": "Use the fewest pieces and silhouette landmarks that retain the specified features.",
    "medium": "Describe the major contours, changes of curvature and distinct functional pieces.",
    "high": "Resolve secondary contours, transitions and edge details visible in the object. "
    "Place extra sections where curvature changes, not uniformly on straight spans.",
}

SUBJECT_INSTRUCTIONS = """Identify what the user's text asks to model. Return a single JSON object.
For a human, traveller, actor or other human figure, set subject to character.
List only explicitly requested items the figure holds, carries or wears in accessories. Preserve
sizes and attachment sites using the user's words. For example, a person carrying a bag
on one shoulder has accessories ["сумка через плечо"]. Ordinary anatomy, the cloak and
its hood are included in character. When no extra object was requested, use []. Specify full
height in metres, build (slim/average/broad), pose (standing/walking/arms_out),
cloak length (none/short/knee/ankle), and whether its hood is raised (true/false).
Defaults are 1.75 m, average build, standing upright, no cloak and no hood.
Use physique male for an explicitly requested man, female for a woman, otherwise neutral.
Only if explicitly specified, put multipliers for individual anatomical regions in
body_scales (1.0 is the reference anatomy). Face regions include head, ears, nose, lips
and chin. Use build alone for an overall broad or slim build.
A cloak of unspecified length reaches the knees. Convert centimetres to metres.
Example for a slim 165 cm tall human with outstretched arms, a short hoodless cape:
{"subject":"character","accessories":[],"character":{"height_m":1.65,"build":"slim","pose":"arms_out","cloak":"short","hood":false}}
For a single continuously branching organism choose subject "branched_organic".
Name its living body in body and describe the forks, limbs or stems growing from it in growth.
For every manufactured item, assemblage of separate objects, stack, vessel or tool choose
subject "object". An attached piece of hardware is a separate component. An unbranched
solid or shell also uses subject "object".
First list components if the text identifies two or more connected pieces or distinct
anatomical regions of the object. A single named object can have several such pieces.
Keep each identifiable role, including the load-bearing part and the part it carries.
Use their short, distinct names in the user's language. Preserve a stated count by listing
the instances separately, each with a unique name. Omit excluded items. Do not
list surface ends or empty space as separate pieces. For an undivided shell use [].
For each component, use the exact name of the piece directly beneath it as support.
Use "@floor" for the floor and "@free" for free positioning. Compare its horizontal width to that
support: wider, narrower, equal, or unspecified. Use the requested proportions and
the ordinary shape of the named object. A broad piece over a column is wider even
if the column is itself thick. Mark the joint "joined" for a continuous or attached
form, "resting" for stacked pieces. For a multi-component object choose form "general".
List supporting pieces first. Example for a thin aerial attached to a box:
[{"name":"Box","support":"@floor","width":"unspecified","joint":"resting"},
{"name":"Aerial","support":"Box","width":"narrower","joint":"joined"}].
For "@floor" and "@free" use width "unspecified" and joint "resting".
Choose form "varying_sections" for a single continuous sculpted solid whose cross-sections
taper asymmetrically or shift toward one side. Such a solid cannot be revolved about a
fixed axis. Extract its bottom and top end independently: flat for a planar base or
cut face, rounded for a dome, pointed for a sharp tip. For top_lean copy an explicit displacement
of the highest point: back (+Y), front (-Y), right (+X), left (-X); otherwise as_drawn.
Choose "general" for assemblies, mechanical frames, objects with holes, turned
objects, or shapes with a constant outline along their length."""

OBJECT_PLACEMENT = """Build objects from the ground up. Each part's support locates its underside:
0 = on the floor. A positive value is the 1-based index of its supporting part.
-1 = freely positioned in world space, e.g. a side-mounted handle.
For a shelf with two feet, give both feet support=0 and the shelf support=1. Never
put a foot on the object it holds. Keep corner supports inside the top footprint,
inset by half their width. Give at least one part of a floor-standing object support=0.
Requested object:\n"""

PRIMITIVE_NAMES = {
    *PRIMITIVES,
    "cube",
    "куб",
    "шар",
    "сфера",
    "цилиндр",
    "конус",
    "тор",
    "труба",
}

SIMPLE_CHARACTER = (
    re.compile(
        r"(?:пожалуйста,? )?(?:(?:сделай|создай|построй|смоделируй|сгенерируй) )?"
        r"(?:фигур[ау] )?(?P<body>человека?|мужчин[ау]|женщин[ау])"
        r"(?:(?P<cloak> с плащом| в плаще)(?: (?:с (?P<hood>капюшоном)|без капюшона))?"
        r"| без плаща)?"
    ),
    re.compile(
        r"(?:please,? )?(?:(?:create|make|model|generate) )?(?:a )?(?P<body>human|person|man|woman)"
        r"(?:(?P<cloak> (?:with|in|wearing) a (?:cape|cloak))"
        r"(?: (?:with a (?P<hood>hood)|without a hood))?| without a (?:cape|cloak))?"
    ),
)

SIGNED_CREST_AXIS = re.compile(
    r"(?:высш\w* точк\w*|верхушк\w*|вершин\w*|highest point|crest|apex|tip)"
    r"[^.!?;()\n]{0,40}(?:смещ\w*|сдвин\w*|наклон\w*|offset|shift\w*|lean\w*)"
    r"[^.!?;()\n]{0,32}\(\s*([+−-])\s*([xy])\s*\)",
    re.IGNORECASE,
)
DIAMETER = re.compile(
    r"(?:\bдиаметр(?:ом|а)?\b|\bdiameter(?: of)?\b|[ø⌀])\s*[:=]?\s*"
    r"(\d+(?:[.,]\d+)?)\s*(мм|см|м|mm|cm|m)\b",
    re.IGNORECASE,
)
METRIC_UNIT = {"мм": 1000, "mm": 1000, "см": 100, "cm": 100, "м": 1, "m": 1}


def simple_character(prompt):
    # Match the whole request: quantities, measurements, negation and accessories
    # outside this small grammar must reach the language model.
    text = " ".join(prompt.casefold().split()).rstrip(".!?").strip()
    for pattern in SIMPLE_CHARACTER:
        match = pattern.fullmatch(text)
        if match:
            spec = {"cloak": "knee" if match["cloak"] else "none", "hood": bool(match["hood"])}
            if match["body"] in ("мужчина", "мужчину", "man"):
                spec["physique"] = "male"
            elif match["body"] in ("женщина", "женщину", "woman"):
                spec["physique"] = "female"
            return spec
    return None


def checked_prompt(prompt):
    if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT or not prompt.strip():
        raise ValueError(f"Введите описание от 1 до {MAX_PROMPT} символов")
    return prompt.strip()


class _InvalidObjectHints(ValueError):
    """The subject was recognized, but its optional component plan is unusable."""


def read_subject(data, detail):
    if isinstance(data, dict) and data.get("subject") == "branched_organic":
        if data.keys() != {"subject", "body", "growth"} or any(
            not isinstance(data[key], str) or not data[key].strip() or len(data[key]) > limit
            for key, limit in (("body", 80), ("growth", 160))
        ):
            raise ValueError("Name the individual organic body and its branching growth")
        return None, [], {"body": data["body"], "growth": data["growth"]}
    if isinstance(data, dict) and data.get("subject") == "object":
        try:
            roles = read_components(data.get("components", []))
        except ValueError as exc:
            raise _InvalidObjectHints(str(exc)) from exc
        data = {k: v for k, v in data.items() if k != "components"}
        if len(roles) > 1:
            return None, [], {"components": roles}
        form = data.get("form", "general")
        if form == "general" and data.keys() <= {"subject", "form"}:
            return None, [], None
        loft_keys = {"subject", "form", "bottom_end", "top_end"}
        if form == "varying_sections" and data.keys() - {"top_lean"} == loft_keys:
            ends = {key: data[key] for key in ("bottom_end", "top_end")}
            lean = data.get("top_lean", "as_drawn")
            if all(end in LOFT_ENDS for end in ends.values()) and lean in (
                "as_drawn",
                *LOFT_LEANS,
            ):
                return None, [], ends | {"top_lean": lean}
        raise ValueError("Expected a general object or a single loft with both end types")
    if not isinstance(data, dict) or data.keys() != {"subject", "character", "accessories"}:
        raise ValueError(
            "A figure requires a subject, its character recipe and its accessory list"
        )
    if data["subject"] != "character":
        raise ValueError("Unknown subject")
    accessories = data["accessories"]
    if (
        not isinstance(accessories, list)
        or len(accessories) > MAX_PARTS
        or any(
            not isinstance(item, str)
            or not item.strip()
            or len(item) > MAX_ACCESSORY_DESCRIPTION
            for item in accessories
        )
    ):
        raise ValueError("List short descriptions of the requested accessories")
    spec = data["character"]
    if not isinstance(spec, dict) or not set(CHARACTER_SCHEMA["required"]) <= spec.keys():
        raise ValueError("A character needs height_m, build, pose, cloak and hood")
    return build_character(spec, detail), accessories, None


POINT_INSTRUCTIONS = """Translate the user's request to an operation on each local profile axis.
Return one JSON object with both named axes. For every unchanged axis return null.
For an axis to change, return {"op":"set" or "add","value":signed number,"unit":"mm","cm" or "m"}.
Copy the unit and quantity from the request. The application handles unit conversion.
Use "set" for an absolute target coordinate, including negative targets. Use "add" for
relative movement, with a negative value for decreasing coordinates. Right/left increase/
decrease the first axis, and up/down increase/decrease the second axis.
On a turned part, radius is measured from the centreline along local X; Z is local height."""

UNITS_PER_METRE = {"m": 1, "cm": 100, "mm": 1000}
POINT_OPERATIONS = ("set", "add")
MAX_POINT_VALUE = 2 * MAX_LENGTH * max(UNITS_PER_METRE.values())
AXIS_COMMAND = {
    "oneOf": [
        {"type": "null"},
        {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": list(POINT_OPERATIONS)},
                "value": {
                    "type": "number",
                    "minimum": -MAX_POINT_VALUE,
                    "maximum": MAX_POINT_VALUE,
                },
                "unit": {"type": "string", "enum": list(UNITS_PER_METRE)},
            },
            "required": ["op", "value", "unit"],
            "additionalProperties": False,
        },
    ]
}


def apply_axis_command(current, command):
    if command is None:
        return current
    if not isinstance(command, dict) or command.keys() != {"op", "unit", "value"}:
        raise ValueError("Expected an axis command with op, value and unit")
    operation, unit, value = command["op"], command["unit"], command["value"]
    if (
        operation not in POINT_OPERATIONS
        or not isinstance(unit, str)
        or unit not in UNITS_PER_METRE
    ):
        raise ValueError("Use set or add, in mm, cm or m")
    if type(value) not in (int, float) or not -MAX_POINT_VALUE <= value <= MAX_POINT_VALUE:
        raise ValueError("Expected a finite coordinate or displacement in range")
    metres = value / UNITS_PER_METRE[unit]
    return current + metres if operation == "add" else metres


class Planner:
    def __init__(self, url=None, model=None, local_server=None):
        self.url = url or os.environ.get("OLLAMA_URL")
        self.model = model or os.getenv("XTRIFID_MODEL", "qwen3.5:4b")
        self.local_server = local_server

    @property
    def url(self):
        return self._configured_url or DEFAULT_URL

    @url.setter
    def url(self, value):
        self._configured_url = value.rstrip("/") if value else None

    def generate(self, prompt, detail="medium"):
        prompt = checked_prompt(prompt)
        if detail not in DETAILS:
            raise ValueError("Выберите уровень детализации")
        modelling = f"{INSTRUCTIONS}\nMesh detail: {detail}. {DETAIL_BRIEFS[detail]}"
        spec = simple_character(prompt)
        if spec is not None:
            return build_character(spec, detail)
        figure, accessories, construction = None, [], None
        if prompt.casefold().rstrip(".!?") not in PRIMITIVE_NAMES:
            try:
                figure, accessories, construction = self._plan(
                    prompt,
                    "",
                    SUBJECT,
                    lambda data: read_subject(data, detail),
                    instructions=SUBJECT_INSTRUCTIONS,
                    num_ctx=2048,
                    num_predict=max(192, min(768, 2 * len(prompt))),
                )
            except PlannerError as exc:
                # After the repair attempt, an identified object can still use direct geometry.
                if not isinstance(exc.__cause__, _InvalidObjectHints):
                    raise
            if figure is not None and not accessories:
                return figure
        if figure is None:
            if construction and "growth" in construction:
                return self._plan(
                    prompt,
                    f"Body and connected growth: {json.dumps(construction, ensure_ascii=False)}.\n"
                    f"Requested detail: {detail}. {DETAIL_BRIEFS[detail]}\n",
                    SCULPT_RECIPE,
                    lambda data: sculpt_scene(data, detail),
                    instructions=SCULPT_INSTRUCTIONS,
                    num_ctx=4096,
                    num_predict=2400,
                )
            diameters = {
                float(value.replace(",", ".")) / METRIC_UNIT[unit.casefold()]
                for value, unit in DIAMETER.findall(prompt)
            }
            if construction and "components" in construction:
                roles = construction["components"]
                context = (
                    "Build one individually editable part for each specified role. Copy every "
                    "given name exactly. Preserve widths, contacts and quantities. The component "
                    "plan is "
                    + json.dumps(roles, ensure_ascii=False)
                    + ".\n"
                    + OBJECT_PLACEMENT
                )
                return self._plan(
                    prompt,
                    context,
                    component_geometry_schema(roles),
                    lambda data: assemble(data, detail, diameters=diameters, roles=roles),
                    instructions=modelling,
                )
            loft_recipe = construction
            axis = SIGNED_CREST_AXIS.search(prompt)
            controls = dict(loft_recipe or {})
            if axis:
                negative, positive = (
                    ("left", "right") if axis[2].lower() == "x" else ("front", "back")
                )
                controls["top_lean"] = positive if axis[1] == "+" else negative
                controls["measured_lean"] = bool(re.search(r"\d", axis[0]))

            def assemble_object(data):
                if controls:
                    parts = data.get("parts") if isinstance(data, dict) else None
                    single_loft = (
                        isinstance(parts, list)
                        and len(parts) == 1
                        and isinstance(parts[0], dict)
                        and parts[0].get("shape") == "loft"
                    )
                    if single_loft:
                        data = {"parts": [parts[0] | controls]}
                    elif loft_recipe:
                        raise ValueError("A single offset silhouette requires one loft")
                return assemble(data, detail, diameters=diameters)

            return self._plan(
                prompt,
                (
                    f"Build one continuous loft, preserving its end treatment {json.dumps(loft_recipe)}. "
                    if loft_recipe
                    else ""
                )
                + OBJECT_PLACEMENT,
                parts_schema(1, LOFT_PART) if loft_recipe else parts_schema(MAX_PARTS),
                assemble_object,
                instructions=modelling,
            )

        limit = MAX_PARTS - len(figure.parts)
        anchors = {
            name: [round(c, 5) for c in character_anchor(figure, name)]
            for name in ("Туловище", "Голова", "Кисть левая", "Кисть правая")
        }
        context = (
            "The body, cloak and hood are already modelled as requested. Generate only these items: "
            + json.dumps(accessories, ensure_ascii=False)
            + ". Use support=-1 "
            "and attach them in world space. The figure faces -Y. Landmark centres in metres: "
            + json.dumps(anchors, ensure_ascii=False)
            + "\nUser request:\n"
        )

        def attach(data):
            parts = data.get("parts") if isinstance(data, dict) else None
            if not isinstance(parts, list) or not 1 <= len(parts) <= limit:
                raise ValueError(f"Return 1 to {limit} requested accessory parts")
            extras = assemble(data, detail)
            return replace(figure, parts=figure.parts + extras.parts)

        return self._plan(prompt, context, parts_schema(limit), attach, instructions=modelling)

    def edit(self, prompt, scene, index):
        if type(index) is not int or not 0 <= index < len(scene.parts):
            raise ValueError("Выберите существующую деталь")
        original = scene.parts[index]
        # Keep large photo/animation payloads local to the editable document.
        parts = [p.data(include_geometry=False) for p in scene.parts]
        for other, part in enumerate(parts):
            if other != index:
                for key in ("sections", "outline", "nodes"):
                    part.pop(key, None)
            elif part.get("sections"):
                part["sections"] = [dict(zip(LOFT_COORDS, row)) for row in part["sections"]]
        context = json.dumps(
            {"parts": parts, "selected": index + 1},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        editable = {k: v for k, v in PART["properties"].items() if k != "name"}
        if original.shape not in OBJECT_SHAPES:
            editable["shape"] = {"type": "string", "enum": [original.shape, *OBJECT_SHAPES]}
        anatomy_hint = ""
        if original.shape == "human":
            editable["pose"] = CHARACTER_SCHEMA["properties"]["pose"]
            editable["physique"] = CHARACTER_SCHEMA["properties"]["physique"]
            editable["body_scales"] = BODY_SCALES_SCHEMA | {"minProperties": 1}
            anatomy_hint = (
                "For an anatomical region, return body_scales with just the changed regions. "
                "Scales are relative to the base anatomy (1.0). Keep whole-body dimensions. "
            )
        patch_schema = {
            "type": "object",
            "properties": editable,
            "minProperties": 1,
            "additionalProperties": False,
        }

        def apply_patch(patch):
            if not isinstance(patch, dict) or not patch or not patch.keys() <= editable.keys():
                raise ValueError("Return only the changed geometry fields of the selected part")
            if "sections" in patch and patch.get("shape", original.shape) != "loft":
                raise ValueError("Only a loft has sections")
            basis = original
            if "pose" in patch:
                if patch.get("shape", original.shape) != "human":
                    raise ValueError("Only a human body can be posed")
                if original.pose != patch["pose"]:
                    basis = with_character_pose(original, patch["pose"])
            original_data = basis.data()
            if "body_scales" in patch:
                scales = patch["body_scales"]
                if (
                    patch.get("shape", original.shape) != "human"
                    or not isinstance(scales, dict)
                    or not scales
                ):
                    raise ValueError("Expected region multipliers on the selected figure")
                patch = patch | {"body_scales": original_data.get("body_scales", {}) | scales}
            updated = Part.parse(original_data | patch)
            # Model decimals can echo a computed coordinate one floating-point step away.
            stable = {}
            for field in ("size", "center", "rotation"):
                stable[field] = tuple(
                    a if abs(a - b) <= max(ulp(a), ulp(b)) else b
                    for a, b in zip(getattr(original, field), getattr(updated, field))
                )
            return replace(updated, **stable)

        return self._plan(
            prompt,
            "Edit only the selected 1-based part. Return only changed geometry fields. "
            "For a size-only edit return just the complete XYZ size array in metres. "
            f"{anatomy_hint}Scene: {context}\nUser request: ",
            patch_schema,
            apply_patch,
        )

    def edit_point(self, prompt, scene, part_index, point_index):
        if type(part_index) is not int or not 0 <= part_index < len(scene.parts):
            raise ValueError("Выберите существующую деталь")
        part = scene.parts[part_index]
        points = profile_points(part)
        if type(point_index) is not int or not 0 <= point_index < len(points):
            raise ValueError("Выберите существующую точку профиля")
        current = points[point_index]
        axes = ("x", "y") if part.shape in ("box", "extrusion") else ("radius", "z")
        context = json.dumps(
            {
                "part": part.name,
                "shape": part.shape,
                "axes": axes,
                "profile_metres": points,
                "selected_point_1_based": point_index + 1,
                "selected_coordinates_metres": dict(zip(axes, current)),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        schema = {
            "type": "object",
            "properties": dict.fromkeys(axes, AXIS_COMMAND),
            "required": list(axes),
            "additionalProperties": False,
        }

        def apply_commands(commands):
            if not isinstance(commands, dict) or commands.keys() != set(axes):
                raise ValueError(f"Expected exactly these axes: {', '.join(axes)}")
            target = tuple(
                apply_axis_command(coord, commands[axis]) for axis, coord in zip(axes, current)
            )
            return move_profile_point(part, point_index, target)

        return self._plan(
            prompt,
            f"Profile: {context}\nUser request: ",
            schema,
            apply_commands,
            instructions=POINT_INSTRUCTIONS,
        )

    def _plan(
        self,
        prompt,
        context,
        schema,
        parse,
        instructions=INSTRUCTIONS,
        num_ctx=8192,
        num_predict=6000,
    ):
        prompt = checked_prompt(prompt)
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": context + prompt},
        ]
        for attempt in range(2):
            content = self._chat(messages, schema, num_ctx=num_ctx, num_predict=num_predict)
            try:
                return parse(json.loads(content))
            except (ValueError, RecursionError) as exc:
                if attempt:
                    raise PlannerError(f"Модель вернула некорректную геометрию. {exc}") from exc
                messages.extend(
                    [
                        {"role": "assistant", "content": content[:16000]},
                        {"role": "user", "content": f"Correct the invalid data: {exc}"},
                    ]
                )

    def _chat(self, messages, schema, *, num_ctx, num_predict):
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "messages": messages,
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
        }
        raw = post_chat(
            self._configured_url, self.model, json.dumps(payload).encode(), self.local_server
        )
        try:
            data = json.loads(raw)
            if data.get("done") is not True or data.get("done_reason") == "length":
                raise PlannerError("Ollama не завершила ответ. Попробуйте упростить описание.")
            content = data["message"]["content"]
            if not isinstance(content, str):
                raise ValueError
            return content
        except (KeyError, ValueError, TypeError, AttributeError, RecursionError) as exc:
            raise PlannerError("Неполный или неизвестный формат ответа Ollama") from exc
