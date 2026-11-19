"""Translate a natural-language request into bounded data using local Ollama."""

import json
import os
import re
from dataclasses import replace
from math import ulp

from .assembly import assemble
from .character import CHARACTER_SCHEMA, build_character
from .model import DETAILS, MAX_LENGTH, MAX_OUTLINE, MAX_PARTS, MIN_SIZE, PRIMITIVES, Part
from .ollama import DEFAULT_URL, PlannerError, post_chat
from .ollama import ModelMissing as ModelMissing
from .profile import move_profile_point, profile_points

MAX_PROMPT = 2000
MAX_ACCESSORY_DESCRIPTION = 160


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
        "shape": {"type": "string", "enum": list(PRIMITIVES)},
        "size": numbers(3, MIN_SIZE, MAX_LENGTH),
        "center": numbers(3, -MAX_LENGTH, MAX_LENGTH),
        "rotation": numbers(3, -360, 360),
        "inner_diameter": {"type": "number", "minimum": 0.000001, "maximum": MAX_LENGTH},
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
    properties["support"] = {"type": "integer", "minimum": -1, "maximum": MAX_PARTS}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


PROFILE_SHAPES = ("extrusion", "lathe")
PLACED_PART = {
    "oneOf": [
        placed_schema([s for s in PRIMITIVES if s not in (*PROFILE_SHAPES, "tube")]),
        placed_schema(list(PROFILE_SHAPES), ("outline",)),
        placed_schema(["tube"], ("inner_diameter",)),
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
            "required": ["subject"],
            "properties": {"subject": {"type": "string", "enum": ["object"]}},
        },
    ]
}


def parts_schema(limit):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["parts"],
        "properties": {
            "parts": {"type": "array", "minItems": 1, "maxItems": limit, "items": PLACED_PART}
        },
    }


INSTRUCTIONS = """You are a precision 3D modeller. Output only JSON following the schema.
Use metres, Z up, X width, Y depth. size = full local bounding dimensions [X,Y,Z].
center = world centre of the part. rotation = degrees about X, then Y, then Z.
Respect quantities, dimensions, relative placement, negative space and symmetry.
Model the requested object in recognisable separate named pieces, with real dimensions
and sensible connections. Put the base on z=0 unless asked otherwise. Name each piece
uniquely in the language of the user. Sphere can be an ellipsoid. Cone points along +Z.
A cylindrical item has size [outer diameter, outer diameter, length] in metres.
For a bushing, hollow straight pipe, sleeve, or flat washer use tube. For tube,
inner_diameter is the full width of the opening in metres. Torus hole faces +Z. For custom silhouettes use extrusion: outline is an ordered
simple XY polygon of 3-32 points in [-1,1], implicitly closed, with no duplicate or
collinear vertices. Its extents stretch to size X and Y, and it extrudes along Z.
For a vase, bottle or other turned solid, use lathe. Its outline
is the closed wall cross-section [radius, height], both in [0,1], before rotation
around Z.
Omit outline for other shapes. No materials, ground plane or surrounding scenery."""

SUBJECT_INSTRUCTIONS = """Identify what the user's text asks to model. Return a single JSON object.
For a human, traveller, actor or other human figure, set subject to character.
List each additional item the figure holds, carries or wears in accessories. Preserve
sizes and attachment sites using the user's words. For example, a person carrying a bag
on one shoulder has accessories ["сумка через плечо"]. Ordinary anatomy, the cloak and
its hood are included in character. When no extra object was requested, use []. Specify full
height in metres, build (slim/average/broad), pose (standing/walking/arms_out),
cloak length (none/short/knee/ankle), and whether its hood is raised (true/false).
Defaults are 1.75 m, average build, standing upright, no cloak and no hood.
A cloak of unspecified length reaches the knees. Convert centimetres to metres.
Example for a slim 165 cm tall human with outstretched arms, a short hoodless cape:
{"subject":"character","accessories":[],"character":{"height_m":1.65,"build":"slim","pose":"arms_out","cloak":"short","hood":false}}
For anything else, even an object made for someone to use, return {"subject":"object"}."""

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
        r"(?:фигур[ау] )?человека?"
        r"(?:(?P<cloak> с плащом| в плаще)(?: (?:с (?P<hood>капюшоном)|без капюшона))?"
        r"| без плаща)?"
    ),
    re.compile(
        r"(?:please,? )?(?:(?:create|make|model|generate) )?(?:a )?(?:human|person)"
        r"(?:(?P<cloak> (?:with|in|wearing) a (?:cape|cloak))"
        r"(?: (?:with a (?P<hood>hood)|without a hood))?| without a (?:cape|cloak))?"
    ),
)


def simple_character(prompt):
    # Match the whole request: quantities, measurements, negation and accessories
    # outside this small grammar must reach the language model.
    text = " ".join(prompt.casefold().split()).rstrip(".!?").strip()
    for pattern in SIMPLE_CHARACTER:
        match = pattern.fullmatch(text)
        if match:
            return {"cloak": "knee" if match["cloak"] else "none", "hood": bool(match["hood"])}
    return None


def checked_prompt(prompt):
    if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT or not prompt.strip():
        raise ValueError(f"Введите описание от 1 до {MAX_PROMPT} символов")
    return prompt.strip()


def read_subject(data, detail):
    if data == {"subject": "object"}:
        return None, []
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
    if not isinstance(spec, dict) or spec.keys() != CHARACTER_SCHEMA["properties"].keys():
        raise ValueError("A character needs height_m, build, pose, cloak and hood")
    return build_character(spec, detail), accessories


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
        spec = simple_character(prompt)
        if spec is not None:
            return build_character(spec, detail)
        figure = None
        if prompt.casefold().rstrip(".!?") not in PRIMITIVE_NAMES:
            figure, accessories = self._plan(
                prompt,
                "",
                SUBJECT,
                lambda data: read_subject(data, detail),
                instructions=SUBJECT_INSTRUCTIONS,
                num_ctx=2048,
                num_predict=192,
            )
            if figure is not None and not accessories:
                return figure
        if figure is None:
            return self._plan(
                prompt,
                OBJECT_PLACEMENT,
                parts_schema(MAX_PARTS),
                lambda data: assemble(data, detail),
            )

        limit = MAX_PARTS - len(figure.parts)
        anchors = {
            p.name: [round(c, 5) for c in p.center]
            for p in figure.parts
            if p.name in ("Туловище", "Голова", "Кисть левая", "Кисть правая")
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

        return self._plan(prompt, context, parts_schema(limit), attach)

    def edit(self, prompt, scene, index):
        if type(index) is not int or not 0 <= index < len(scene.parts):
            raise ValueError("Выберите существующую деталь")
        original = scene.parts[index]
        parts = scene.data()["parts"]
        for part in parts:
            # Loft sections are fixed anatomical controls. Dimensions and transforms
            # suffice for this operation and fit into the constrained model context.
            part.pop("sections", None)
        context = json.dumps(
            {"parts": parts, "selected": index + 1},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        editable = {k: v for k, v in PART["properties"].items() if k != "name"}
        if original.shape not in PRIMITIVES:
            editable["shape"] = {"type": "string", "enum": [original.shape, *PRIMITIVES]}
        patch_schema = {
            "type": "object",
            "properties": editable,
            "minProperties": 1,
            "additionalProperties": False,
        }

        def apply_patch(patch):
            if not isinstance(patch, dict) or not patch or not patch.keys() <= editable.keys():
                raise ValueError("Return only the changed geometry fields of the selected part")
            updated = Part.parse(original.data() | patch)
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
            f"Scene: {context}\nUser request: ",
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
