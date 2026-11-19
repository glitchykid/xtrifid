"""Translate a natural-language request into bounded data using local Ollama."""

import http.client
import json
import os
from dataclasses import replace
from math import ulp
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .assembly import assemble
from .model import MAX_LENGTH, MAX_OUTLINE, MAX_PARTS, MIN_SIZE, SHAPES, Part
from .profile import move_profile_point, profile_points

MAX_PROMPT = 2000
RESPONSE_LIMIT = 524288


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
        "shape": {"type": "string", "enum": list(SHAPES)},
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
        placed_schema([s for s in SHAPES if s not in (*PROFILE_SHAPES, "tube")]),
        placed_schema(list(PROFILE_SHAPES), ("outline",)),
        placed_schema(["tube"], ("inner_diameter",)),
    ]
}
SCENE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["parts"],
    "properties": {
        "parts": {"type": "array", "minItems": 1, "maxItems": MAX_PARTS, "items": PLACED_PART}
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


class PlannerError(Exception):
    pass


class Planner:
    def __init__(self, url=None, model=None):
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("XTRIFID_MODEL", "qwen3.5:4b")

    def generate(self, prompt):
        return self._plan(
            prompt,
            """Build from the ground up. Give each part a support value meaning 'my bottom rests on':
0 = this part touches the floor. Positive = the 1-based index of a part BELOW this one.
-1 = freely positioned in world space, e.g. a side-mounted handle.
Example: a stand with two feet and a shelf: feet are parts 1 and 2, support=0 for BOTH;
shelf is part 3, support=1 (both feet have the same height). Its underside touches the feet.
Never put a foot on the object that it holds up. For a floor-standing object at least one
part has support=0. Keep corner supports inside the top footprint, inset by half their width.
Emit only the requested object. User request follows.\n""",
            SCENE,
            assemble,
        )

    def edit(self, prompt, scene, index):
        context = json.dumps(
            {"selected": index + 1, "parts": scene.data()["parts"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        editable = {k: v for k, v in PART["properties"].items() if k != "name"}
        patch_schema = {
            "type": "object",
            "properties": editable,
            "minProperties": 1,
            "additionalProperties": False,
        }

        def apply_patch(patch):
            if not isinstance(patch, dict) or not patch or not patch.keys() <= editable.keys():
                raise ValueError("Return only the changed geometry fields of the selected part")
            original = scene.parts[index]
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

    def _plan(self, prompt, context, schema, parse, instructions=INSTRUCTIONS):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT:
            raise ValueError(f"Введите описание от 1 до {MAX_PROMPT} символов")
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": context + prompt.strip()},
        ]
        for attempt in range(2):
            content = self._chat(messages, schema)
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

    def _chat(self, messages, schema):
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "messages": messages,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 6000},
        }
        request = Request(
            self.url + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=300) as response:
                raw = response.read(RESPONSE_LIMIT + 1)
            if len(raw) > RESPONSE_LIMIT:
                raise PlannerError("Ответ Ollama слишком большой")
            data = json.loads(raw)
            if data.get("done") is not True or data.get("done_reason") == "length":
                raise PlannerError("Ollama не завершила ответ. Попробуйте упростить описание.")
            content = data["message"]["content"]
            if not isinstance(content, str):
                raise ValueError
            return content
        except HTTPError as exc:
            detail = f" Проверьте наличие модели {self.model}." if exc.code == 404 else ""
            raise PlannerError(f"Ollama вернула HTTP {exc.code}.{detail}") from exc
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise PlannerError(
                "Нет ответа от Ollama. Проверьте сервис и повторите запрос."
            ) from exc
        except (KeyError, ValueError, TypeError, AttributeError, RecursionError) as exc:
            raise PlannerError("Неполный или неизвестный формат ответа Ollama") from exc
