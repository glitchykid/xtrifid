"""Translate a natural-language request into bounded data using local Ollama."""

import http.client
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .assembly import assemble
from .model import MAX_LENGTH, MAX_OUTLINE, MAX_PARTS, MIN_SIZE, SHAPES, Part

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
            return Part.parse(scene.parts[index].data() | patch)

        return self._plan(
            prompt,
            "Edit only the selected 1-based part. Return only changed geometry fields. "
            "For a size-only edit return just the complete XYZ size array in metres. "
            f"Scene: {context}\nUser request: ",
            patch_schema,
            apply_patch,
        )

    def _plan(self, prompt, context, schema, parse):
        if not prompt.strip() or len(prompt) > MAX_PROMPT:
            raise ValueError(f"Введите описание от 1 до {MAX_PROMPT} символов")
        messages = [
            {"role": "system", "content": INSTRUCTIONS},
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
