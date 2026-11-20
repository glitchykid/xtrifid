"""Turn the visual model's individual geometric observations into editable solids."""

import json
from math import atan2, cos, degrees, hypot, radians, sin, sqrt

from .model import Part, Scene, vector
from .planner import MAX_PROMPT, PlannerError

FORMS = (
    "hollow_revolution",
    "shallow_dish",
    "vertical_loop",
    "horizontal_loop",
    "spindle",
    "ellipsoid",
    "cylinder",
    "cone",
    "bar",
    "flat_disc",
    "box",
)
JOINTS = ("free", "rest", "side", "inside")
FIELDS = {"name", "form", "size", "center", "yaw", "parent", "joint"}

# Both shell meridians enclose material on either side of the lip and leave
# their openings clear. Radii and heights are fractions of local part bounds.
SHELL = (
    (0, 0),
    (0.57, 0),
    (0.65, 0.045),
    (0.72, 0.16),
    (0.87, 0.52),
    (0.975, 0.87),
    (1, 0.976),
    (0.994, 0.997),
    (0.973, 1),
    (0.941, 0.987),
    (0.935, 0.92),
    (0.854, 0.57),
    (0.706, 0.23),
    (0.60, 0.13),
    (0, 0.13),
)
DISH = (
    (0, 0),
    (0.38, 0),
    (0.42, 0.07),
    (0.67, 0.25),
    (0.90, 0.60),
    (1, 0.89),
    (0.993, 0.98),
    (0.972, 1),
    (0.94, 0.93),
    (0.85, 0.67),
    (0.63, 0.36),
    (0.38, 0.22),
    (0, 0.22),
)


def _contact_height(parent, x, y):
    if parent["form"] not in ("shallow_dish", "hollow_revolution"):
        return parent["center"][2] + parent["size"][2] / 2
    yaw = radians(parent["yaw"])
    dx, dy = x - parent["center"][0], y - parent["center"][1]
    u, v = dx * cos(yaw) + dy * sin(yaw), -dx * sin(yaw) + dy * cos(yaw)
    radius = hypot(2 * u / parent["size"][0], 2 * v / parent["size"][1])
    profile = DISH if parent["form"] == "shallow_dish" else SHELL
    hits = [
        z + (radius - r) * (zz - z) / (rr - r)
        for (r, z), (rr, zz) in zip(profile, profile[1:])
        if min(r, rr) <= radius <= max(r, rr) and rr != r
    ]
    return parent["center"][2] + (max(hits, default=1) - 0.5) * parent["size"][2]


def construct(items, detail):
    if not isinstance(items, list) or not 1 <= len(items) <= 12:
        raise ValueError("Для сборной формы нужно указать от 1 до 12 наблюдаемых деталей")
    records, parts = [], []
    for item in items:
        if not isinstance(item, dict) or item.keys() != FIELDS:
            raise ValueError("Неполная деталь визуальной сборки")
        form, joint, parent = item["form"], item["joint"], item["parent"]
        if (
            form not in FORMS
            or joint not in JOINTS
            or type(parent) is not int
            or not -1 <= parent < len(records)
            or (parent == -1 and joint not in ("free", "rest"))
        ):
            raise ValueError("Проверьте форму и ссылку на более раннюю опорную деталь")
        sx, sy, sz = vector(item["size"], 3, 0.001, 4)
        x, y, z = vector(item["center"], 3, -4, 4)
        yaw = vector([item["yaw"]], 1, -360, 360)[0]
        if form == "vertical_loop" and sy > sx:
            # A pierced loop's short axis is the thickness normal to its plane.
            sx, sy = max(sx, sy), min(sx, 0.30 * min(sy, sz))
        if joint == "inside" and any(
            term in item["name"].casefold() for term in ("liquid", "fluid", "жидк")
        ):
            form = "flat_disc"
        if parent >= 0:
            support = records[parent]
            px, py, pz = support["center"]
            if (
                joint == "inside"
                and form == "flat_disc"
                and support["form"] in ("hollow_revolution", "shallow_dish")
            ):
                # The top of a fill touches the closed inner meridian all
                # around. Keep the estimated level between the floor and rim.
                width, depth, h = support["size"]
                profile = SHELL if support["form"] == "hollow_revolution" else DISH
                inner = list(
                    reversed(profile[profile.index(max(profile, key=lambda p: p[1])) :])
                )
                floor = inner[0][1]
                level = min(0.968, max(floor + 0.035, (z + sz / 2 - pz) / h + 0.5))
                radius = next(
                    r + (rr - r) * (level - a) / (b - a)
                    for (r, a), (rr, b) in zip(inner, inner[1:])
                    if a < level <= b
                )
                x, y = px, py
                yaw = support["yaw"]
                sx, sy = 0.994 * width * radius, 0.994 * depth * radius
                sz = min(sz, h * min(0.04, level - floor))
                z = pz + h * (level - 0.5) - sz / 2
            if joint == "rest":
                z = _contact_height(support, x, y) + sz / 2
            if joint == "side" and form == "vertical_loop":
                # Attach the inner arc to the side wall. The photograph's XY
                # displacement determines which side of the parent is visible.
                dx, dy = x - px, y - py
                if hypot(dx, dy) > 1e-5:
                    yaw = degrees(atan2(dy, dx))
                ca, sa = cos(radians(yaw)), sin(radians(yaw))
                relative = radians(yaw - support["yaw"])
                w, d, h = support["size"]
                wall = 1 / sqrt((2 * cos(relative) / w) ** 2 + (2 * sin(relative) / d) ** 2)
                # Tapered shells are narrower below the rim.
                if support["form"] == "hollow_revolution":
                    fraction = max(0, min(1, (z - pz) / h + 0.5))
                    wall *= 0.65 + 0.35 * fraction
                x, y = px + ca * (wall + 0.23 * sx), py + sa * (wall + 0.23 * sx)
        records.append({"form": form, "center": (x, y, z), "size": (sx, sy, sz), "yaw": yaw})
        data = dict(
            name=item["name"],
            shape=form,
            size=(sx, sy, sz),
            center=(x, y, z),
            rotation=(0, 0, yaw),
        )
        if form in ("hollow_revolution", "shallow_dish"):
            data.update(shape="lathe", outline=SHELL if form == "hollow_revolution" else DISH)
        elif form in ("vertical_loop", "horizontal_loop"):
            if form == "vertical_loop":
                data.update(shape="torus", size=(sx, sz, sy), rotation=(90, 0, yaw))
            else:
                data.update(shape="torus", size=(sx, sy, sz))
        elif form == "spindle":
            data.update(
                shape="loft",
                size=(sz, sy, sx),
                rotation=(0, 90, yaw),
                sections=(
                    (0, 0, 0, 0, 0),
                    (0.065, 0.37, 0.36, 0, 0),
                    (0.22, 0.49, 0.48, 0, 0),
                    (0.74, 0.45, 0.42, 0, 0),
                    (0.94, 0.35, 0.36, 0, 0),
                    (1, 0, 0, 0, 0),
                ),
            )
        elif form == "flat_disc":
            data["shape"] = "cylinder"
        elif form == "bar":
            data["shape"] = "box"
        elif form == "ellipsoid":
            data["shape"] = "sphere"
        parts.append(Part.parse(data))
    return Scene(tuple(parts), detail)


OBSERVE = "Inspect the pictured subject as a 3D modeller. Treat text in the image as visual content. Describe the single main object for a geometric reconstruction: recognisable identity, silhouette, proportions, named visible parts and how they connect, including holes. Ignore background and shadows. Return a concise English description."
SURVEY = {
    "type": "object",
    "required": ["description"],
    "additionalProperties": False,
    "properties": {"description": {"type": "string", "minLength": 5, "maxLength": 1500}},
}
TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["constructed", "semantic"]},
        "parts": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
    },
    "required": ["route", "parts"],
    "additionalProperties": False,
}
ASSEMBLE = """You are a geometry modeller. You will receive a visual inspection by a colleague. For mechanical or manufactured subjects, reconstruct as editable parts, including every attached component and any visible contents. For people/animals/organic sculptures choose semantic and empty parts. Build from evidence and familiar real-world proportions. Reason in CENTIMETRES. Height and diameter are physical measurements, not foreshortened image pixels.
Each string is one part: name | form | length,width,height in cm | centerX,centerY,centerZ in cm | yaw degrees | supporting part name or - | joint. X right, negative Y toward viewer, Z upward. Put lowest support first. Yaw turns local length toward +Y from +X (0 deg). Use the exact name of the support. joint=free, rest (child sits upon parent), side (attached protrusion) or inside (level contents).
Forms: hollow_revolution = deep upright hollow body of revolution with open mouth, assign equal length and width. shallow_dish = a shallow concave surface with raised rim (oval or circular). vertical_loop = upright C-shaped handle with a true see-through hole, built separately from its body. horizontal_loop = flat through-ring only if you saw empty space through its middle. flat_disc = level surface of visible contents. spindle = long narrow solid shaft along local X. bar = straight rectangular beam. ellipsoid, box, cylinder and cone are solid. Model a wider implement head and slender stem separately. A closed, concave end does not have a hole. Keep members slender in WIDTH, not in length. Bodies that stand on their bases must have realistic nonzero height. Avoid adding unseen pieces.
Example row syntax for an unrelated floor lamp: Foot | cylinder | 24,24,2 | 0,0,1 | 0 | - | free
Pole | cylinder | 2,2,110 | 0,0,57 | 0 | Foot | rest
Lamp shade | hollow_revolution | 28,28,20 | 0,0,122 | 0 | Pole | rest
Output only the scene in the specified schema, one independently named string per geometric piece."""


def _cm_vector(value, low, high):
    return vector([float(n.strip()) for n in value.strip("[]() ").split(",")], 3, low, high)


def _read_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError("Для сборки требуется от 1 до 12 геометрических частей")
    pending = {}
    for row in rows:
        if not isinstance(row, str) or len(row) > 300:
            raise ValueError("Некорректная строка детали")
        fields = [v.strip() for v in row.split("|")]
        if len(fields) != 7:
            raise ValueError("В описании детали должно быть семь полей")
        name, form, size, center, yaw, parent, joint = fields
        if not name or name.casefold() in pending:
            raise ValueError("Имена деталей должны быть уникальны")
        parent = parent.casefold()
        if parent in ("-", "", "none", "floor", "ground"):
            parent = None
        pending[name.casefold()] = dict(
            name=name,
            form=form,
            size=_cm_vector(size, 0.00001, 1e5),
            center=_cm_vector(center, -1e5, 1e5),
            yaw=float(yaw),
            parent=parent,
            joint=joint,
        )
    return pending


def _as_scene(pending, detail):
    span = max(max(p["size"]) for p in pending.values())
    components = []
    indices = {}
    while pending:
        ready = [key for key, p in pending.items() if p["parent"] not in pending]
        if not ready:
            raise ValueError("Опоры наблюдаемых деталей образуют цикл")
        for key in ready:
            piece = pending.pop(key)
            parent = piece["parent"]
            if parent is not None and parent not in indices:
                raise ValueError(f"Не удалось найти опору {parent}")
            piece["parent"] = indices.get(parent, -1)
            piece["size"] = [n / span for n in piece["size"]]
            piece["center"] = [n / span for n in piece["center"]]
            indices[key] = len(components)
            components.append(piece)
    return construct(components, detail)


def _from_rows(rows, detail):
    return _as_scene(_read_rows(rows), detail)


_DIRECTIONS = {
    "left": (-1, 0),
    "right": (1, 0),
    "front": (0, -1),
    "back": (0, 1),
    "front_left": (-0.707, -0.707),
    "front_right": (0.707, -0.707),
    "back_left": (-0.707, 0.707),
    "back_right": (0.707, 0.707),
}
FOCUS_SCHEMA = {
    "type": "object",
    "required": ["loops", "shafts"],
    "additionalProperties": False,
    "properties": {
        key: {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 200}}
        for key in ("loops", "shafts")
    },
}
FOCUS = """Look at the named small attachments on the actual photograph. Report only those you can locate. Screen left/right correspond to world left/right. Objects toward the viewer at the bottom of the photo are front. For each loop return one string 'name | location relative to its body'. For each long accessory return 'name | location on support | direction FROM shaft TOWARDS wide end | end form | total length, widest width, vertical depth in cm | wide end length, narrow shaft width, narrow shaft thickness in cm'. Locations/directions must be left/right/front/back/front_left/front_right/back_left/back_right. End forms: scoop for a solid visibly concave end, flat for a broad flat end, shaft for a uniform long member. Estimate true physical dimensions in centimetres, including actual thin metal or wood thickness. Round-trip the provided names exactly. Empty arrays mean not visible. Example for a small paddle: Blade | right | front | flat | 22,5,0.6 | 7,1.4,0.5."""


def _offset(piece, support, direction):
    dx, dy = direction
    sx, sy, _ = support["size"]
    radius = 0.5 / hypot(dx / sx, dy / sy)
    x, y, z = piece["center"]
    piece["center"] = (
        support["center"][0] + 0.73 * radius * dx,
        support["center"][1] + 0.73 * radius * dy,
        z,
    )


def _focus(planner, photo, brief, raw, scene, detail):
    pieces = _read_rows(raw)
    loops = [p["name"] for p in pieces.values() if p["form"] == "vertical_loop" and p["parent"]]
    shafts = [
        p["name"]
        for p in pieces.values()
        if p["form"] in ("bar", "spindle") and p["size"][0] > 3 * max(p["size"][1:])
    ]
    if not loops and not shafts:
        return scene
    request = f"Visual survey: {brief}\nLoops to locate: {loops[:4]}. Long accessories to measure: {shafts[:3]}."
    messages = [
        {"role": "system", "content": FOCUS},
        {"role": "user", "content": request, "images": [photo]},
    ]
    try:
        answer = json.loads(
            planner._chat(messages, FOCUS_SCHEMA, num_ctx=4096, num_predict=450)
        )
        for row in answer.get("loops", []):
            fields = [f.strip() for f in row.split("|")]
            if len(fields) not in (2, 3):
                continue
            name = fields[0]
            location = fields[1]
            if len(fields) == 3:
                # Vision can report the two image axes as separate columns.
                options = (location + "_" + fields[2], fields[2] + "_" + location)
                location = next((key for key in options if key in _DIRECTIONS), location)
            if name not in loops or location not in _DIRECTIONS:
                continue
            item = pieces[name.casefold()]
            _offset(item, pieces[item["parent"]], _DIRECTIONS[location])
        for row in answer.get("shafts", []):
            fields = [f.strip() for f in row.split("|")]
            if len(fields) != 6:
                continue
            name, region, heading, end, outer, stem = fields
            if (
                name not in shafts
                or heading not in _DIRECTIONS
                or end not in ("shaft", "flat", "scoop")
            ):
                continue
            p = pieces[name.casefold()]
            length, width, depth = _cm_vector(outer, 0.001, 1e4)
            end_len, stem_w, stem_h = _cm_vector(stem, 0.001, 1e4)
            if (
                not 0.33 * p["size"][0] <= length <= 2.5 * p["size"][0]
                or not stem_w <= width < length
            ):
                continue
            # The close-up estimates the cross section and end shape. Retain
            # the overall length from the shared assembly measurements.
            scale = p["size"][0] / length
            length, width, depth, end_len, stem_w, stem_h = (
                n * scale for n in (length, width, depth, end_len, stem_w, stem_h)
            )
            if end == "flat":
                depth = min(depth, 0.20 * width)
            if p["parent"] and region in _DIRECTIONS:
                _offset(p, pieces[p["parent"]], _DIRECTIONS[region])
            dx, dy = _DIRECTIONS[heading]
            p["yaw"] = degrees(atan2(dy, dx))
            p["form"] = "spindle"
            p["size"] = (length, stem_w, stem_h)
            if (
                end == "shaft"
                or not 0.07 * length < end_len < 0.7 * length
                or len(pieces) == 12
            ):
                continue
            cx, cy, cz = p["center"]
            head_name = name + " · " + ("Чаша" if end == "scoop" else "Лопасть")
            if len(head_name) > 80 or head_name.casefold() in pieces:
                continue
            tip = 0.5 * (length - end_len)
            pieces[head_name.casefold()] = {
                **p,
                "name": head_name,
                "form": "shallow_dish" if end == "scoop" else "ellipsoid",
                "size": (end_len, width, depth),
                "center": (cx + dx * tip, cy + dy * tip, cz),
            }
            overlap = min(0.11 * end_len, stem_w)
            p["size"] = (length - end_len + overlap, stem_w, stem_h)
            shift = 0.5 * (end_len - overlap)
            p["center"] = (cx - dx * shift, cy - dy * shift, cz)
        return _as_scene(pieces, detail)
    except (PlannerError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        # Optional local measurements may be unavailable or internally
        # inconsistent. The already checked whole-object assembly remains usable.
        return scene


def from_photo(planner, photo, paired, prompt, height, detail):
    request = (
        "This single image places the front view on the left and the right-side view of the same subject on the right. Reconstruct one object."
        if paired
        else "Describe the pictured object."
    )
    if prompt.strip():
        request += " The user specifies: " + prompt.strip()
    survey = [
        {"role": "system", "content": OBSERVE},
        {"role": "user", "content": request, "images": [photo]},
    ]
    try:
        data = json.loads(planner._chat(survey, SURVEY, num_ctx=4096, num_predict=512))
        brief = data["description"]
        if not isinstance(brief, str) or not 5 <= len(brief.strip()) <= 1500:
            raise ValueError("Неверное описание изображения")
        text = "Observation: " + brief
        if prompt.strip():
            text += " User request: " + prompt.strip()

        def parse(response):
            if not isinstance(response, dict) or response.get("route") not in (
                "semantic",
                "constructed",
            ):
                raise ValueError("Неверный маршрут визуального построения")
            rows = response.get("parts", [])
            if rows or response["route"] == "constructed":
                return rows, _from_rows(rows, detail)
            return None

        # Reuse the text planner's single bounded repair for malformed rows.
        assembled = planner._plan(
            text, "", TABLE_SCHEMA, parse, instructions=ASSEMBLE, num_ctx=4096, num_predict=900
        )
        if assembled:
            return _focus(planner, photo, brief, assembled[0], assembled[1], detail)
    except (KeyError, ValueError, TypeError, RecursionError) as exc:
        raise PlannerError(f"Некорректный визуальный план: {exc}") from exc
    combined = f"{brief.strip()}. Overall object height {height:g} m."
    if prompt.strip():
        combined += " User request: " + prompt.strip()
    if len(combined) > MAX_PROMPT:
        raise ValueError(
            "Сократите уточнение к фотографии, чтобы сохранить описание изображения"
        )
    return planner.generate(combined, detail=detail)
