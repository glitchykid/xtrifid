"""Build a dimensioned, dressed figure from a small semantic description."""

from math import atan2, degrees, hypot

import numpy as np

from .geometry import rotation_matrix
from .model import DETAILS, Part, Scene

BUILDS = {"slim": 0.84, "average": 1.0, "broad": 1.19}
POSES = ("standing", "walking", "arms_out")
HEMS = {"short": 0.56, "knee": 0.285, "ankle": 0.075}
MIN_HEIGHT, MAX_HEIGHT = 0.3, 3.0
CHARACTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "height_m": {"type": "number", "minimum": MIN_HEIGHT, "maximum": MAX_HEIGHT},
        "build": {"type": "string", "enum": list(BUILDS)},
        "pose": {"type": "string", "enum": list(POSES)},
        "cloak": {"type": "string", "enum": ["none", *HEMS]},
        "hood": {"type": "boolean"},
    },
    "required": ["height_m", "build", "pose", "cloak", "hood"],
}

# Cross-sections span the limb from distal to proximal. Radii and offsets use
# fractions of standing height, so dimensions stay proportional at any scale.
# Each row contains height along the segment, X/Y radii and X/Y offsets.
TORSO = (
    (0, 0.065, 0.052, 0, 0.004),
    (0.065, 0.089, 0.066, 0, 0.007),
    (0.18, 0.103, 0.071, 0, 0.01),
    (0.29, 0.103, 0.065, 0, 0.007),
    (0.40, 0.084, 0.057, 0, 0.003),
    (0.55, 0.091, 0.067, 0, 0),
    (0.70, 0.111, 0.081, 0, -0.004),
    (0.79, 0.123, 0.08, 0, -0.003),
    (0.865, 0.126, 0.067, 0, 0),
    (0.91, 0.117, 0.061, 0, 0.001),
    (0.956, 0.08, 0.049, 0, 0.002),
    (1, 0.039, 0.039, 0, 0.004),
)
# The clothed torso includes the tunic hem, avoiding a hidden duplicate torso mesh.
TUNIC = (
    (0, 0.123, 0.097, 0, 0.001),
    (0.019, 0.124, 0.098, 0, 0.001),
    (0.036, 0.118, 0.094, 0, 0.001),
    (0.16, 0.111, 0.078, 0, 0.005),
    (0.29, 0.105, 0.067, 0, 0.007),
    *TORSO[4:],
)
BELT = (
    (0, 0.092, 0.062, 0, 0.004),
    (0.15, 0.094, 0.064, 0, 0.004),
    (0.85, 0.090, 0.063, 0, 0.003),
    (1, 0.088, 0.061, 0, 0.003),
)
ARM = (
    (0, 0.016, 0.022, 0, 0),
    (0.07, 0.017, 0.023, 0, 0),
    (0.27, 0.027, 0.031, 0, -0.004),
    (0.37, 0.030, 0.032, 0, -0.003),
    (0.46, 0.022, 0.026, 0, 0.001),
    (0.51, 0.023, 0.027, 0, 0.004),
    (0.65, 0.030, 0.038, 0, 0),
    (0.77, 0.035, 0.038, 0, -0.002),
    (0.88, 0.039, 0.038, 0, 0),
    (0.95, 0.032, 0.031, 0, 0),
    (1, 0.014, 0.021, 0, 0),
)
LEG = (
    (0, 0.021, 0.027, 0, 0),
    (0.12, 0.022, 0.026, 0, 0),
    (0.25, 0.034, 0.039, 0, 0.004),
    (0.33, 0.039, 0.042, 0, 0.008),
    (0.43, 0.033, 0.036, 0, 0.004),
    (0.51, 0.031, 0.037, 0, -0.004),
    (0.56, 0.035, 0.038, 0, -0.005),
    (0.68, 0.044, 0.046, 0, -0.003),
    (0.84, 0.055, 0.059, 0, 0),
    (0.93, 0.055, 0.060, 0, 0.007),
    (1, 0.042, 0.045, 0, 0.008),
)
BOOT = (
    (0, 0.031, 0.075, 0, -0.025),
    (0.12, 0.033, 0.077, 0, -0.026),
    (0.25, 0.033, 0.072, 0, -0.022),
    (0.43, 0.029, 0.054, 0, -0.011),
    (0.63, 0.024, 0.033, 0, 0.001),
    (0.90, 0.024, 0.031, 0, 0.001),
    (1, 0.027, 0.032, 0, 0.001),
)
PALM = (
    (0, 0.011, 0.022, 0, 0),
    (0.17, 0.014, 0.027, 0, 0),
    (0.49, 0.016, 0.028, 0, 0),
    (0.75, 0.016, 0.025, 0, 0),
    (1, 0.014, 0.019, 0, 0),
)
DIGIT = (
    (0, 0, 0, 0, 0),
    (0.12, 0.0038, 0.0037, 0, 0),
    (0.33, 0.0048, 0.0045, 0, 0),
    (0.56, 0.0046, 0.0047, 0, 0),
    (0.80, 0.0052, 0.0050, 0, 0),
    (1, 0.0056, 0.0052, 0, 0),
)


def _frame(start, end):
    dx, dy, dz = np.asarray(end) - start
    rotation = (degrees(atan2(-dy, hypot(dx, dz))), degrees(atan2(dx, dz)), 0)
    return rotation, rotation_matrix(rotation)


def _limb(name, rows, start, end, stature, girth=1.0):
    start = np.asarray(start, dtype=float)
    rotation, frame = _frame(start, end)
    z, rx, ry, cx, cy = np.asarray(rows, dtype=float).T
    rx, ry, cx, cy = (a * girth for a in (rx, ry, cx, cy))
    xmin, xmax = min(cx - rx), max(cx + rx)
    ymin, ymax = min(cy - ry), max(cy + ry)
    width, depth = xmax - xmin, ymax - ymin
    mx, my = (xmax + xmin) / 2, (ymax + ymin) / 2
    length = float(np.linalg.norm(np.asarray(end) - start))
    center = start + frame @ np.array((mx, my, length / 2))
    sections = np.column_stack(
        (z, rx / width, ry / depth, (cx - mx) / width, (cy - my) / depth)
    )
    return Part.parse(
        {
            "name": name,
            "shape": "loft",
            "size": (np.array([width, depth, length]) * stature).tolist(),
            "center": (center * stature).tolist(),
            "rotation": rotation,
            "sections": sections.tolist(),
        }
    )


def _arm(side, label, pose, stature, girth):
    shoulder = np.array((side * 0.122 * girth, 0, 0.827))
    if pose == "arms_out":
        shoulder = np.array((side * 0.102 * girth, 0, 0.818))
        wrist = (side * (0.122 * girth + 0.341), -0.009, 0.815)
    elif pose == "walking":
        wrist = (side * 0.157 * girth, side * 0.094 - 0.014, 0.473)
    else:
        wrist = (side * 0.163 * girth, -0.024, 0.467)
    wrist = np.asarray(wrist)
    _, frame = _frame(wrist, shoulder)

    def hand(x, y, z):
        return wrist + frame @ np.array([x * girth, y * girth, z])

    parts = [
        _limb("Рука " + label, ARM, wrist, shoulder, stature, girth),
        _limb("Кисть " + label, PALM, hand(0, 0, -0.052), wrist, stature, girth),
    ]
    for digit, y, length, radius in (
        ("Указательный", -0.018, 0.039, 0.94),
        ("Средний", -0.006, 0.044, 1.0),
        ("Безымянный", 0.006, 0.040, 0.95),
        ("Мизинец", 0.018, 0.032, 0.82),
    ):
        base = hand(0, y, -0.050)
        tip = hand(-side * 0.006, y * 1.12, -0.050 - length)
        parts.append(
            _limb(f"{digit} · {label} кисть", DIGIT, tip, base, stature, girth * radius)
        )
    parts.append(
        _limb(
            f"Большой палец · {label} кисть",
            DIGIT,
            hand(-side * 0.007, -0.036, -0.054),
            hand(0, -0.015, -0.013),
            stature,
            girth * 1.25,
        )
    )
    return parts


def build_character(spec, detail="medium"):
    if not isinstance(spec, dict) or not spec.keys() <= CHARACTER_SCHEMA["properties"].keys():
        raise ValueError("Неверные параметры персонажа")
    stature = spec.get("height_m", 1.75)
    build = spec.get("build", "average")
    pose = spec.get("pose", "standing")
    cloak = spec.get("cloak", "none")
    hood = spec.get("hood", False)
    if type(stature) not in (int, float) or not MIN_HEIGHT <= stature <= MAX_HEIGHT:
        raise ValueError("Рост персонажа должен составлять от 0,3 до 3 м")
    if build not in tuple(BUILDS) or pose not in POSES or cloak not in ("none", *HEMS):
        raise ValueError("Неизвестное телосложение, поза или длина плаща")
    if type(hood) is not bool or (hood and cloak == "none"):
        raise ValueError("Капюшону нужен плащ")
    if detail not in DETAILS:
        raise ValueError("Неизвестный уровень детализации")
    girth = BUILDS[build]

    def part(name, shape, size, center, **parameters):
        return Part.parse(
            {
                "name": name,
                "shape": shape,
                "size": [s * stature for s in size],
                "center": [c * stature for c in center],
                "rotation": [0, 0, 0],
                **parameters,
            }
        )

    clothed = cloak != "none"
    parts = [
        _limb(
            "Туловище",
            TUNIC if clothed else TORSO,
            (0, 0, 0.431 if clothed else 0.454),
            (0, 0, 0.865),
            stature,
            girth,
        ),
        _limb(
            "Шея",
            ((0, 0.048, 0.043, 0, 0), (0.33, 0.033, 0.036, 0, 0), (1, 0.035, 0.035, 0, 0)),
            (0, 0.002, 0.835),
            (0, -0.002, 0.902),
            stature,
        ),
    ]
    head_top = 0.975 if hood else 1.0
    parts.append(
        part(
            "Голова",
            "head",
            (0.113, 0.132, head_top - 0.856),
            (0, -0.006, (head_top + 0.856) / 2),
        )
    )
    for side, label, boot_name, ear_name in (
        (1, "левая", "Ботинок левый", "Ухо левое"),
        (-1, "правая", "Ботинок правый", "Ухо правое"),
    ):
        ankle = (side * 0.055 * girth, -side * 0.098 if pose == "walking" else -0.002, 0.081)
        hip = (side * 0.054 * girth, 0.004, 0.534)
        parts.append(_limb("Нога " + label, LEG, ankle, hip, stature, girth))
        parts.append(
            _limb(boot_name, BOOT, (*ankle[:2], 0), (*ankle[:2], 0.120), stature, girth)
        )
        parts.extend(_arm(side, label, pose, stature, girth))
        parts.append(
            part(
                ear_name,
                "sphere",
                (0.014, 0.028, 0.039),
                (side * 0.055, 0.001, head_top - 0.068),
            )
        )
    if clothed:
        hem = HEMS[cloak]
        thrown_back = pose == "arms_out"
        # Bounds of the pattern in stature units, with the shoulder centred on the body.
        width, depth, center_y = (0.540, 0.336, 0.114) if thrown_back else (0.547, 0.389, 0.086)
        parts.append(_limb("Пояс", BELT, (0, 0, 0.594), (0, 0, 0.618), stature, girth))
        parts.append(
            part(
                "Плащ",
                "cloak",
                (width * girth, depth * girth, 0.86 - hem),
                (0, center_y * girth, (0.86 + hem) / 2),
                cloak_back=thrown_back,
            )
        )
        parts.append(
            part(
                "Застёжка плаща",
                "sphere",
                (0.045 * girth, 0.016 * girth, 0.020),
                (
                    0,
                    (-0.048 if thrown_back else -0.062) * girth,
                    0.858 if thrown_back else 0.854,
                ),
            )
        )
    if hood:
        parts.append(part("Капюшон", "hood", (0.164, 0.174, 0.182), (0, 0.003, 0.909)))
    return Scene(tuple(parts), detail)
