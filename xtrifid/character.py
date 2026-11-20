"""Build a human scene and explicitly requested garments from a semantic description."""

import numpy as np

from .geometry import rotation_matrix
from .human_body import LANDMARKS, character_points, pose_frame
from .model import (
    BODY_REGIONS,
    DETAILS,
    MAX_BODY_SCALE,
    MIN_BODY_SCALE,
    PHYSIQUES,
    POSES,
    Part,
    Scene,
)

BUILDS = {"slim": 0.84, "average": 1.0, "broad": 1.19}
HEMS = {"short": 0.56, "knee": 0.285, "ankle": 0.075}
MIN_HEIGHT, MAX_HEIGHT = 0.3, 3.0
BODY_SCALES_SCHEMA = {
    "type": "object",
    "properties": dict.fromkeys(
        BODY_REGIONS, {"type": "number", "minimum": MIN_BODY_SCALE, "maximum": MAX_BODY_SCALE}
    ),
    "additionalProperties": False,
}
CHARACTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "height_m": {"type": "number", "minimum": MIN_HEIGHT, "maximum": MAX_HEIGHT},
        "build": {"type": "string", "enum": list(BUILDS)},
        "pose": {"type": "string", "enum": list(POSES)},
        "physique": {"type": "string", "enum": list(PHYSIQUES)},
        "cloak": {"type": "string", "enum": ["none", *HEMS]},
        "hood": {"type": "boolean"},
        "body_scales": BODY_SCALES_SCHEMA,
    },
    "required": ["height_m", "build", "pose", "cloak", "hood"],
}


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

    crown = 0.978 if hood else 1
    skin_height = stature * crown
    size, center = pose_frame(pose)
    # The reference Y coordinate passes through the spine. Ground Z at the
    # lowest sole for this stride and scale every axis by standing height.
    scale = np.array((girth, girth, 1)) * (skin_height / size[2])
    center = (center - (0, 0.05, center[2] - size[2] / 2)) * scale
    body = Part.parse(
        {
            "name": "Человек",
            "shape": "human",
            "size": (size * scale).tolist(),
            "center": center.tolist(),
            "rotation": (0, 0, 0),
            "pose": pose,
            "physique": spec.get("physique", "neutral"),
            "body_scales": spec.get("body_scales", {}),
        }
    )
    parts = [body]
    if cloak != "none":
        hem = HEMS[cloak] * crown
        collar = (0.860 if body.physique == "male" else 0.851) * crown
        thrown_back = pose == "arms_out"
        # Bounds of the pattern in stature units, with the shoulder centred on the body.
        width, depth, center_y = (0.50, 0.27, 0.090) if thrown_back else (0.46, 0.23, 0.05)
        parts.append(
            part(
                "Плащ",
                "cloak",
                (width * girth * crown, depth * girth * crown, collar - hem),
                (0, center_y * girth * crown, (collar + hem) / 2),
                cloak_back=thrown_back,
            )
        )
    if hood:
        parts.append(part("Капюшон", "hood", (0.164, 0.174, 0.182), (0, -0.012, 0.909)))
    return Scene(tuple(parts), detail)


def with_character_pose(part, pose):
    """Preserve stature, girth and the ground anchor when changing a posed skin."""
    if part.shape != "human" or pose not in POSES:
        raise ValueError("Неизвестная поза человека")
    old_size, old_mid = pose_frame(part.pose)
    new_size, new_mid = pose_frame(pose)
    scale = np.asarray(part.size) / old_size
    new_scale = scale * (old_size[2] / new_size[2])
    old_offset = (old_mid - (0, 0.05, old_mid[2] - old_size[2] / 2)) * scale
    new_offset = (new_mid - (0, 0.05, new_mid[2] - new_size[2] / 2)) * new_scale
    world_offset = rotation_matrix(part.rotation) @ (new_offset - old_offset)
    return Part.parse(
        part.data()
        | {
            "size": (new_size * new_scale).tolist(),
            "center": (np.asarray(part.center) + world_offset).tolist(),
            "pose": pose,
        }
    )


def character_anchor(scene, anchor):
    """Locate an anatomical feature after posing, local edits and object transforms."""
    body = next((p for p in scene.parts if p.shape == "human"), None)
    if body is None:
        return next(p.center for p in scene.parts if p.name == anchor)
    if anchor not in LANDMARKS:
        raise ValueError("Неизвестный ориентир фигуры")
    posed = character_points(body, list(LANDMARKS.values()), "landmarks")[
        list(LANDMARKS).index(anchor)
    ]
    size, center = pose_frame(body.pose)
    local = (posed - center) * np.asarray(body.size) / size
    return tuple(map(float, rotation_matrix(body.rotation) @ local + body.center))
