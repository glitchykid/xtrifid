"""Pose the welded hm08 skin using its authored joint locations and weights."""

import json
from functools import cache
from pathlib import Path

import numpy as np

# MakeHuman hm08 joint helper centroids, in units of standing height.
SHOULDER = np.array((0.1007, 0.0572, 0.8052))
ELBOW = np.array((0.1878, 0.0581, 0.7000))
WRIST = np.array((0.2588, -0.0394, 0.6375))
HIP = np.array((0.0662, 0.0586, 0.5195))
KNEE = np.array((0.0949, 0.0468, 0.2685))
ANKLE = np.array((0.1318, 0.0666, 0.0432))
PALM = np.array((0.2710, -0.0650, 0.6150))
FACE = np.array((0.0, -0.0190, 0.9280))
REST_JOINTS = (SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE)
LANDMARKS = {
    "Голова": FACE,
    "Туловище": (0, 0.05, 0.68),
    "Кисть левая": PALM,
    "Кисть правая": PALM * (-1, 1, 1),
}


# Pivot and compact support of each localized facial scale in hm08 rest space.
# The envelope follows the nose, paired lips or mental eminence and fades
# before the neighbouring feature. Both sides receive the same displacement.
FACIAL_SCALES = {
    "nose": ((0, -0.016, 0.910), (0.016, 0.036, 0.017), (1.0, 1.0, 0.55)),
    "lips": ((0, -0.016, 0.887), (0.025, 0.030, 0.011), (0.85, 1.0, 0.90)),
    "chin": ((0, -0.006, 0.865), (0.033, 0.032, 0.018), (1.0, 1.0, 0.75)),
}


@cache
def _weight_field():
    with np.load(
        Path(__file__).with_name("assets") / "human-rig.npz", allow_pickle=False
    ) as data:
        return data["weights"].astype(float) / 255


def _weights(rest):
    field = _weight_field()
    grid = np.clip((rest + (0, 0.14, 0)) * 100, 0, np.array(field.shape[:3]) - 1)
    index = np.minimum(grid.astype(int), np.array(field.shape[:3]) - 2)
    t = grid - index
    # Trilinear skin-weight interpolation keeps all LODs on the same skeleton.
    weights = np.zeros((len(rest), 4))
    for corner in np.ndindex(2, 2, 2):
        blend = np.prod(np.where(corner, t, 1 - t), axis=1)
        weights += field[tuple((index + corner).T)] * blend[:, None]
    return weights / np.maximum(1, weights.sum(axis=1))[:, None]


def _unit(vec):
    return vec / np.linalg.norm(vec)


def _align(source, target):
    a, b = _unit(source), _unit(np.asarray(target))
    x, y, z = np.cross(a, b)
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return np.eye(3) + skew + (skew @ skew) / (1 + np.dot(a, b))


def _chain(points, first, joint, end, direction, terminal):
    upper = _align(joint - first, direction)
    lower = _align(upper @ (end - joint), terminal) @ upper
    posed_joint = upper @ (joint - first) + first
    return ((points - first) @ upper.T + first, (points - joint) @ lower.T + posed_joint, lower)


def _fade(low, high, value):
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def _shape(points, scales, offsets=None):
    # Work on the mirrored resting skin before joint transforms, so editing either
    # ear, palm or foot uses the same anatomical location in every pose.
    edited = points.copy()
    x, y, z = points.T
    # The authored control cage leaves the mouth slightly open. Bring the two
    # vermilion borders into a relaxed resting position while preserving a
    # positive gap. The vertical mapping is monotone, so the lips cannot cross.
    lip_rim = z - 0.88714
    mouth = (1 - _fade(0.014, 0.018, x)) * (1 - _fade(-0.018, -0.009, y))
    if offsets is not None:
        # An authored physique can change both the position and the aperture of
        # the mouth. Close the lips around the translated contact line.
        border = (mouth > 0.95) & (abs(lip_rim) < 0.0008)
        if border.any():
            lip_rim = lip_rim + offsets[:, 2] - np.median(offsets[border, 2])
    # The upper lip overlaps the lower lip in front view. A slight contact
    # slope through their thickness retains separate surfaces in cross-section.
    slope = 0.33 * np.clip(y + 0.025, -0.002, 0.002)
    tubercle = 0.00045 * np.exp(-(((x - 0.005) / 0.0034) ** 2)) - 0.00055 * np.exp(
        -((x / 0.0038) ** 2)
    )
    edited[:, 2] += (
        mouth * (slope + tubercle - 0.90 * lip_rim) * (1 - _fade(0.0008, 0.0048, abs(lip_rim)))
    )
    for name, scale in scales:
        if name == "head":
            pivot, mask, axes = (0, 0.043, 0.85), _fade(0.835, 0.875, z), (1, 1, 1)
        elif name == "ears":
            pivot = (0.043, 0.043, 0.927)
            mask = (
                _fade(0.034, 0.048, x)
                * (1 - _fade(0.014, 0.038, abs(y - 0.042)))
                * (1 - _fade(0.017, 0.038, abs(z - 0.926)))
            )
            axes = (1, 1, 1)
        elif name in FACIAL_SCALES:
            pivot, radius, axes = FACIAL_SCALES[name]
            # Ellipsoidal smooth falloff preserves the surrounding lip/eyelid loops.
            distance = np.sum(((points - pivot) / radius) ** 2, axis=1)
            mask = (1 - np.minimum(1, distance)) ** 2 * (1 - _fade(-0.005, 0.012, y))
        elif name == "hands":
            pivot, mask, axes = (
                WRIST,
                _fade(0.232, 0.257, x) * (1 - _fade(0.64, 0.68, z)),
                (1, 1, 1),
            )
        elif name == "feet":
            pivot, mask, axes = ANKLE, 1 - _fade(0.043, 0.082, z), (1, 1, 0)
        elif name == "waist":
            pivot, mask, axes = (
                (0, 0.057, 0.60),
                (1 - _fade(0.025, 0.105, abs(z - 0.603))) * (1 - _fade(0.12, 0.165, x)),
                (1, 1, 0),
            )
        else:  # shoulder breadth: move the attached arms with the shoulder girdle
            pivot, mask, axes = (
                (0, 0.057, 0.80),
                _fade(0.05, 0.11, x) * _fade(0.50, 0.58, z) * (1 - _fade(0.81, 0.86, z)),
                (1, 0, 0),
            )
        edited += (points - pivot) * axes * ((scale - 1) * mask)[:, None]
    return edited


def deform(rest, pose, scales=(), *, offsets=None, joints=REST_JOINTS):
    mirrored = np.array(rest, dtype=float)
    side = np.where(mirrored[:, 0] < 0, -1.0, 1.0)
    mirrored[:, 0] = abs(mirrored[:, 0])
    weights = _weights(mirrored)
    base = _shape(mirrored, scales, offsets)
    if offsets is not None:
        base += offsets
        base[:, 0] += offsets[:, 0] * (side - 1)
    shoulder, elbow, wrist, hip, knee, ankle = joints
    result = base.copy()
    for sign in (-1, 1):
        mask = side == sign
        if not mask.any():
            continue
        p, w = base[mask], weights[mask]
        if pose == "arms_out":
            upper_arm, forearm = (1, 0, 0.035), (1, -0.025, -0.035)
        else:
            swing = sign * 0.24 if pose == "walking" else 0
            upper_arm, forearm = (0.17, swing, -1), (0.10, swing - 0.12, -1)
        upper, lower, _ = _chain(p, shoulder, elbow, wrist, upper_arm, forearm)
        arm = (upper - p) * w[:, 0, None] + (lower - p) * w[:, 1, None]
        step = -sign * 0.25 if pose == "walking" else 0
        shin_angle = step + (0.13 if pose == "walking" and sign < 0 else 0)
        thigh, shin, shin_rotation = _chain(
            p, hip, knee, ankle, (0.015, step, -1), (0.015, shin_angle, -1)
        )
        # Keep the sole horizontal as the shin swings about the hip and knee.
        flat_foot = (p - ankle) - (p - ankle) @ shin_rotation.T
        shin += flat_foot * (1 - _fade(0.046, 0.084, mirrored[mask, 2]))[:, None]
        result[mask] += arm + (thigh - p) * w[:, 2, None] + (shin - p) * w[:, 3, None]
    result[:, 0] *= side
    return result


# Bounds of the full-detail, neutrally proportioned skin after posing. Use a
# common reference frame for all LODs, accessories and local mesh edits.
POSE_BOUNDS = {
    "standing": ((-0.14673359, -0.05674571, -0.00571546), (0.14673359, 0.12683764, 1)),
    "walking": ((-0.14556276, -0.17365241, 0.00867323), (0.14643022, 0.23174089, 1)),
    "arms_out": ((-0.47300296, -0.05674571, -0.00571546), (0.47300296, 0.12683764, 1)),
}


def pose_frame(pose):
    low, high = np.asarray(POSE_BOUNDS[pose])
    return (high - low), (high + low) / 2


def pose_heights(z, source_floor, pose):
    reference_floor = POSE_BOUNDS[pose][0][2]
    return (z - source_floor) * (1 - reference_floor) / (1 - source_floor) + reference_floor


@cache
def _extremity_offsets():
    path = Path(__file__).with_name("assets") / "human-extrema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def character_points(part, points, field):
    if part.physique == "neutral":
        return deform(points, part.pose, part.body_scales)
    path = Path(__file__).with_name("assets") / f"human-{part.physique}.npz"
    with np.load(path, allow_pickle=False) as asset:
        offsets = asset[field].astype(float) / 262144
        posed = deform(
            points, part.pose, part.body_scales, offsets=offsets, joints=asset["joints"]
        )
        source_floor = float(asset[part.pose])
    posed[:, 2] = pose_heights(posed[:, 2], source_floor, part.pose)
    return posed


def posed_mesh(part, n):
    from .human_asset import rest_mesh

    points, faces = rest_mesh(n)
    size, center = pose_frame(part.pose)
    posed = character_points(part, points, f"lod{n}")
    # A decimated sole or subdivided crown can lose millimetres. Correct only
    # their end bands in rest space so edits, attachment points and all other
    # vertices stay fixed. Baseline offsets were measured without local edits.
    floor, crown = _extremity_offsets()[part.physique][part.pose][str(n)]
    z = points[:, 2]
    posed[:, 2] += floor * (1 - _fade(0.025, 0.065, z)) + crown * _fade(0.955, 0.982, z)
    return (posed - center) / size, faces, (True,) * len(faces)
