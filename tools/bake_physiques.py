"""Bake the adult MakeHuman v1.3.0 male/female macro targets onto the shipped LODs."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.bake_human import eyes, read_body, subdivide
from xtrifid.human_body import POSE_BOUNDS, deform, pose_heights

# The three ancestry macro targets receive equal weight, as in MakeHuman's default
# setting. Its adult average-weight, average-muscle universal targets contain zero deltas.
SOURCES = {
    "male": (
        ("african", "894abc1fbb3d28543a51fef16f89d5d4bdf9aa2e1534413339811a3d47818b7d"),
        ("asian", "ed2e8c191cb6b87b4a2d97c80486acb2604fa549316c7aa2a738a5d5a14334dc"),
        ("caucasian", "70e228ba7164737dae664454394536fc5935fa48d333c1a97d77e2dc6eacc5f5"),
    ),
    "female": (
        ("african", "92d61eeb3c164b421fd5a7c3537ee45e7e1a51de4d49bf312e19c3df2be1d8fc"),
        ("asian", "095fe79694fa19e1fe98d93009ec116199bd524e081c640351a10eccf2cca1eb"),
        ("caucasian", "118379f6e8ba9266247fdb8788a20e1df40a239f97ced0b9905bcbcc74f6e820"),
    ),
}
JOINTS = ("shoulder", "elbow", "hand", "upper-leg", "knee", "ankle")


def source_geometry(obj):
    raw, groups, group = [], defaultdict(set), ""
    for line in obj.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            raw.append(tuple(map(float, fields[1:4])))
        elif fields[0] == "g":
            group = fields[1]
        elif fields[0] == "f":
            groups[group].update(int(item.split("/")[0]) - 1 for item in fields[1:])
    return np.asarray(raw), {name: sorted(group) for name, group in groups.items()}


def quantize(v):
    units = np.rint(v * 262144)
    if np.abs(units).max() >= 32768:
        raise ValueError("Morph exceeds signed 16-bit range")
    return units.astype("<i2")


def bake(obj, targets, destination):
    body, cage = read_body(obj)
    raw, groups = source_geometry(obj)
    base = (raw[:, (0, 2, 1)] - [0, 1.09965, -8.1676]) * [1, -1, 1] / 16.6589
    ids = {tuple(v): i for i, v in enumerate(base[: len(body)].astype("<f4"))}
    extrema = {kind: {pose: {} for pose in POSE_BOUNDS} for kind in ("neutral", *SOURCES)}
    for kind, sources in SOURCES.items():
        morphed = raw.copy()
        for family, checksum in sources:
            path = targets / f"{family}-{kind}-young.target"
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Unexpected macro target revision: {path.name}")
            table = np.loadtxt(path)
            morphed[table[:, 0].astype(int)] += table[:, 1:] / len(sources)
        lo, hi = morphed[: len(body)].min(axis=0), morphed[: len(body)].max(axis=0)
        # Align stature and shoulder depth, so garments and the ground plane fit
        # in the same reference frame while preserving anatomical proportions.
        shoulder = groups["joint-l-shoulder"]
        target = (morphed[:, (0, 2, 1)] - [0, 0, lo[1]]) * [1, -1, 1] / (hi[1] - lo[1])
        target[:, 1] += base[shoulder, 1].mean() - target[shoulder, 1].mean()
        delta = target - base
        joints = np.array([target[groups["joint-l-" + name]].mean(axis=0) for name in JOINTS])
        landmark_groups = (
            ("joint-l-eye", "joint-r-eye"),
            ("joint-spine-1",),
            ("joint-l-hand", "joint-l-finger-3-1"),
            ("joint-r-hand", "joint-r-finger-3-1"),
        )
        landmarks = [
            delta[np.concatenate([groups[name] for name in names])].mean(0)
            for names in landmark_groups
        ]
        output = {"joints": joints, "landmarks": quantize(np.asarray(landmarks))}
        for n, spokes in ((8, 12), (16, 16), (24, 20), (32, 24), (48, 32), (64, 48)):
            with np.load(destination / f"human-{n}.npz") as data:
                rest = data["vertices"].astype(float)
            eye_vertices, _, _ = eyes(spokes)
            skin_count = len(rest) - len(eye_vertices)
            if n == 64:
                skin_offsets, _ = subdivide(delta[: len(body)], cage)
            else:
                skin_offsets = delta[[ids[tuple(v)] for v in rest[:skin_count]]]
            moved = [skin_offsets]
            for side, eye in zip(("r", "l"), np.array_split(rest[skin_count:], 2)):
                # Fit the two published eye-helper surfaces to carry the custom
                # iris, limbus and cornea into the corresponding morphed socket.
                helper = groups[f"helper-{side}-eye"]
                control = base[helper]
                warp = np.linalg.lstsq(
                    control - control.mean(0), delta[helper] - delta[helper].mean(0), rcond=None
                )[0]
                moved.append((eye - control.mean(0)) @ warp + delta[helper].mean(0))
            offsets = quantize(np.concatenate(moved))
            output[f"lod{n}"] = offsets
            for pose in POSE_BOUNDS:
                posed = deform(
                    rest, pose, offsets=offsets.astype(float) / 262144, joints=joints
                )
                extrema[kind][pose][n] = [posed[:, 2].min(), posed[:, 2].max()]
                if n == 48:
                    output[pose] = posed[:, 2].min()
                if kind == "male":
                    reference = deform(rest, pose)
                    extrema["neutral"][pose][n] = [reference[:, 2].min(), reference[:, 2].max()]
        np.savez_compressed(destination / f"human-{kind}.npz", **output)
    for kind, poses in extrema.items():
        for pose, levels in poses.items():
            floor = levels[48][0]
            for n, bounds in levels.items():
                actual = np.asarray(bounds)
                if kind != "neutral":
                    actual = pose_heights(actual, floor, pose)
                levels[n] = (np.array((POSE_BOUNDS[pose][0][2], 1)) - actual).tolist()
    (destination / "human-extrema.json").write_text(
        json.dumps(extrema, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    bake(*map(Path, sys.argv[1:]))
