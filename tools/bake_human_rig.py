"""Bake MakeHuman v1.3.0 hm08 CC0 skin weights into an LOD-independent volume.

Run in the development image with base.obj and default_weights.mhw from
https://github.com/makehumancommunity/makehuman/tree/v1.3.0/makehuman/data .
The volume interpolates four nearest skin vertices. The stored channels are
upper arm, forearm/hand, thigh and shin/foot, mirrored at the sagittal plane.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_skin(obj):
    points, skin, group = [], set(), ""
    for line in obj.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            points.append(tuple(map(float, fields[1:4])))
        elif fields[0] == "g":
            group = fields[1]
        elif fields[0] == "f" and group == "body":
            skin.update(int(v.split("/")[0]) - 1 for v in fields[1:])
    return np.asarray(points), np.asarray(sorted(skin))


def bake(obj, rig, output):
    raw, skin = load_skin(obj)
    positions = np.column_stack((raw[:, 0], 1.09965 - raw[:, 2], raw[:, 1] + 8.1676)) / 16.6589
    channels = (
        ("upperarm",),
        ("lowerarm", "finger", "metacarpal", "wrist"),
        ("upperleg",),
        ("lowerleg", "foot", "toe"),
    )
    weights = np.zeros((len(raw), len(channels)))
    for name, pairs in json.loads(rig.read_text())["weights"].items():
        for channel, prefixes in enumerate(channels):
            if name.endswith(".L") and name.startswith(prefixes):
                indices, values = np.asarray(pairs).T
                weights[indices.astype(int), channel] += values
    skin = skin[(positions[skin, 0] >= 0) & (positions[skin, 2] < 0.89)]
    points = positions[skin].astype(np.float32)
    weights = weights[skin]
    shape = (33, 29, 90)
    grid = np.moveaxis(np.indices(shape), 0, -1).reshape(-1, 3).astype(np.float32)
    grid = grid * 0.01 + np.array((0, -0.14, 0), dtype=np.float32)
    samples = np.empty((len(grid), 4), dtype=np.uint8)
    lengths = (points * points).sum(axis=1)
    for start in range(0, len(grid), 128):
        block = grid[start : start + 128]
        distance = lengths + (block * block).sum(axis=1)[:, None] - 2 * block @ points.T
        idx = np.argpartition(distance, 3, axis=1)[:, :4]
        proximity = 1 / np.maximum(1e-10, np.take_along_axis(distance, idx, axis=1))
        blend = proximity / proximity.sum(axis=1)[:, None]
        samples[start : start + len(block)] = np.rint(
            255 * (weights[idx] * blend[:, :, None]).sum(1)
        )
    field = samples.reshape((*shape, 4))
    field[:, :, -1, :] = 0
    np.savez_compressed(output, weights=field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("obj", "rig", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    bake(args.obj, args.rig, args.output)
