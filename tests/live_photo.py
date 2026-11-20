"""Exercise the full photo path against an installed Ollama vision model."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from xtrifid.geometry import build, check_scene_budget
from xtrifid.planner import Planner
from xtrifid.reference import photo_scene
from xtrifid.storage import export_obj, load_project, save_project


def verify(scene, height, espresso):
    check_scene_budget(scene, "game")
    meshes = [build(p, "game", scene.detail, scene.style) for p in scene.parts]
    points = np.concatenate([m.vertices for m in meshes])
    np.testing.assert_allclose([points[:, 2].min(), points[:, 2].max()], [0, height], atol=1e-6)
    for mesh in meshes:
        assert all(len(face) == 3 for face in mesh.faces)
        assert np.isfinite(mesh.vertices).all()
    if espresso:
        # The published photograph clearly shows a loop on the cup and a separate
        # saucer. A plausible text identification alone cannot satisfy this case.
        pans = [
            p for p in scene.parts if p.shape == "lathe" and p.size[2] < min(p.size[:2]) / 4
        ]
        handles = [
            (p, m)
            for p, m in zip(scene.parts, meshes)
            if p.shape == "torus" and abs(p.rotation[0]) == 90
        ]
        assert pans and handles, [(p.name, p.shape, p.size) for p in scene.parts]
        assert any(p.shape == "lathe" and p.size[2] > min(p.size[:2]) / 4 for p in scene.parts)
        for _, mesh in handles:
            edges = Counter(
                tuple(sorted((a, b))) for f in mesh.faces for a, b in zip(f, (*f[1:], f[0]))
            )
            assert all(n == 2 for n in edges.values())
            assert len(mesh.vertices) - len(edges) + len(mesh.faces) == 0
    print("Measured XYZ (m)", np.ptp(points, axis=0))
    print("Parts", [(p.name, p.shape) for p in scene.parts])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--height", type=float, default=0.12)
    parser.add_argument("--detail", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--check-espresso", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional destination for the XTRI and OBJ")
    args = parser.parse_args()
    scene = photo_scene(
        Planner(), args.front, args.right, height_m=args.height, detail=args.detail
    )
    verify(scene, args.height, args.check_espresso)
    if args.output:
        save_project(args.output, scene, "game")
        assert load_project(args.output) == (scene, "game")
        export_obj(args.output.with_suffix(".obj"), scene, "game")
