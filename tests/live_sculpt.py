"""Run the Russian branching-object acceptance case with the real local model."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from xtrifid.geometry import build, check_scene_budget, triangle_bound
from xtrifid.planner import Planner
from xtrifid.storage import load_project, save_project

PROMPT = (
    "Сделай ветвистый коралл высотой сорок сантиметров. От одного толстого ствола "
    "расходятся три тонкие ветви влево, вправо и назад."
)


def check(scene):
    (part,) = scene.parts
    assert part.shape == "sculpt" and abs(part.size[2] - 0.4) < 1e-6
    children = Counter(node[1] for node in part.nodes)
    tips = [n for i, n in enumerate(part.nodes) if not children[i]]
    root = next(n for n in part.nodes if n[1] < 0)
    assert len(tips) == 3 and all(n[5] < root[5] for n in tips)
    assert min(n[2] for n in tips) < root[2] < max(n[2] for n in tips)
    assert max(n[3] for n in tips) > root[3]
    for mode in ("game", "render"):
        check_scene_budget(scene, mode)
        mesh = build(part, mode, scene.detail)
        np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), part.size)
        assert abs(mesh.vertices[:, 2].min()) < 1e-8
        directed = Counter((a, b) for f in mesh.faces for a, b in zip(f, (*f[1:], f[0])))
        assert all(count == 1 and directed[b, a] == 1 for (a, b), count in directed.items())
        assert len(mesh.vertices) - len(directed) // 2 + len(mesh.faces) == 2
        assert triangle_bound(part, mode, scene.detail) == sum(len(f) - 2 for f in mesh.faces)
        if mode == "game":
            assert all(len(f) == 3 for f in mesh.faces)
        print(mode, len(mesh.vertices), "vertices", len(mesh.faces), "faces", flush=True)


if __name__ == "__main__":
    scene = Planner().generate(PROMPT, "medium")
    check(scene)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        save_project(path, scene, "render")
        assert load_project(path) == (scene, "render")
