import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from xtrifid.geometry import build, triangles
from xtrifid.model import Document, Part, Scene
from xtrifid.storage import export_obj, load_project, save_project


def spec(shape="box", **kw):
    return dict(
        name="Основание",
        shape=shape,
        size=[2, 3, 4],
        center=[0, 0, 2],
        rotation=[0, 0, 0],
        **kw,
    )


class MeshTests(unittest.TestCase):
    def test_closed_outward_mesh_for_every_shape_and_profile(self):
        shapes = [spec(s) for s in ("box", "cylinder", "cone", "sphere", "torus")]
        shapes.append(
            spec("extrusion", outline=[[0, 0], [1, 0], [1, 0.3], [0.3, 0.3], [0.3, 1], [0, 1]])
        )
        shapes.append(spec("lathe", outline=[[0.5, 0], [1, 0], [1, 1], [0.5, 1]]))
        shapes.append(spec("lathe", outline=[[0, 0], [0.7, 0], [1, 0.3], [0.6, 1], [0, 1]]))
        for shape in shapes:
            for mode in ("game", "render"):
                with self.subTest(shape=shape["shape"], mode=mode):
                    mesh = build(Part.parse(shape), mode)
                    v = np.array(mesh.vertices)
                    edges = Counter(
                        (a, b) for f in mesh.faces for a, b in zip(f, f[1:] + f[:1])
                    )
                    self.assertTrue(
                        all(n == 1 and edges[b, a] == 1 for (a, b), n in edges.items())
                    )
                    self.assertEqual(len(set(map(tuple, v))), len(v))
                    t = v[list(triangles(mesh.faces))]
                    area = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                    self.assertGreater(np.linalg.norm(area, axis=1).min(), 1e-10)
                    volume = (t[:, 0] * area).sum() / 6
                    self.assertGreater(volume, 0)
                    if mode == "game" or shape["shape"] == "cone":
                        self.assertEqual({len(f) for f in mesh.faces}, {3})
                    else:
                        self.assertIn(4, {len(f) for f in mesh.faces})

    def test_hollow_lathe_has_open_bore_and_correct_cap_area(self):
        p = Part.parse(spec("lathe", outline=[[0.5, 0], [1, 0], [1, 1], [0.5, 1]]))
        mesh = build(p, "render")
        self.assertTrue(all(len(f) == 4 for f in mesh.faces))
        v = mesh.vertices
        self.assertTrue(all((x / 1) ** 2 + (y / 1.5) ** 2 >= 0.25 - 1e-8 for x, y, z in v))
        np.testing.assert_allclose(np.ptp(v, axis=0), p.size)
        with self.assertRaises(ValueError):
            Part.parse(spec("lathe", outline=[[0, 0.5], [1, 0], [1, 1]]))

    def test_silhouette_and_curved_density(self):
        for shape in ("box", "sphere", "cylinder", "torus"):
            p = Part.parse(spec(shape))
            game, render = [build(p, mode) for mode in ("game", "render")]
            np.testing.assert_allclose(np.ptp(game.vertices, axis=0), p.size, atol=1e-8)
            if shape == "box":
                self.assertEqual((len(game.faces), len(render.faces)), (12, 6))
            else:
                self.assertGreater(len(render.vertices), len(game.vertices))
        p = Part.parse(spec() | dict(rotation=[90, 0, 0]))
        np.testing.assert_allclose(np.ptp(build(p, "game").vertices, axis=0), [2, 4, 3])

    def test_concave_cap_keeps_cutout(self):
        outline = [[0, 0], [0, 1], [0.3, 1], [0.3, 0.3], [1, 0.3], [1, 0]]
        mesh = build(Part.parse(spec("extrusion", outline=outline)), "game")
        t = np.array(mesh.vertices)[list(triangles(mesh.faces))]
        area = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        self.assertAlmostEqual((t[:, 0] * area).sum() / 6, 2 * 3 * 4 * 0.51)

    def test_invalid_parts_and_response_limits(self):
        for bad in (
            dict(size=[1, 0, 1]),
            dict(center=[False, 0, 0]),
            dict(rotation=[0, float("nan"), 0]),
            dict(shape="unknown"),
            dict(name="foo\nv 1 2 3"),
            dict(size=[1] * 4),
            dict(shape="extrusion", outline=[[0, 0], [1, 1], [0, 1], [1, 0]]),
            dict(shape="extrusion", outline=[[0, 0], [1, 0], [1, 0]]),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Part.parse(spec() | bad)
        for parts in ([], [spec()] * 49, None):
            with self.assertRaises(ValueError):
                Scene.parse({"parts": parts})

    def test_local_edit_history_and_disk_round_trip(self):
        scene = Scene.parse({"parts": [spec(), spec("cylinder") | dict(name="Ножка")]})
        doc = Document(scene, "game")
        other = scene.parts[1]
        doc.replace(0, Part.parse(spec() | dict(size=[1, 1, 1])))
        self.assertIs(doc.scene.parts[1], other)
        self.assertNotEqual(doc.scene, scene)
        doc.undo()
        self.assertEqual(doc.scene, scene)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "Стол.xtri"
            save_project(project, doc.scene, "render")
            self.assertEqual(load_project(project), (scene, "render"))
            project.write_text(json.dumps({"version": 3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_project(project)
            target = Path(tmp) / "Стол.obj"
            export_obj(target, scene, "render")
            lines = target.read_text().splitlines()
            self.assertEqual(len([line for line in lines if line.startswith("o ")]), 2)
            n = 0
            for line in lines:
                if line.startswith("v "):
                    n += 1
                if line.startswith("f "):
                    self.assertTrue(all(1 <= int(i) <= n for i in line.split()[1:]))
            self.assertTrue(
                any(line.startswith("f ") and len(line.split()) == 5 for line in lines)
            )
