import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_model import spec

from xtrifid.geometry import Mesh, build, check_scene_budget, triangle_bound
from xtrifid.model import Part, Scene
from xtrifid.profile import profile_guide
from xtrifid.resources import MIB, Capacity
from xtrifid.storage import export_obj, load_project, save_project


class WorkshopTests(unittest.TestCase):
    def test_project_size_and_available_memory_are_checked_before_json_decode(self):
        from xtrifid.storage import MAX_PROJECT_BYTES

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.xtri"
            with path.open("wb") as f:
                f.truncate(MAX_PROJECT_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "большой"):
                load_project(path)
            with path.open("wb") as f:
                f.truncate(2 * MIB)
            with (
                patch("xtrifid.storage.capacity", return_value=Capacity(MIB, 1)),
                self.assertRaisesRegex(ValueError, "памяти"),
            ):
                load_project(path)

    def test_low_poly_is_a_style_at_each_density_and_exports_flat_geometry(self):
        part = Part.parse(spec("sphere"))
        for mode in ("game", "render"):
            levels = []
            for detail in ("low", "medium", "high"):
                regular = build(part, mode, detail)
                faceted = build(part, mode, detail, "low_poly")
                levels.append(len(faceted.faces))
                self.assertLess(len(faceted.vertices), len(regular.vertices))
                self.assertFalse(any(faceted.smooth))
                np.testing.assert_allclose(np.ptp(faceted.vertices, axis=0), part.size)
                self.assertEqual(
                    triangle_bound(part, mode, detail, "low_poly"),
                    sum(len(f) - 2 for f in faceted.faces),
                )
            self.assertTrue(levels[0] < levels[1] < levels[2])
        with tempfile.TemporaryDirectory() as tmp:
            scene = Scene((part,), "high", "low_poly")
            path = Path(tmp) / "Facets.xtri"
            save_project(path, scene, "render")
            self.assertEqual(load_project(path), (scene, "render"))
            export_obj(path.with_suffix(".obj"), scene, "render")
            obj = path.with_suffix(".obj").read_text()
            self.assertNotIn("\ns 1\n", obj)
            self.assertEqual(
                obj.count("\nf "), len(build(part, "render", "high", "low_poly").faces)
            )
        for bad in ("", "low", None):
            with self.assertRaises(ValueError):
                Scene.parse(scene.data() | {"style": bad})

    def test_style_budget_and_point_guide_use_the_same_tessellation(self):
        p = Part.parse(spec("cylinder"))
        s = Scene((p,), "high", "low_poly")
        triangles = triangle_bound(p, "game", s.detail, s.style)
        with patch("xtrifid.geometry.mesh_triangle_budget", return_value=triangles):
            check_scene_budget(s, "game")
            with self.assertRaises(ValueError):
                check_scene_budget(replace(s, style="standard"), "game")
        guide = profile_guide(p, 1, "game", s.detail, s.style)
        mesh = build(p, "game", s.detail, s.style)
        self.assertEqual(
            len(guide) - 1, sum(v[2] == mesh.vertices[:, 2].min() for v in mesh.vertices) - 1
        )

    def test_baked_frame_round_trips_with_quad_order_smoothing_and_flat_extent(self):
        from xtrifid.surface import baked_part

        vertices = np.array(
            [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [2.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
        )
        mesh = Mesh(vertices, ((0, 1, 2, 3),), (True,))
        baked = baked_part("Ткань", mesh)
        self.assertEqual(baked.shape, "surface")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloth.xtri"
            save_project(path, Scene((baked,)), "render")
            restored, _ = load_project(path)
            out = build(restored.parts[0], "render")
            np.testing.assert_allclose(out.vertices, vertices)
            self.assertEqual((out.faces, out.smooth), (mesh.faces, mesh.smooth))
            export_obj(path.with_suffix(".obj"), restored, "game")
            self.assertEqual(path.with_suffix(".obj").read_text().count("\nf "), 2)
            raw = json.loads(path.read_text())
            raw["parts"][0]["surface"]["vertices"] = "???"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_project(path)
        for bad in (
            Mesh(vertices, ((0, 1, 4),), (True,)),
            Mesh(vertices, ((0, 1, 1),), (False,)),
            Mesh(np.zeros_like(vertices), ((0, 1, 2),), (True,)),
            Mesh(
                np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [2, 0, 0]]), ((0, 1, 2, 3),), (True,)
            ),
            Mesh(vertices * float("nan"), mesh.faces, mesh.smooth),
        ):
            with self.assertRaises(ValueError):
                baked_part("err", bad)
