import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_model import spec
from test_studio import until

from xtrifid import geometry
from xtrifid.gtk import Gio, Gtk
from xtrifid.model import Part, Scene
from xtrifid.profile import profile_guide
from xtrifid.storage import export_obj, load_project, save_project


class DetailTests(unittest.TestCase):
    def test_releasing_dense_mesh_frees_its_geometry_after_an_edit(self):
        import gc
        import weakref

        part = Part.parse(spec("torus"))
        mesh = geometry.build(part, "render", "high")
        retained = weakref.ref(mesh.vertices)
        del mesh
        gc.collect()
        self.assertIsNone(retained())

    def test_detail_refines_curves_preserving_dimensions_and_planar_topology(self):
        parts = [Part.parse(spec(kind)) for kind in ("sphere", "box", "torus")]
        tube = Part.parse(spec("tube", inner_diameter=1))
        for mode in ("game", "render"):
            counts = []
            for detail in ("low", "medium", "high"):
                curved, box, torus = [geometry.build(p, mode, detail) for p in parts]
                np.testing.assert_allclose(np.ptp(curved.vertices, axis=0), parts[0].size)
                self.assertEqual(len(box.faces), 12 if mode == "game" else 6)
                counts.append(len(curved.vertices))
                for p, mesh in zip(parts, (curved, box, torus)):
                    actual = sum(len(f) - 2 for f in mesh.faces)
                    self.assertEqual(geometry.triangle_bound(p, mode, detail), actual)
                guide = profile_guide(tube, 0, mode, detail)
                self.assertEqual(len(guide), geometry.radial_segments(mode, detail) + 1)
                np.testing.assert_allclose(guide[0], guide[-1])
            self.assertLess(counts[0], counts[1])
            self.assertLess(counts[1], counts[2])
        with self.assertRaises(ValueError):
            geometry.build(tube, "game", "unbounded")

    def test_project_version_and_export_use_selected_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.xtri"
            path.write_text(
                json.dumps({"version": 1, "mode": "game", "parts": [spec("sphere")]})
            )
            scene, mode = load_project(path)
            self.assertEqual(scene.detail, "medium")
            high = replace(scene, detail="high")
            save_project(path, high, mode)
            self.assertEqual(json.loads(path.read_text())["version"], 4)
            compatible = json.loads(path.read_text()) | {"version": 2}
            path.write_text(json.dumps(compatible))
            self.assertEqual(load_project(path), (high, mode))
            obj = path.with_suffix(".obj")
            export_obj(obj, high, mode)
            self.assertEqual(
                sum(line.startswith("f ") for line in obj.read_text().splitlines()),
                len(geometry.build(high.parts[0], mode, "high").faces),
            )
            for value in ("unlimited", None):
                with self.subTest(value=value):
                    data = json.loads(path.read_text()) | {"detail": value}
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        load_project(path)

    def test_rejected_export_does_not_allocate_or_replace_existing_file(self):
        scene = Scene.parse({"parts": [spec("sphere")]})
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("xtrifid.geometry.mesh_triangle_budget", return_value=1),
        ):
            path = Path(tmp) / "existing.obj"
            path.write_text("original")
            with (
                patch("xtrifid.storage.build") as mesh,
                self.assertRaisesRegex(ValueError, "памят|лимит"),
            ):
                export_obj(path, scene, "game")
            mesh.assert_not_called()
            self.assertEqual(path.read_text(), "original")


@unittest.skipUnless(os.getenv("DISPLAY"), "requires X11")
class DetailWindowTests(unittest.TestCase):
    def setUp(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid." + self._testMethodName,
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        from xtrifid.ui import Studio

        self.window = Studio(app)
        self.addCleanup(self.window.close)

    def test_human_face_and_anatomy_controls_follow_selection(self):
        from xtrifid.character import build_character

        w = self.window
        w.present()
        scene = build_character({"cloak": "knee"}, "low")
        w.install(scene, "game")
        until(lambda: w.view.drawn)
        self.assertEqual(w.focus_part.get_label(), "К лицу")
        w.focus_part.emit("clicked")
        self.assertGreater(w.view.zoom, 3)
        self.assertGreater(w.view.target[2], 0.55)
        self.assertTrue(w.anatomy.get_visible())
        w.zone.set_selected(1)
        w.zone_scale.set_value(1.12)
        w.pose.set_selected(2)
        w.physique.set_selected(1)
        w.apply_anatomy()
        self.assertEqual(dict(w.doc.scene.parts[0].body_scales)["ears"], 1.12)
        self.assertEqual(w.doc.scene.parts[0].pose, "arms_out")
        self.assertEqual(w.doc.scene.parts[0].physique, "male")
        self.assertEqual(w.doc.scene.parts[1:], scene.parts[1:])
        w.undo()
        self.assertEqual(w.doc.scene, scene)
        w.select(1)
        self.assertFalse(w.anatomy.get_visible())

    def test_preview_reveals_curvature_on_the_back_of_a_model(self):
        from OpenGL import GL

        w = self.window
        w.present()
        sphere = spec("sphere") | {"size": [2, 2, 2], "center": [0, 0, 0]}
        w.install(Scene.parse({"parts": [sphere]}), "render")
        v = w.view
        until(lambda: v.drawn)
        v.yaw, v.pitch = math.pi, 0
        v.make_current()
        v.attach_buffers()
        v.render_mesh(v, v.get_context())
        green = []
        for side in (-0.45, 0.45):
            x, y = v.screen_position([side, 0.89, 0])
            rgb = GL.glReadPixels(
                int(x), v.get_height() - int(y), 1, 1, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
            )
            green.append(rgb[1])
        self.assertGreater(abs(green[1] - green[0]), 8, green)

    def test_explicit_choice_and_protection_before_generation(self):
        w = self.window
        w.prompt.set_text("Сделай человека с плащом")
        with patch.object(w.planner, "generate") as plan:
            w.prompt.emit("activate")
            plan.assert_not_called()
            self.assertFalse(w.generate.get_sensitive())
            self.assertIn("детализац", w.status.get_text())
            w.detail.set_selected(1)
            self.assertTrue(w.generate.get_sensitive())
            with patch("xtrifid.geometry.mesh_triangle_budget", return_value=0):
                w.prompt.emit("activate")
                self.assertFalse(w.busy)
                plan.assert_not_called()
                self.assertIn("памят", w.status.get_text())

    def test_refused_resolution_change_preserves_scene_preview_history_and_selection(self):
        w = self.window
        scene = Scene.parse({"parts": [spec("box"), spec("sphere")]})
        w.install(scene, "game")
        w.select(1)
        initial = (w.doc.scene, w.view.buffer, tuple(w.doc.history))
        with (
            patch("xtrifid.geometry.mesh_triangle_budget", return_value=240),
            patch("xtrifid.ui.build") as mesh,
        ):
            w.detail.set_selected(3)
            mesh.assert_not_called()
            self.assertIs(w.doc.scene, initial[0])
            self.assertIs(w.view.buffer, initial[1])
            self.assertEqual(tuple(w.doc.history), initial[2])
            self.assertEqual(w.detail.get_selected(), 2)
            self.assertEqual(w.selected, 1)
            self.assertIn("лимит", w.status.get_text())
        w.detail.set_selected(1)
        self.assertEqual(w.doc.scene.detail, "low")
        self.assertEqual(w.doc.scene.parts, scene.parts)
        self.assertNotIn("лимит", w.status.get_text())
        w.undo()
        self.assertEqual(w.doc.scene, scene)
        self.assertEqual(w.detail.get_selected(), 2)

    def test_preview_allocation_failure_preserves_camera_mesh_and_picking(self):
        w = self.window
        w.install(Scene.parse({"parts": [spec("box")]}), "game")
        view = w.view
        before = (view.center, view.radius, view.owners, view.pick_points, view.buffer)
        replacement = Scene.parse({"parts": [spec("sphere") | {"center": [9, 5, 2]}]})
        with patch("xtrifid.viewport.batches", side_effect=MemoryError):
            with self.assertRaisesRegex(ValueError, "памяти"):
                w.install(replacement, "game")
        for original, current in zip(
            before, (view.center, view.radius, view.owners, view.pick_points, view.buffer)
        ):
            self.assertIs(original, current)
        self.assertEqual(w.doc.scene.parts[0].shape, "box")
        ranges = view.ranges
        with (
            patch("xtrifid.viewport.np.column_stack", side_effect=MemoryError),
            self.assertRaisesRegex(ValueError, "памяти"),
        ):
            w.install(replacement, "game")
        self.assertIs(view.ranges, ranges)

    def test_new_small_model_can_replace_unaffordable_old_scene(self):
        w = self.window
        w.install(Scene.parse({"parts": [spec("torus")]}), "render")
        box = Scene.parse({"parts": [spec("box")]})
        w.prompt.set_text("box")
        with (
            patch("xtrifid.geometry.mesh_triangle_budget", return_value=12),
            patch.object(w.planner, "generate", return_value=box) as plan,
        ):
            w.prompt.emit("activate")
            until(lambda: not w.busy)
            plan.assert_called_once()
            self.assertEqual(w.doc.scene, box)

    def test_overbudget_project_offers_a_lower_detail_before_building(self):
        w = self.window
        scene = Scene.parse({"parts": [spec("sphere")], "detail": "high"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.xtri"
            save_project(path, scene, "game")
            with (
                patch("xtrifid.geometry.mesh_triangle_budget", return_value=100),
                patch.object(Gtk.AlertDialog, "choose") as choose,
            ):
                w.open_project(Gio.File.new_for_path(str(path)))
                choose.assert_called_once()
                self.assertIsNone(w.doc.scene)
                callback = choose.call_args.args[-1]
                dialog = Gtk.AlertDialog()
                with patch.object(Gtk.AlertDialog, "choose_finish", return_value=1):
                    callback(dialog, None)
                self.assertEqual(w.doc.scene, replace(scene, detail="low"))
                self.assertEqual(load_project(path)[0], scene)
