import os
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from test_model import spec
from test_reference import drawing
from test_studio import until

from xtrifid.gtk import Gio, Gtk
from xtrifid.model import Scene
from xtrifid.ui import Studio


@unittest.skipUnless(os.getenv("DISPLAY"), "requires GTK")
class WorkshopWindowTests(unittest.TestCase):
    def setUp(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid." + self._testMethodName,
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        self.w = Studio(app)
        self.w.present()
        self.addCleanup(self.w.close)

    def test_refining_the_image_choice_skips_superseded_decodes(self):
        from xtrifid.reference import read_image

        self.w.detail.set_selected(2)
        self.w.choose_references()
        dialog = self.w.reference_dialog
        started, resume = Event(), Event()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            drawing(path)
            visits = []

            def decode(chosen):
                visits.append(chosen)
                if chosen == "first.png":
                    started.set()
                    self.assertTrue(resume.wait(3))
                return read_image(path)

            with patch("xtrifid.reference_ui.read_image", side_effect=decode):
                dialog.set_path(0, "first.png")
                try:
                    self.assertTrue(started.wait(3))
                    dialog.set_path(1, "superseded.png")
                    dialog.set_path(1, "right.png")
                finally:
                    resume.set()
                until(lambda: dialog.generate.get_sensitive())
            self.assertEqual(visits, ["first.png", "right.png"])
            dialog.close()

    def test_style_rebuilds_scene_and_undo_restores_polygons(self):
        w = self.w
        scene = Scene.parse({"parts": [spec("sphere")], "detail": "high"})
        w.install(scene, "game")
        normal = w.stats.get_text()
        w.low_poly.set_active(True)
        self.assertEqual(w.doc.scene.style, "low_poly")
        self.assertNotEqual(w.stats.get_text(), normal)
        w.undo()
        self.assertEqual(w.stats.get_text(), normal)
        self.assertFalse(w.low_poly.get_active())

    def test_reference_brush_removes_volume_and_undo_restores_both_views(self):
        from xtrifid.reference import reference_scene

        w = self.w
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "block.png"
            drawing(path)
            w.install(reference_scene(path, path, detail="low"), "game")
            editor = w.mask_editor
            before = w.doc.scene.parts[0].masks
            editor.set_expanded(True)
            editor.ink.set_selected(1)
            editor.brush.set_value(7)
            editor.stamp(len(before[0][0]) // 2, len(before[0]) // 2)
            editor.apply_changes()
            after = w.doc.scene.parts[0].masks
            self.assertNotEqual(before[0], after[0])
            self.assertEqual(before[1], after[1])
            w.undo()
            self.assertEqual(w.doc.scene.parts[0].masks, before)

    def test_photo_file_dialog_generates_silhouette_at_selected_style_and_detail(self):
        w = self.w
        w.detail.set_selected(2)
        w.low_poly.set_active(True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "кольцо.png"
            drawing(path, hole=True)
            w.reference_button.emit("clicked")
            d = w.reference_dialog
            d.rounded.set_active(True)
            d.set_path(0, path)
            until(lambda: d.generate.get_sensitive())
            d.generate.emit("clicked")
            until(lambda: w.doc.scene is not None)
            s = w.doc.scene
            self.assertTrue(w.mask_editor.rounded.get_active())
            w.mask_editor.rounded.set_active(False)
            w.mask_editor.apply_changes()
            self.assertFalse(w.doc.scene.parts[0].rounded)
            w.undo()
            self.assertTrue(w.doc.scene.parts[0].rounded)
            self.assertEqual(
                (s.parts[0].shape, s.detail, s.style), ("reference", "medium", "low_poly")
            )
            self.assertIsNotNone(w.simulation.get_child())
            self.assertIsNotNone(w.material.get_child())
