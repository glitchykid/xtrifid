import os
import unittest

from test_model import spec

from xtrifid.gtk import Gio, GLib, Gtk
from xtrifid.model import Scene
from xtrifid.ui import Studio


def until(predicate, seconds=12):
    loop = GLib.MainLoop()
    expired = []

    def check():
        if predicate():
            loop.quit()
            return False
        return True

    def timeout():
        expired.append(True)
        loop.quit()
        return False

    tick = GLib.timeout_add(20, check)
    limit = GLib.timeout_add_seconds(seconds, timeout)
    loop.run()
    GLib.source_remove(tick if expired else limit)
    if expired:
        raise AssertionError("GTK did not reach the expected state")


@unittest.skipUnless(os.getenv("DISPLAY"), "requires X11")
class StudioTests(unittest.TestCase):
    def test_window_mesh_selection_exact_edit_profile_and_undo(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.Test",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.present()
        scene = Scene.parse(
            {"parts": [spec(), spec("sphere") | {"center": [3, 0.123456, 2], "name": "Шар"}]}
        )
        window.install(scene, "game")
        until(lambda: window.view.drawn and window.view.get_mapped())
        self.assertIsNone(window.view.get_error())
        self.assertIsNone(window.view.problem)
        self.assertEqual(window.view.pick(*window.view.screen_position([3, 0, 2])), 1)
        window.select(1)
        window.fields["size"][0].set_value(0.75)
        window.apply_numeric()
        self.assertIs(window.doc.scene.parts[0], scene.parts[0])
        self.assertEqual(window.doc.scene.parts[1].size, (0.75, 3, 4))
        self.assertEqual(window.doc.scene.parts[1].center, scene.parts[1].center)
        window.mode.set_selected(1)
        self.assertEqual(window.doc.mode, "render")
        window.undo()
        window.undo()
        self.assertEqual(window.doc.scene, scene)

    def test_tube_inspector_rejects_closed_bore_without_changing_scene(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.BoreTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        scene = Scene.parse({"parts": [spec("tube", inner_diameter=1)]})
        window.install(scene, "render")
        self.assertTrue(window.bore_row.get_visible())
        window.bore.set_value(2.5)
        window.apply_numeric()
        self.assertEqual(window.doc.scene, scene)
        self.assertIn("диаметр", window.status.get_text())
        window.bore.set_value(0.5)
        window.apply_numeric()
        self.assertEqual(window.doc.scene.parts[0].inner_diameter, 0.5)

    def test_export_dialog_writes_selected_mesh_at_small_window_size(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.FileTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.set_default_size(760, 540)
        window.present()
        window.install(Scene.parse({"parts": [spec()]}), "game")
        until(lambda: window.view.drawn)
        self.assertGreater(window.view.get_width(), 200)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"XTRIFID_FILES": tmp}),
        ):
            window.exporter.emit("clicked")

            def chooser():
                return next(
                    (
                        w
                        for w in Gtk.Window.list_toplevels()
                        if isinstance(w, Gtk.FileChooserDialog) and w.get_visible()
                    ),
                    None,
                )

            until(
                lambda: (
                    chooser()
                    and chooser().get_current_folder()
                    and chooser().get_current_folder().get_path() == tmp
                )
            )
            dialog = chooser()
            dialog.set_current_name("part.obj")
            dialog.response(Gtk.ResponseType.ACCEPT)
            target = Path(tmp) / "part.obj"
            until(target.is_file)
            self.assertEqual(
                sum(line.startswith("f ") for line in target.read_text().splitlines()), 12
            )
