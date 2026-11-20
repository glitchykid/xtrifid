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
    def test_focus_frames_selected_part_and_keeps_picking_in_sync(self):
        from xtrifid.gtk import Gdk

        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.FocusTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.present()
        scene = Scene.parse(
            {"parts": [spec(), spec("sphere") | {"center": [6, 0, 4], "size": [0.5, 0.5, 0.5]}]}
        )
        window.install(scene, "render")
        view = window.view
        until(lambda: view.drawn)
        window.select(1)
        window.focus_part.emit("clicked")
        cx, cy = view.get_width() / 2, view.get_height() / 2
        for yaw in (0, 0.7, 1.57):
            view.yaw = yaw
            x, y = view.screen_position([6, 0, 4])
            self.assertAlmostEqual(x, cx, places=4)
            self.assertAlmostEqual(y, cy, places=4)
            self.assertEqual(view.pick(cx, cy), 1)
        self.assertGreater(view.zoom, 3)
        view.key(None, Gdk.KEY_Home, 0, 0)
        self.assertEqual(view.zoom, 1)
        self.assertNotAlmostEqual(view.screen_position([6, 0, 4])[0], cx)

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

    def test_selected_profile_ring_appears_in_gl_frame_and_follows_point(self):
        import numpy as np
        from OpenGL import GL

        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.GuideTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.present()
        scene = Scene.parse({"parts": [spec("tube", inner_diameter=1)]})
        window.install(scene, "render")
        until(lambda: window.view.drawn)
        window.profile_section.set_expanded(True)
        window.point_editor.selector.set_value(3)
        view = window.view

        def pixels():
            view.make_current()
            view.attach_buffers()
            view.render_mesh(view, view.get_context())
            w, h = GL.glGetIntegerv(GL.GL_VIEWPORT)[2:]
            data = GL.glReadPixels(0, 0, int(w), int(h), GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            rgb = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            return int(((rgb[:, 0] > 240) & (rgb[:, 1] > 145) & (rgb[:, 2] < 145)).sum())

        self.assertGreater(pixels(), 30)
        self.assertEqual(len(view.guide), 49)
        window.mode.set_selected(0)
        self.assertEqual(len(view.guide), 17)
        window.profile_section.set_expanded(False)
        self.assertEqual(pixels(), 0)
        self.assertIsNone(view.problem)

    def test_profile_controls_follow_selection_and_commit_only_target_point(self):
        from threading import Event
        from unittest.mock import patch

        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.PointTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.present()
        scene = Scene.parse({"parts": [spec(), spec("sphere") | {"name": "Шар"}]})
        window.install(scene, "game")
        self.assertTrue(window.profile_section.get_visible())
        window.select(1)
        self.assertFalse(window.profile_section.get_visible())
        window.select(0)
        window.profile_section.set_expanded(True)
        arrived, release = Event(), Event()
        self.addCleanup(release.set)

        def edit(prompt, source, index, point):
            from xtrifid.profile import move_profile_point, profile_points

            arrived.set()
            release.wait(5)
            x, y = profile_points(source.parts[index])[point]
            return move_profile_point(source.parts[index], point, (x + 0.05, y))

        with patch.object(window.planner, "edit_point", side_effect=edit) as planner:
            window.request_point(0, "Смести точку по X на 5 см")
            until(arrived.is_set)
            window.select(1)
            self.assertEqual(window.selected, 0)
            self.assertFalse(window.inspector.get_sensitive())
            release.set()
            until(lambda: not window.busy)
            planner.assert_called_once_with("Смести точку по X на 5 см", scene, 0, 0)
        self.assertIs(window.doc.scene.parts[1], scene.parts[1])
        self.assertEqual(window.doc.scene.parts[0].shape, "extrusion")
        window.undo()
        self.assertEqual(window.doc.scene, scene)
