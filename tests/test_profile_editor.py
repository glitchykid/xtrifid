import os
import unittest

from test_model import spec

from xtrifid.gtk import GLib, Gtk
from xtrifid.model import Part
from xtrifid.profile_editor import ProfileEditor


@unittest.skipUnless(os.getenv("DISPLAY"), "requires X11")
class ProfileEditorTests(unittest.TestCase):
    def setUp(self):
        self.applied = []
        self.errors = []
        self.requests = []
        self.editor = ProfileEditor(
            self.applied.append,
            self.errors.append,
            lambda i, text: self.requests.append((i, text)),
        )
        self.part = Part.parse(spec() | {"size": [2, 2, 1]})
        self.editor.set_part(self.part)
        self.window = Gtk.Window(default_width=300)
        self.addCleanup(self.window.close)
        self.window.set_child(self.editor)
        self.window.present()
        context = GLib.MainContext.default()
        while context.pending():
            context.iteration(False)

    def test_numeric_selection_edit_request_and_reset(self):
        self.editor.selector.set_value(2)
        self.assertEqual(self.editor.selected_index, 1)
        self.editor.prompt.set_text("  Смести вершину вверх  ")
        self.editor.prompt.emit("activate")
        self.assertEqual(self.requests, [(1, "Смести вершину вверх")])

        old_x = float(self.editor.fields[0].get_text())
        self.editor.fields[0].set_text(str(old_x * 0.75).replace(".", ","))
        self.editor.apply_button.emit("clicked")
        self.assertEqual(len(self.applied), 1)
        self.assertEqual(self.applied[0].shape, "extrusion")
        self.editor.set_part(self.applied[0])
        self.assertEqual(self.editor.selected_index, 1)
        self.editor.set_part(self.part, reset_selection=True)
        self.assertEqual(self.editor.selected_index, 0)
        self.assertEqual(self.editor.prompt.get_text(), "")

    def test_selection_notifies_with_updated_coordinates_and_valid_indices(self):
        selections = []
        editor = ProfileEditor(
            self.applied.append,
            self.errors.append,
            lambda *_: None,
            lambda: selections.append(
                (editor.selected_index, tuple(float(e.get_text()) for e in editor.fields))
            ),
        )
        editor.set_part(self.part)
        editor.selector.set_value(4)
        self.assertEqual(selections[-1][0], 3)
        self.assertEqual(selections[-1][1], editor.points[3])
        editor.set_part(Part.parse(spec("cone")))
        self.assertEqual(selections[-1], (2, editor.points[2]))
        editor.set_part(None)
        self.assertEqual(len(selections), 2)

    def test_rejects_invalid_values_and_ignores_unsupported_or_busy(self):
        initial = [entry.get_text() for entry in self.editor.fields]
        for value in ("NaN", "", "1e500", "очень много"):
            with self.subTest(value=value):
                self.editor.fields[0].set_text(value)
                self.editor.apply_button.emit("clicked")
                self.assertEqual([entry.get_text() for entry in self.editor.fields], initial)
        self.assertEqual(len(self.errors), 4)
        self.assertFalse(self.applied)
        self.editor.set_sensitive(False)
        self.editor.prompt.set_text("правка")
        self.editor.prompt.emit("activate")
        self.assertFalse(self.requests)
        self.editor.set_sensitive(True)
        for part in (Part.parse(spec("sphere")), None):
            self.editor.set_part(part)
            self.assertFalse(self.editor.get_visible())
            self.editor.apply_button.emit("clicked")
            self.assertFalse(self.applied)

    def test_drag_commits_once_and_cancel_restores_coordinates(self):
        from xtrifid.profile import profile_points

        p = profile_points(self.part)
        x, y = self.editor.screen_point(p[2])
        self.editor.drag.emit("drag-begin", x, y)
        self.editor.drag.emit("drag-end", 0.0, 0.0)
        self.assertEqual(self.editor.selected_index, 2)
        self.assertFalse(self.applied)

        self.editor.drag.emit("drag-begin", x, y)
        self.editor.drag.emit("drag-update", 4.0, 2.0)
        self.editor.drag.emit("drag-update", 10.0, 5.0)
        self.assertFalse(self.applied)
        self.editor.drag.emit("drag-end", 10.0, 5.0)
        self.assertEqual(len(self.applied), 1)
        self.assertFalse(self.errors)

        self.editor.set_part(self.applied[0])
        coords = [entry.get_text() for entry in self.editor.fields]
        x, y = self.editor.screen_point(profile_points(self.applied[0])[2])
        self.editor.drag.emit("drag-begin", x, y)
        self.editor.drag.emit("drag-update", -5.0, -7.0)
        self.editor.drag.emit("cancel", None)
        self.editor.drag.emit("drag-end", -5.0, -7.0)
        self.assertEqual(len(self.applied), 1)
        self.assertEqual([entry.get_text() for entry in self.editor.fields], coords)
