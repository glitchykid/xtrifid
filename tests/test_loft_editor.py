import os
import unittest

import numpy as np
from test_freeform import SHELL
from test_studio import until

from xtrifid.geometry import build
from xtrifid.gtk import Gio, Gtk
from xtrifid.model import Part, Scene
from xtrifid.ui import Studio


@unittest.skipUnless(os.getenv("DISPLAY"), "requires X11")
class LoftEditorTests(unittest.TestCase):
    def test_radius_edit_preserves_unselected_rings_and_rejects_crossing(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.Sections",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        w = Studio(app)
        self.addCleanup(w.close)
        part = Part.parse(SHELL)
        w.install(Scene((part,), "low"), "game")
        w.present()
        until(lambda: w.view.drawn)
        editor = w.loft
        editor.set_expanded(True)
        editor.number.set_value(2)
        self.assertEqual(len(w.view.guide), 9)
        before = build(part, "game", "low").vertices
        editor.fields[1].set_value(editor.fields[1].get_value() + 0.013)
        editor.commit()
        updated = w.doc.scene.parts[0]
        after = build(updated, "game", "low").vertices
        self.assertGreater(np.max(abs(before[8:16] - after[8:16])), 0.01)
        np.testing.assert_allclose(before[:8], after[:8], atol=1e-10)
        np.testing.assert_allclose(before[16:], after[16:], atol=1e-10)
        self.assertEqual(editor.number.get_value_as_int(), 2)
        editor.fields[0].set_value(part.size[2])
        editor.commit()
        self.assertIs(w.doc.scene.parts[0], updated)
        self.assertIn("сечен", w.status.get_text())
        w.undo()
        self.assertEqual(w.doc.scene.parts[0], part)
