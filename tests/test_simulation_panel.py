import os
import unittest

import numpy as np
from test_simulation import support
from test_studio import until

from xtrifid.geometry import build
from xtrifid.gtk import Gtk
from xtrifid.simulation_panel import SimulationPanel


@unittest.skipUnless(os.getenv("DISPLAY"), "requires GTK")
class SimulationPanelTests(unittest.TestCase):
    def test_play_pause_scrub_bake_and_selection_change(self):
        previews, bakes, errors = [], [], []
        panel = SimulationPanel(
            lambda mesh, replace: previews.append((mesh, replace)),
            lambda name, mesh, replace: bakes.append((name, mesh, replace)),
            errors.append,
        )
        window = Gtk.Window(child=panel)
        self.addCleanup(window.close)
        window.present()
        part = support()
        panel.set_target(part, build(part, "render"), "low")
        panel.set_expanded(True)
        panel.kind.set_selected(2)
        panel.play.emit("clicked")
        until(lambda: panel.sim is not None and panel.sim.frame >= 2)
        panel.play.emit("clicked")
        self.assertFalse(panel.playing)
        panel.timeline.set_value(23)
        until(lambda: panel.sim.frame == 23)
        visible = previews[-1][0].vertices.copy()
        panel.bake.emit("clicked")
        self.assertEqual(len(bakes), 1)
        self.assertEqual(bakes[0][0], "Водная поверхность")
        self.assertFalse(bakes[0][2])
        np.testing.assert_array_equal(bakes[0][1].vertices, visible)
        panel.timeline.set_value(10)
        until(lambda: panel.sim is not None and panel.sim.frame == 10)
        panel.first.emit("clicked")
        self.assertEqual(panel.sim.frame, 0)
        panel.play.emit("clicked")
        panel.set_target(None, None, "low")
        self.assertFalse(panel.playing)
        self.assertIsNone(panel.sim)
        self.assertIsNone(previews[-1][0])
        self.assertFalse(errors, errors)

    def test_preview_callback_can_cancel_the_session_during_playback(self):
        emitted = []

        def preview(mesh, replaces):
            emitted.append(mesh)
            if mesh is not None:
                panel.set_target(None, None, "low")

        panel = SimulationPanel(preview, lambda *_: None, self.fail)
        part = support()
        panel.set_target(part, build(part, "game"), "low")
        panel.play.emit("clicked")
        self.assertFalse(panel._tick())
        self.assertIsNone(panel.sim)
        self.assertIsNone(emitted[-1])
