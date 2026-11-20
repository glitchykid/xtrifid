import os
import unittest

from test_sculpt_planner import RECIPE

from xtrifid.sculpt_plan import sculpt_scene


@unittest.skipUnless(os.getenv("DISPLAY"), "requires GTK display")
class SculptPanelTests(unittest.TestCase):
    def test_edit_named_tip_keeps_the_other_nodes_and_undoable_input(self):
        from xtrifid.sculpt_panel import SculptPanel

        scene = sculpt_scene(RECIPE, "high")
        saved = []
        panel = SculptPanel(saved.append, self.fail)
        part = scene.parts[0]
        panel.set_part(part, True)
        panel.node.set_selected(3)
        panel.fields[2].set_value(0.95)
        panel.commit()
        (changed,) = saved
        self.assertEqual(changed.nodes[:3], part.nodes[:3])
        self.assertEqual(changed.nodes[3][:4], part.nodes[3][:4])
        self.assertEqual(changed.nodes[3][4:], (0.95, 0.09))
        self.assertEqual(part.nodes[3][4], 0.8)
        panel.set_part(changed, False)
        self.assertEqual(panel.node.get_selected(), 3)
        panel.commit()
        self.assertEqual(len(saved), 1)
        panel.set_part(None, True)
        self.assertFalse(panel.get_visible())
