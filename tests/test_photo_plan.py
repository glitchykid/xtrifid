import json
import unittest
from math import cos, radians, sin
from unittest.mock import Mock

import numpy as np

from xtrifid.geometry import build
from xtrifid.photo_plan import _from_rows, construct, from_photo
from xtrifid.planner import Planner, PlannerError


def item(name, form, size, center, parent=-1, joint="free", yaw=0):
    return dict(
        name=name, form=form, size=size, center=center, yaw=yaw, parent=parent, joint=joint
    )


class PhotoPlanTests(unittest.TestCase):
    def test_revolved_shell_sits_in_tray_and_visible_vertical_loop_connects_with_real_hole(
        self,
    ):
        scene = construct(
            [
                item("Tray", "shallow_dish", [1, 1, 0.06], [0, 0, 0.03]),
                item(
                    "Shell",
                    "hollow_revolution",
                    [0.58, 0.58, 0.46],
                    [0.03, 0.02, 0.23],
                    0,
                    "rest",
                ),
                item(
                    "Loop",
                    "vertical_loop",
                    [0.23, 0.055, 0.29],
                    [-0.39, -0.26, 0.29],
                    1,
                    "side",
                ),
            ],
            "medium",
        )
        plate, body, handle = scene.parts
        self.assertEqual((plate.shape, body.shape, handle.shape), ("lathe", "lathe", "torus"))
        for p in (plate, body):
            self.assertGreater(
                max(z for r, z in p.outline if r > 0.4), max(z for r, z in p.outline if r == 0)
            )
        pb = build(body, "render")
        self.assertLess(pb.vertices[:, 2].min(), 0.035)
        self.assertGreater(pb.vertices[:, 2].min(), 0)
        direction = np.array(
            (cos(radians(handle.rotation[2])), sin(radians(handle.rotation[2])))
        )
        self.assertLess(direction[0], 0)
        self.assertLess(direction[1], 0)
        mesh = build(handle, "render")
        local = mesh.vertices[:, :2] - handle.center[:2]
        lateral = local @ direction
        thickness = local @ (-direction[1], direction[0])
        self.assertLess(np.ptp(thickness), 0.061)
        self.assertAlmostEqual(np.ptp(lateral), 0.23, places=6)
        self.assertAlmostEqual(np.ptp(mesh.vertices[:, 2]), 0.29, places=6)
        # There is a continuous through opening in the vertical plane.
        self.assertGreater(
            np.min(np.hypot(lateral, mesh.vertices[:, 2] - handle.center[2])), 0.035
        )

    def test_horizontal_appendage_follows_yaw_and_remains_a_separate_part(self):
        scene = construct(
            [item("Rounded grip", "spindle", [0.8, 0.07, 0.035], [0, 0, 0.02], yaw=35)], "low"
        )
        p = scene.parts[0]
        v = build(p, "render").vertices
        self.assertEqual(p.shape, "loft")
        self.assertGreater(np.ptp(v[:, 0]), 0.55)
        self.assertGreater(np.ptp(v[:, 1]), 0.39)
        self.assertLess(np.ptp(v[:, 2]), 0.036)

    def test_unknown_shape_and_forward_attachment_are_rejected(self):
        for part in (
            item("Ghost", "catalogue", [1, 1, 1], [0, 0, 0.5]),
            item("Detached", "box", [1, 1, 1], [0, 0, 0.5], 1, "rest"),
        ):
            with self.assertRaises(ValueError):
                construct([part], "low")

    def test_observed_geometry_is_built_from_visual_brief_in_shared_centimetres(self):
        planner = Planner()
        planner._chat = Mock()
        resolution = {
            "route": "constructed",
            "parts": [
                "Plate | box | 22,13,1 | 0,0,0.5 | 0 | - | free",
                "Loop | vertical_loop | 7,1,4 | 12,0,1 | 0 | Plate | side",
            ],
        }
        planner._chat.side_effect = [
            json.dumps({"description": "A rectangular plate with an upright loop."}),
            json.dumps(resolution),
            PlannerError("Model unavailable"),
        ]
        scene = from_photo(planner, "encoded_pixels", False, "Preserve the hole", 1.0, "low")
        self.assertEqual([p.shape for p in scene.parts], ["box", "torus"])
        survey, design, focus = planner._chat.call_args_list
        self.assertEqual(focus.args[0][-1]["images"], ["encoded_pixels"])
        self.assertEqual(survey.args[0][-1]["images"], ["encoded_pixels"])
        self.assertIn("upright loop", design.args[0][-1]["content"])
        self.assertIn("Preserve the hole", design.args[0][-1]["content"])

    def test_level_contents_wet_inside_the_rim_and_stay_under_it(self):
        result = construct(
            [
                item("Reservoir", "hollow_revolution", [1, 1, 0.8], [0, 0, 0.4]),
                item("Fill", "flat_disc", [1.2, 0.8, 0.2], [0.1, 0.1, 0.9], 0, "inside"),
            ],
            "low",
        )
        body, fill = result.parts
        self.assertEqual(fill.center[:2], body.center[:2])
        self.assertLess(fill.center[2] + fill.size[2] / 2, body.center[2] + body.size[2] / 2)
        self.assertLess(fill.size[0], body.size[0])
        self.assertAlmostEqual(fill.size[0], fill.size[1])

    def test_named_supports_can_be_observed_later_in_the_image(self):
        scene = _from_rows(
            [
                "Pod | box | 4,2,6 | 0,0,4 | 0 | Stand | rest",
                "Stand | cylinder | 7,7,1 | 0,0,.5 | 0 | - | free",
            ],
            "medium",
        )
        self.assertEqual([p.name for p in scene.parts], ["Stand", "Pod"])
        self.assertAlmostEqual(scene.parts[1].center[2] - scene.parts[1].size[2] / 2, 1 / 7)

    def test_inconsistent_table_gets_one_repair_with_explanation(self):
        planner = Planner()
        planner._chat = Mock()
        planner._chat.side_effect = [
            json.dumps({"description": "Two blocks are stacked."}),
            json.dumps({"route": "constructed", "parts": ["unknown"]}),
            json.dumps(
                {
                    "route": "constructed",
                    "parts": [
                        "Base | box | 2,3,1 | 0,0,.5 | 0 | - | free",
                        "Top | box | 1,2,1 | 0,0,1.5 | 0 | Base | rest",
                    ],
                }
            ),
        ]
        scene = from_photo(planner, "pixels", False, "", 1, "low")
        self.assertEqual(len(scene.parts), 2)
        self.assertIn("семь", planner._chat.call_args.args[0][-1]["content"])

    def test_one_attachment_lookup_accepts_both_image_axes_and_measures_a_wider_end(self):
        planner = Planner()
        planner._chat = Mock(
            side_effect=[
                json.dumps(
                    {"description": "A body with a pierced side loop and a slender accessory."}
                ),
                json.dumps(
                    {
                        "route": "semantic",
                        "parts": [
                            "Body | hollow_revolution | 8,8,9 | 0,0,4.5 | 0 | - | free",
                            "Loop | vertical_loop | 3,1,4 | 5,0,4 | 0 | Body | side",
                            "Blade | bar | 12,2,1 | 4,0,0.5 | 0 | - | free",
                        ],
                    }
                ),
                json.dumps(
                    {
                        "loops": ["Loop | left | front"],
                        "shafts": ["Blade | right | back | flat | 18,5,2.5 | 3,1.5,0.3"],
                    }
                ),
            ]
        )
        scene = from_photo(planner, "pixels", False, "", 0.12, "medium")
        names = {p.name: p for p in scene.parts}
        self.assertLess(names["Loop"].center[0], 0)
        self.assertLess(names["Loop"].center[1], 0)
        self.assertEqual(names["Blade"].shape, "loft")
        self.assertTrue(any(p.shape == "sphere" for p in scene.parts))
        self.assertEqual(planner._chat.call_count, 3)
