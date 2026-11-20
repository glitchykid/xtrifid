import json
import unittest
from unittest.mock import patch

from xtrifid.model import Scene
from xtrifid.planner import Planner, PlannerError

BRANCH = {"subject": "branched_organic", "body": "побег", "growth": "разветвление побега"}
RECIPE = {
    "name": "Отросток",
    "width_m": 0.6,
    "depth_m": 0.24,
    "height_m": 0.8,
    "root": {"name": "основание", "right": 0, "back": 0, "up": -0.8, "radius": 0.2},
    "nodes": [
        {"name": "узел", "parent": "основание", "right": 0, "back": 0, "up": 0, "radius": 0.15},
        {
            "name": "левый",
            "parent": "узел",
            "right": -0.8,
            "back": 0.2,
            "up": 0.7,
            "radius": 0.08,
        },
        {
            "name": "правый",
            "parent": "узел",
            "right": 0.7,
            "back": -0.1,
            "up": 0.8,
            "radius": 0.09,
        },
    ],
}


class SculptPlannerTests(unittest.TestCase):
    def test_a_photographed_assembly_cannot_take_a_shortcut_into_the_organic_schema(self):
        invalid = {"subject": "object", "form": "branched"}
        assembly = {"subject": "object", "components": [], "form": "general"}
        geometry = {
            "parts": [
                {
                    "name": "Stand",
                    "shape": "box",
                    "size": [0.2, 0.15, 0.03],
                    "center": [0, 0, 0.015],
                    "rotation": [0, 0, 0],
                    "support": 0,
                }
            ]
        }
        with patch.object(
            Planner, "_chat", side_effect=map(json.dumps, [invalid, assembly, geometry])
        ) as chat:
            scene = Planner().generate("A ceramic stand with an attached handle")
        self.assertEqual(scene.parts[0].shape, "box")
        alternatives = chat.call_args_list[0].args[1]["oneOf"]
        object_forms = [
            branch["properties"]["form"]["enum"]
            for branch in alternatives
            if branch["properties"]["subject"]["enum"] == ["object"]
        ]
        self.assertFalse(any("branched" in options for options in object_forms))
        self.assertEqual(chat.call_count, 3)

    def test_edit_sends_only_reference_metadata_and_preserves_the_mask(self):
        bitmap = ("10" * 48,) * 96
        scene = Scene.parse(
            {
                "parts": [
                    {
                        "name": "Силуэт",
                        "shape": "reference",
                        "size": [1, 1, 1],
                        "center": [0, 0, 0.5],
                        "rotation": [0, 0, 0],
                        "masks": [bitmap],
                    }
                ]
            }
        )
        with patch.object(Planner, "_chat", return_value='{"size":[2,1,1]}') as chat:
            edited = Planner().edit("Удвой ширину", scene, 0)
        message = chat.call_args.args[0][1]["content"]
        self.assertLess(len(message), 1000)
        self.assertNotIn(bitmap[0], message)
        self.assertEqual(edited.masks, scene.parts[0].masks)

    def test_connected_form_uses_the_named_skeleton_and_requested_size(self):
        with patch.object(Planner, "_chat", side_effect=map(json.dumps, [BRANCH, RECIPE])):
            scene = Planner().generate(
                "Разветвлённый побег высотой восемьдесят сантиметров", "high"
            )
        self.assertEqual(scene.detail, "high")
        (part,) = scene.parts
        self.assertEqual(
            (part.shape, part.size, part.center), ("sculpt", (0.6, 0.24, 0.8), (0, 0, 0.4))
        )
        self.assertEqual([row[1] for row in part.nodes], [-1, 0, 1, 1])

    def test_disconnected_or_cyclic_nodes_must_be_repaired(self):
        broken = RECIPE | {
            "nodes": [RECIPE["nodes"][0] | {"parent": "узел"}, RECIPE["nodes"][1]]
        }
        with (
            patch.object(
                Planner, "_chat", side_effect=map(json.dumps, [BRANCH, broken, broken])
            ),
            self.assertRaises(PlannerError),
        ):
            Planner().generate("Живой разветвлённый побег")
