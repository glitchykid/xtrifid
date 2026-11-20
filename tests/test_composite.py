import json
import unittest
from unittest.mock import patch

import numpy as np

from xtrifid.geometry import build
from xtrifid.planner import Planner, PlannerError

SUPPORT = {"name": "Колонна", "support": 0, "width": "unspecified", "joint": "resting"}
COVER = {"name": "Навес", "support": 1, "width": "wider", "joint": "joined"}
SUBJECT = {"subject": "object", "components": [SUPPORT, COVER], "form": "general"}
COLUMN = {
    "name": "Колонна",
    "shape": "cylinder",
    "size": [0.1, 0.1, 0.4],
    "center": [0, 0, 0.1],
    "rotation": [0, 0, 0],
    "support": 2,
}
CANOPY = {
    "name": "Навес",
    "shape": "box",
    "size": [0.1, 0.08, 0.04],
    "center": [0, 0, 0.1],
    "rotation": [0, 0, 0],
    "support": 0,
}


class CompositeTests(unittest.TestCase):
    def test_named_strictly_monotonic_outer_meridian_ends_in_two_planar_caps(self):
        from xtrifid.assembly import assemble

        profile = [
            {"radius": r, "height": h}
            for r, h in ((0.19, 0), (0.185, 0.2), (0.18, 0.4), (0.175, 0.6), (0.17, 0.8))
        ]
        solid = assemble(
            {"parts": [COLUMN | {"shape": "lathe", "outline": profile, "support": 0}]}
        ).parts[0]
        self.assertEqual({p for p in solid.outline if p[0] == 0}, {(0, 0), (0, 0.8)})
        self.assertEqual(len(solid.outline), 4)

    def test_turned_components_use_named_nonnegative_radii_in_the_generation_schema(self):
        turned = CANOPY | {
            "shape": "lathe",
            "outline": [
                {"radius": 0.6, "height": 0},
                {"radius": 0.42, "height": 0.4},
                {"radius": 0, "height": 1},
            ],
        }
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(json.dumps, [SUBJECT, {"parts": [COLUMN, turned]}]),
        ) as chat:
            scene = Planner().generate("Круглый колпак на колонне")
        shapes = chat.call_args.args[1]["properties"]["parts"]["items"]["oneOf"]
        branch = next(s for s in shapes if "lathe" in s["properties"]["shape"]["enum"])
        point = branch["properties"]["outline"]["items"]
        self.assertEqual(point["properties"]["radius"]["minimum"], 0)
        self.assertTrue(all(r >= 0 for r, z in scene.parts[1].outline))

    def test_attached_part_first_resolves_its_only_possible_support(self):
        reversed_roles = SUBJECT | {"components": [COVER, SUPPORT]}
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(json.dumps, [reversed_roles, {"parts": [COLUMN, CANOPY]}]),
        ):
            scene = Planner().generate("Широкий навес на колонне")
        upper, lower = [build(p, "render").vertices for p in scene.parts]
        self.assertGreater(upper[:, 2].min(), lower[:, 2].min())
        self.assertGreater(lower[:, 2].max(), upper[:, 2].min())

    def test_a_known_eleven_to_ten_width_ratio_remains_exact(self):
        specified = CANOPY | {"size": [0.11, 0.08, 0.04]}
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(json.dumps, [SUBJECT, {"parts": [COLUMN, specified]}]),
        ):
            scene = Planner().generate(
                "Колонна шириной 10 см, наверху площадка чуть шире, ровно 11 см."
            )
        self.assertEqual(scene.parts[1].size, (0.11, 0.08, 0.04))

    def test_a_monotonic_turned_component_with_a_terminal_pole_closes_to_the_axis(self):
        from collections import Counter

        from xtrifid.assembly import assemble

        part = COLUMN | {
            "name": "Рукоять",
            "shape": "lathe",
            "support": 0,
            "outline": [[0.6, 0], [0.48, 0.3], [0.3, 0.6], [0, 0.8]],
        }
        surface = assemble({"parts": [part]}).parts[0]
        self.assertIn((0, 0), surface.outline)
        edges = Counter(
            tuple(sorted((a, b)))
            for face in build(surface, "render").faces
            for a, b in zip(face, (*face[1:], face[0]))
        )
        self.assertEqual(set(edges.values()), {2})

    def test_width_comparison_on_the_floor_is_repaired_before_geometry(self):
        ambiguous = SUBJECT | {
            "components": [COVER | {"support": 0}, SUPPORT | {"width": "narrower"}]
        }
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(json.dumps, [ambiguous, SUBJECT, {"parts": [COLUMN, CANOPY]}]),
        ) as chat:
            scene = Planner().generate("Широкая площадка наверху колонны")
        self.assertEqual(chat.call_count, 3)
        self.assertEqual([p.name for p in scene.parts], ["Колонна", "Навес"])

    def test_a_supported_piece_cannot_land_entirely_beside_its_support(self):
        disjoint = CANOPY | {"center": [1, 2, 0]}
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(json.dumps, [SUBJECT, {"parts": [COLUMN, disjoint]}]),
        ):
            scene = Planner().generate("Собери указатель с площадкой на колонне")
        a, b = (build(p, "render").vertices for p in scene.parts)
        for axis in (0, 1):
            self.assertGreater(
                min(a[:, axis].max(), b[:, axis].max()), max(a[:, axis].min(), b[:, axis].min())
            )

    def test_roles_width_and_integral_connection_survive_reordered_geometry(self):
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(json.dumps, [SUBJECT, {"parts": [CANOPY, COLUMN]}]),
        ) as chat:
            scene = Planner().generate("Сделай широкий навес, соединённый с узкой колонной")
        self.assertEqual([p.name for p in scene.parts], ["Колонна", "Навес"])
        self.assertEqual(chat.call_count, 2)
        a, b = (build(p, "render").vertices for p in scene.parts)
        self.assertAlmostEqual(a[:, 2].min(), 0)
        self.assertGreater(np.ptp(b[:, 0]), 1.25 * np.ptp(a[:, 0]))
        self.assertGreater(a[:, 2].max(), b[:, 2].min())
        self.assertGreater(b[:, 2].max(), a[:, 2].max())

    def test_missing_named_component_is_reported_before_a_scene_is_installed(self):
        client = Planner()
        with (
            patch.object(
                client,
                "_chat",
                side_effect=map(
                    json.dumps, [SUBJECT, {"parts": [COLUMN]}, {"parts": [COLUMN]}]
                ),
            ) as chat,
            self.assertRaises(PlannerError),
        ):
            client.generate("Сделай вертикальную колонну с соединённым навесом")
        self.assertEqual(chat.call_count, 3)
