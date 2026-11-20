import json
import unittest
from unittest.mock import patch

from xtrifid.geometry import build
from xtrifid.model import DETAILS, Scene
from xtrifid.planner import Planner, PlannerError

FIELDS = ("height", "radius_x", "radius_y", "offset_right", "offset_back")
SHELL = {
    "name": "Оболочка",
    "shape": "loft",
    "size": [0.07, 0.12, 0.04],
    "center": [0, 0, 0],
    "rotation": [25, -13, 8],
    "support": 0,
    "sections": [
        [0, 0.4, 0.48, -0.08, 0],
        [0.3, 0.5, 0.5, 0, 0.03],
        [0.8, 0.35, 0.23, 0.09, 0.14],
        [1, 0, 0, 0.1, 0.19],
    ],
}
BUTTON = {
    "name": "Кнопка",
    "shape": "cylinder",
    "size": [0.01, 0.01, 0.002],
    "center": [0, 0, 0],
    "rotation": [0, 0, 0],
    "support": 1,
}


class FreeformTests(unittest.TestCase):
    def test_explicit_diameter_yields_a_circular_face_with_world_ordered_sizes(self):
        disk = BUTTON | {
            "size": [0.2, 0.005, 0.2],
            "world_size": [0.2, 0.005, 0.2],
            "axis": "width_x",
            "support": 0,
        }
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(json.dumps, [{"subject": "object"}, {"parts": [disk]}]),
        ):
            scene = Planner().generate(
                "Вертикальное круглое стекло диаметром 20 см и толщиной 5 мм, плоскость XZ."
            )
        glass = scene.parts[0]
        self.assertEqual(glass.size, (0.2, 0.2, 0.005))
        for x, _, z in build(glass, "render").vertices:
            self.assertLessEqual(
                (x - glass.center[0]) ** 2 + (z - glass.center[2]) ** 2, 0.1**2 + 1e-9
            )

    def test_explicit_signed_axis_survives_an_unspecified_model_lean(self):
        subject = {
            "subject": "object",
            "form": "varying_sections",
            "bottom_end": "flat",
            "top_end": "rounded",
            "top_lean": "as_drawn",
        }
        fitting = SHELL | {
            "rotation": [0, 0, 0],
            "sections": [[0, 0.5, 0.35, 0, 0], [0.5, 0.5, 0.38, 0, 0], [1, 0.5, 0.42, 0, 0.05]],
        }
        with patch.object(
            Planner, "_chat", side_effect=map(json.dumps, [subject, {"parts": [fitting]}])
        ):
            scene = Planner().generate(
                "Оболочка. Высшая точка смещена назад (+Y) от середины корпуса."
            )
        rows = scene.parts[0].sections
        middle = (
            min(y - ry for _, _, ry, _, y in rows) + max(y + ry for _, _, ry, _, y in rows)
        ) / 2
        self.assertGreater(rows[-1][4] - middle, 0.01)

    def test_direction_from_text_is_preserved_when_section_centres_remain_symmetric(self):
        from xtrifid.assembly import assemble

        base = SHELL | {
            "sections": [[0, 0.5, 0.5, 0, 0], [0.5, 0.4, 0.4, 0, 0], [1, 0, 0, 0, 0]],
            "bottom_end": "flat",
            "top_end": "rounded",
            "top_lean": "back",
        }
        scene = assemble({"parts": [base]})
        rows = scene.parts[0].sections
        self.assertGreater(rows[-1][4], rows[0][4])
        self.assertEqual(rows[0][1:], (0.5, 0.5, 0, 0))

    def test_world_dimensions_preserve_a_floor_plate_when_axis_labels_disagree(self):
        from xtrifid.assembly import assemble

        plate = BUTTON | {
            "size": [0.14, 0.09, 0.01],
            "world_size": [0.14, 0.09, 0.01],
            "axis": "width_x",
            "support": 0,
        }
        scene = assemble({"parts": [plate]})
        mesh = build(scene.parts[0], "game").vertices
        self.assertAlmostEqual(mesh[:, 2].max(), 0.01)
        self.assertAlmostEqual(mesh[:, 0].max() - mesh[:, 0].min(), 0.14)

    def test_rounded_tips_refine_a_supplied_pole_and_flat_ends_require_a_ring(self):
        body = SHELL | {
            "bottom_end": "flat",
            "top_end": "rounded",
            "sections": [[0, 0.5, 0.5, 0, 0], [0.5, 0.45, 0.4, 0, 0.1], [1, 0, 0, 0, 0.2]],
        }
        from xtrifid.assembly import assemble

        rows = assemble({"parts": [body]}).parts[0].sections
        self.assertGreater(len(rows), 3)
        self.assertGreater(rows[-2][4], rows[1][4])
        with self.assertRaises(ValueError):
            assemble({"parts": [body | {"top_end": "flat"}]})

    def test_model_routes_a_single_offset_silhouette_to_sections(self):
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(
                json.dumps,
                [
                    {
                        "subject": "object",
                        "form": "varying_sections",
                        "bottom_end": "flat",
                        "top_end": "rounded",
                    },
                    {"parts": [SHELL | {"bottom_end": "rounded", "top_end": "flat"}]},
                ],
            ),
        ) as chat:
            scene = Planner().generate("Асимметричная скорлупа, смещённая к одному краю")
        props = chat.call_args.args[1]["properties"]["parts"]
        self.assertEqual(props["maxItems"], 1)
        self.assertEqual(props["items"]["properties"]["shape"]["enum"], ["loft"])
        self.assertEqual(scene.parts[0].shape, "loft")
        self.assertEqual(scene.parts[0].sections[0], tuple(SHELL["sections"][0]))
        self.assertGreater(len(scene.parts[0].sections), len(SHELL["sections"]))

    def test_a_vertical_rod_and_front_facing_disk_use_the_requested_world_axes(self):
        rod = BUTTON | {
            "size": [0.01, 0.01, 0.15],
            "rotation": [90, 0, 0],
            "axis": "vertical_z",
            "support": 0,
        }
        disk = BUTTON | {"size": [0.2, 0.2, 0.005], "axis": "depth_y", "support": 1}
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(
                json.dumps,
                [
                    {"subject": "object"},
                    {"parts": [rod, disk]},
                ],
            ),
        ):
            scene = Planner().generate("Вертикальная стойка держит диск в плоскости XZ")
        a, b = [build(p, "game", "low").vertices for p in scene.parts]
        self.assertAlmostEqual(a[:, 2].max(), 0.15)
        self.assertAlmostEqual(b[:, 2].min(), 0.15)
        self.assertAlmostEqual(b[:, 2].max(), 0.35)

    def test_requested_round_end_converges_to_a_single_pole_above_a_flat_base(self):
        part = SHELL | {
            "rotation": [0, 0, 0],
            "bottom_end": "flat",
            "top_end": "rounded",
            "sections": [
                [0, 0.5, 0.5, 0, 0],
                [0.4, 0.5, 0.45, 0, 0.05],
                [1, 0.45, 0.4, 0, 0.14],
            ],
        }
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(
                json.dumps,
                [
                    {"subject": "object"},
                    {"parts": [part]},
                ],
            ),
        ):
            scene = Planner().generate("Округлый верх и плоское дно")
        rows = scene.parts[0].sections
        self.assertEqual(rows[-1][1:3], (0, 0))
        self.assertTrue(rows[-3][1] > rows[-2][1] > 0)
        self.assertEqual(rows[0], (0, 0.5, 0.5, 0, 0))

    def test_generated_outline_can_include_redundant_points_on_a_straight_edge(self):
        part = BUTTON | {
            "shape": "extrusion",
            "support": 0,
            "outline": [[-1, -1], [0, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]],
        }
        with patch.object(
            Planner,
            "_chat",
            side_effect=map(
                json.dumps,
                [
                    {"subject": "object"},
                    {"parts": [part]},
                ],
            ),
        ) as chat:
            scene = Planner().generate("Плоская прямоугольная заготовка")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(len(scene.parts[0].outline), 4)
        self.assertEqual(len(build(scene.parts[0], "game").faces), 12)
        with self.assertRaises(ValueError):
            Scene.parse({"parts": [part]})

    def test_asymmetric_shell_can_support_a_part_at_every_detail_level(self):
        generated = SHELL | {"sections": [dict(zip(FIELDS, row)) for row in SHELL["sections"]]}
        planner = Planner()
        counts = []
        for detail in DETAILS:
            with (
                self.subTest(detail=detail),
                patch.object(
                    planner,
                    "_chat",
                    side_effect=map(
                        json.dumps,
                        [
                            {"subject": "object"},
                            {"parts": [generated, BUTTON]},
                        ],
                    ),
                ) as chat,
            ):
                scene = planner.generate(
                    "Наклонная оболочка выгнута назад, кнопка поверх.", detail
                )
                messages, schema = chat.call_args.args
                self.assertIn(f"Mesh detail: {detail}", messages[0]["content"])
                choices = schema["properties"]["parts"]["items"]["oneOf"]
                loft = next(c for c in choices if "sections" in c["required"])
                self.assertEqual(
                    set(loft["properties"]["sections"]["items"]["required"]), set(FIELDS)
                )
                shell, button = (build(p, "render", detail).vertices for p in scene.parts)
                counts.append(len(shell))
                self.assertAlmostEqual(shell[:, 2].min(), 0)
                self.assertAlmostEqual(shell[:, 2].max(), button[:, 2].min())
                self.assertEqual(scene.parts[0].size, tuple(SHELL["size"]))
                self.assertEqual(scene.parts[0].sections, tuple(map(tuple, SHELL["sections"])))
        self.assertTrue(counts[0] < counts[1] < counts[2])

    def test_invalid_loft_cannot_trigger_mesh_construction(self):
        malformed = SHELL | {
            "sections": [[0, 0.5, 0.5, 0, 0], [0.6, 0, 0, 0, 0], [1, 0.1, 0.2, 0, 0]]
        }
        with (
            patch.object(
                Planner,
                "_chat",
                side_effect=map(
                    json.dumps,
                    [
                        {"subject": "object"},
                        {"parts": [malformed]},
                        {"parts": [malformed]},
                    ],
                ),
            ) as chat,
            self.assertRaises(PlannerError),
        ):
            Planner().generate("Корпус мыши")
        self.assertEqual(chat.call_count, 3)

    def test_only_selected_surface_supplies_its_cross_sections_for_a_local_edit(self):
        scene = Scene.parse({"parts": [SHELL, SHELL | {"name": "Крышка"}]})
        changed = [*SHELL["sections"][:-1], [1, 0, 0, 0.1, 0.25]]
        with patch.object(
            Planner, "_chat", return_value=json.dumps({"sections": changed})
        ) as chat:
            result = Planner().edit("Смести верхушку чуть назад", scene, 0)
        messages, schema = chat.call_args.args
        self.assertEqual(messages[-1]["content"].count('"sections"'), 1)
        self.assertIn("sections", schema["properties"])
        self.assertEqual(result.sections[-1][-1], 0.25)
        self.assertEqual(result.size, scene.parts[0].size)
        self.assertEqual(scene.parts[1].sections[-1][-1], 0.19)
