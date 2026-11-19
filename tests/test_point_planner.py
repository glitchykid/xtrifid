import numpy as np
from test_model import spec
from test_planner import PlannerHTTPCase

from xtrifid.geometry import build
from xtrifid.model import Scene
from xtrifid.planner import MAX_PROMPT, PlannerError


def change(op, value, unit):
    return {"op": op, "value": value, "unit": unit}


class PointPlannerTests(PlannerHTTPCase):
    def setUp(self):
        super().setUp()
        self.scene = Scene.parse(
            {
                "parts": [
                    spec("sphere") | {"name": "Дальняя сфера"},
                    spec("extrusion", outline=[[-1, -1], [1, -1], [1, 1], [-1, 1]]),
                ]
            }
        )

    def test_transmitted_target_and_delta_only_move_one_corner(self):
        part = self.scene.parts[1]
        original = build(part, "render").vertices.copy()
        self.answer({"x": change("add", 2, "cm"), "y": None})
        result = self.client.edit_point("Сдвинь по X на 2 см", self.scene, 1, 0)
        expected = original.copy()
        expected[[0, 4], 0] += 0.02
        np.testing.assert_allclose(
            build(result, "render").vertices, expected, rtol=0, atol=1e-14
        )
        np.testing.assert_array_equal(build(part, "render").vertices, original)
        request = self.calls[0]
        schema = request["format"]
        self.assertEqual(set(schema["required"]), {"x", "y"})
        self.assertFalse(schema["additionalProperties"])
        user_text = request["messages"][-1]["content"]
        self.assertIn("Основание", user_text)
        self.assertIn("Сдвинь по X на 2 см", user_text)
        self.assertNotIn("Дальняя сфера", user_text)

    def test_radial_delta_changes_only_one_ring(self):
        scene = Scene.parse({"parts": [spec("tube", inner_diameter=1)]})
        part = scene.parts[0]
        before = build(part, "render").vertices
        self.answer({"radius": change("add", -5, "cm"), "z": None})
        updated = self.client.edit_point("Уменьши радиус на 5 см", scene, 0, 0)
        after = build(updated, "render").vertices
        self.assertAlmostEqual(after[0, 0], before[0, 0] - 0.05)
        np.testing.assert_allclose(after[:, 2], before[:, 2], rtol=0, atol=1e-14)
        np.testing.assert_allclose(after[48:], before[48:], rtol=0, atol=1e-14)
        self.assertIn(
            "radius", " ".join(m["content"] for m in self.calls[0]["messages"]).lower()
        )

    def test_set_and_add_with_different_units_cross_zero(self):
        part = self.scene.parts[1]
        original = build(part, "render").vertices.copy()
        self.answer({"x": change("set", 17, "mm"), "y": change("add", 3, "cm")})
        result = self.client.edit_point(
            "Поставь X на 17 мм, подними Y на 3 см", self.scene, 1, 0
        )
        self.assertEqual(len(self.calls), 1)
        expected = original.copy()
        expected[[0, 4], 0] = 0.017
        expected[[0, 4], 1] += 0.03
        np.testing.assert_allclose(
            build(result, "render").vertices, expected, rtol=0, atol=1e-14
        )

    def test_repair_uses_original_point_and_may_return_no_change(self):
        # The first move collides with the adjacent corner.
        self.answer({"x": change("add", 2, "m"), "y": None})
        self.answer({"x": None, "y": None})
        result = self.client.edit_point("Оставь на месте", self.scene, 1, 0)
        self.assertIs(result, self.scene.parts[1])
        self.assertEqual(len(self.calls), 2)

    def test_invalid_response_is_bounded_and_original_is_intact(self):
        part = self.scene.parts[1]
        for answer in (
            None,
            [0, 0],
            {},
            {"x": None},
            {"x": None, "y": None, "selected": 3},
            {"x": change("set", True, "m"), "y": None},
            {"x": change("set", float("nan"), "m"), "y": None},
            {"x": change("set", float("inf"), "m"), "y": None},
            {"x": change("multiply", 2, "m"), "y": None},
            {"x": change("add", 1, "km"), "y": None},
            {"x": change("set", 2, "m") | {"target": 7}, "y": None},
        ):
            with self.subTest(answer=answer):
                self.answer(answer)
                self.answer(answer)
                with self.assertRaises(PlannerError):
                    self.client.edit_point("Измени точку", self.scene, 1, 0)
                self.assertFalse(self.answers)
                self.assertIs(part, self.scene.parts[1])

    def test_invalid_targets_and_input_issue_no_requests(self):
        for indices in ((0, 0), (-1, 0), (2, 0), (True, 0), (1, -1), (1, 4), (1, True)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.client.edit_point("Сдвинь", self.scene, *indices)
        for prompt in (None, " ", "x" * (MAX_PROMPT + 1)):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                self.client.edit_point(prompt, self.scene, 1, 0)
        self.assertFalse(self.calls)
