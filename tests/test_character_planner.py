import json
import unittest
from unittest.mock import patch

from test_model import spec

from xtrifid.planner import MAX_PROMPT, Planner, PlannerError


class CharacterPlanningTests(unittest.TestCase):
    def test_basic_explicit_physiques_work_offline_and_leave_garments_opt_in(self):
        examples = [
            ("Сделай мужчину", "male", False),
            ("женщина с плащом", "female", True),
            ("create a woman", "female", False),
            ("a man in a cloak", "male", True),
            ("человек", "neutral", False),
        ]
        with patch.object(Planner, "_chat", side_effect=AssertionError("unexpected chat")):
            for prompt, expected, cloak in examples:
                with self.subTest(prompt=prompt):
                    scene = Planner().generate(prompt)
                    self.assertEqual(scene.parts[0].physique, expected)
                    self.assertEqual(any(p.shape == "cloak" for p in scene.parts), cloak)

    def setUp(self):
        self.planner = Planner()
        self.character = {
            "height_m": 1.92,
            "build": "broad",
            "pose": "walking",
            "cloak": "ankle",
            "hood": True,
        }
        self.subject = {"subject": "character", "character": self.character, "accessories": []}

    def test_basic_descriptions_work_offline_and_respect_clothing(self):
        examples = [
            ("Сделай человека с плащом", True, False),
            ("ПОЖАЛУЙСТА, создай фигуру человека в плаще с капюшоном!", True, True),
            ("Сделай человека без плаща.", False, False),
            ("Create a person in a cloak without a hood", True, False),
            ("a human", False, False),
        ]
        with patch.object(self.planner, "_chat", side_effect=AssertionError("unexpected chat")):
            for prompt, cloak, hood in examples:
                with self.subTest(prompt=prompt):
                    result = self.planner.generate(prompt, detail="low")
                    names = {p.name for p in result.parts}
                    self.assertEqual("Плащ" in names, cloak)
                    self.assertEqual("Капюшон" in names, hood)
                    self.assertEqual(result.detail, "low")

    def test_unconsumed_modifiers_reach_the_model_and_shape_the_character(self):
        from xtrifid.character import build_character

        prompt = "Покажи широкоплечего путника, шагающего в плаще до щиколоток с капюшоном, рост 192 см."
        with patch.object(
            self.planner,
            "_chat",
            return_value=json.dumps(self.subject),
        ) as chat:
            result = self.planner.generate(prompt, "high")
        chat.assert_called_once()
        messages, schema = chat.call_args.args
        self.assertIn(prompt, messages[-1]["content"])
        self.assertNotIn('"parts"', json.dumps(schema))
        self.assertLessEqual(chat.call_args.kwargs["num_predict"], 192)
        self.assertEqual(result, build_character(self.character, "high"))

    def test_character_can_have_additional_independently_editable_parts(self):
        prompt = "Сделай человека с плащом и маленькой сумкой через плечо"
        bag = spec() | dict(name="Сумка", size=[0.18, 0.07, 0.20], support=-1)
        with patch.object(
            self.planner,
            "_chat",
            side_effect=map(
                json.dumps,
                [
                    self.subject | {"accessories": ["Маленькая сумка через плечо"]},
                    {"parts": [bag]},
                ],
            ),
        ) as chat:
            scene = self.planner.generate(prompt, "medium")
        self.assertIn(prompt, chat.call_args.args[0][-1]["content"])
        self.assertIn("Кисть левая", chat.call_args.args[0][-1]["content"])
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(scene.parts[-1].name, "Сумка")
        self.assertEqual(scene.parts[-1].size, (0.18, 0.07, 0.2))
        self.assertIn("Плащ", [p.name for p in scene.parts])

    def test_human_subject_requires_an_anatomical_recipe(self):
        from xtrifid.character import build_character

        cube = spec() | {"support": 0}
        answers = [
            {"subject": "character", "parts": [cube]},
            self.subject,
        ]
        with patch.object(self.planner, "_chat", side_effect=map(json.dumps, answers)) as chat:
            result = self.planner.generate("Покажи шагающего путешественника")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result, build_character(self.character))

    def test_an_object_for_a_person_stays_an_object_and_keeps_detail(self):
        prompt = "Сделай стул для человека в плаще"
        raw = {"parts": [spec() | {"name": "Сиденье", "support": 0}]}
        with patch.object(
            self.planner, "_chat", side_effect=map(json.dumps, [{"subject": "object"}, raw])
        ) as chat:
            scene = self.planner.generate(prompt, "low")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual([p.name for p in scene.parts], ["Сиденье"])
        self.assertEqual(scene.detail, "low")

    def test_editing_anatomy_preserves_cross_sections_and_keeps_the_request_small(self):
        scene = self.planner.generate("человек в плаще")
        torso = next(i for i, p in enumerate(scene.parts) if p.shape == "human")
        selected = scene.parts[torso]
        width, depth, height = selected.size
        with patch.object(
            self.planner,
            "_chat",
            return_value=json.dumps({"size": [width * 1.1, depth, height]}),
        ) as chat:
            edited = self.planner.edit("Сделай туловище на 10% шире", scene, torso)
        messages, schema = chat.call_args.args
        self.assertNotIn('"sections"', messages[-1]["content"])
        self.assertIn("loft", schema["properties"]["shape"]["enum"])
        self.assertIn(selected.name, messages[-1]["content"])
        self.assertEqual(edited.sections, selected.sections)
        self.assertEqual(edited.size, (width * 1.1, depth, height))

    def test_local_anatomy_edit_preserves_an_earlier_region_scale(self):
        scene = self.planner.generate("человек в плаще")
        with patch.object(self.planner, "_chat", return_value='{"body_scales":{"hands":1.1}}'):
            first = self.planner.edit("Увеличь кисти на 10 процентов", scene, 0)
        with patch.object(
            self.planner, "_chat", return_value='{"body_scales":{"ears":0.9}}'
        ) as chat:
            second = self.planner.edit(
                "Уменьши уши на 10 процентов", scene.with_part(0, first), 0
            )
        self.assertIn("ears", chat.call_args.args[1]["properties"]["body_scales"]["properties"])
        self.assertEqual(second.data()["body_scales"]["hands"], 1.1)
        self.assertEqual(second.data()["body_scales"]["ears"], 0.9)
        self.assertEqual(second.size, first.size)

    def test_explicit_anatomical_proportions_survive_generation(self):
        character = self.character | {
            "cloak": "none",
            "hood": False,
            "body_scales": {"ears": 1.1},
        }
        with patch.object(
            self.planner,
            "_chat",
            return_value=json.dumps(self.subject | {"character": character}),
        ) as chat:
            scene = self.planner.generate(
                "Сделай шагающего человека с ушами на 10 процентов больше обычного"
            )
        chat.assert_called_once()
        self.assertEqual(scene.parts[0].data()["body_scales"]["ears"], 1.1)
        self.assertEqual(len(scene.parts), 1)

    def test_pose_edit_preserves_stature_and_recomputes_the_arm_span(self):
        from xtrifid.character import build_character

        scene = self.planner.generate("человек")
        with patch.object(self.planner, "_chat", return_value='{"pose":"arms_out"}'):
            edited = self.planner.edit("Разведи руки в стороны", scene, 0)
        target = build_character({"pose": "arms_out"}).parts[0]
        for actual, expected in zip(edited.size + edited.center, target.size + target.center):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_part_selection_is_rejected_before_an_edit_request(self):
        scene = self.planner.generate("человек")
        with patch.object(self.planner, "_chat", side_effect=AssertionError("unexpected chat")):
            for index in (-1, True, len(scene.parts)):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.planner.edit("Увеличь на 5%", scene, index)

    def test_invalid_character_and_overbudget_assembly_are_rejected(self):
        from xtrifid.model import MAX_PARTS

        for raw in (
            {},
            self.subject | {"character": None},
            self.subject | {"character": {}},
            self.subject | {"character": self.character | {"cloak": "mirrored"}},
            self.subject | {"accessories": [False]},
            self.subject | {"accessories": [""]},
        ):
            with (
                self.subTest(raw=str(raw)[:90]),
                patch.object(self.planner, "_chat", return_value=json.dumps(raw)) as chat,
                self.assertRaises(PlannerError),
            ):
                self.planner.generate("Фигура путешественника со сложными деталями")
            self.assertEqual(chat.call_count, 2)
        over = {"parts": [spec() | {"support": 0}] * MAX_PARTS}
        answers = [
            self.subject | {"accessories": ["рюкзак"]},
            over,
            over,
        ]
        with (
            patch.object(self.planner, "_chat", side_effect=map(json.dumps, answers)),
            self.assertRaises(PlannerError),
        ):
            self.planner.generate("Путник с рюкзаком")

    def test_bad_detail_and_oversized_prompts_do_not_call_the_model(self):
        with patch.object(self.planner, "_chat", side_effect=AssertionError("unexpected chat")):
            with self.assertRaises(ValueError):
                self.planner.generate("человек в плаще", "unlimited")
            with self.assertRaises(ValueError):
                self.planner.generate("человек" + " " * MAX_PROMPT)
