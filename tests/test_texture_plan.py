import unittest
from dataclasses import replace

import numpy as np
from test_planner import PlannerHTTPCase

from xtrifid.ollama import PlannerError
from xtrifid.texture_plan import text_to_texture
from xtrifid.textures import TextureSpec, pixels, texture_from_prompt


class TexturePlannerTests(PlannerHTTPCase):
    def test_arbitrary_description_uses_bounded_recipe_and_keeps_user_raster_choices(self):
        prompt = "Глазурь полярного фарфора с кобальтовой росписью чешуйками"
        answer = {
            "material": "paint",
            "motif": "scales",
            "color_a": "#e9eaed",
            "color_b": "#f4f4ff",
            "accent": "#0d3678",
            "scale": 1.6,
        }
        self.answer(answer)
        result = text_to_texture(self.client, prompt, 31, 128)
        self.assertEqual(
            (result.seed, result.resolution, result.motif, result.accent),
            (31, 128, "scales", "#0d3678"),
        )
        request = self.calls[0]
        self.assertIn(prompt, request["messages"][-1]["content"])
        self.assertLessEqual(request["options"]["num_predict"], 400)
        pixels_with = pixels(result)
        plain = pixels(replace(result, motif="none"))
        self.assertGreater(
            (np.abs(pixels_with.astype(int) - plain.astype(int)).max(axis=2) > 24).mean(), 0.07
        )
        self.assertEqual(TextureSpec.parse(result.data()), result)

    def test_untrusted_recipe_cannot_change_budget_or_write_paths(self):
        bad = {"material": "paint", "motif": "spots", "accent": "/tmp/here", "resolution": 4096}
        self.answer(bad)
        self.answer(bad)
        with self.assertRaises(PlannerError):
            text_to_texture(self.client, "Перламутровые снежинки", 19, 128)
        self.assertEqual(len(self.calls), 2)
        with self.assertRaises(ValueError):
            text_to_texture(self.client, "test", 19, 999999)
        self.assertEqual(len(self.calls), 2)

    def test_missing_accent_is_retried_for_an_ai_recipe(self):
        valid = {
            "material": "paint",
            "motif": "scales",
            "color_a": "#000000",
            "color_b": "#eeeeee",
            "accent": "#3456a1",
            "scale": 1,
        }
        self.answer({**valid, "accent": None})
        self.answer(valid)
        result = text_to_texture(self.client, "Чешуйки синим пигментом", 6, 128)
        self.assertEqual(result.accent, "#3456a1")
        self.assertEqual(len(self.calls), 2)


class MotifTests(unittest.TestCase):
    def test_different_motifs_paint_distinct_visible_pixels_and_repeat_with_fixed_seed(self):
        base = texture_from_prompt("синяя керамика в белую полоску", seed=12, resolution=128)
        self.assertEqual((base.material, base.motif), ("paint", "stripes"))
        self.assertIsNotNone(base.accent)
        patterns = []
        for motif in ("none", "spots", "stripes", "scales"):
            spec = replace(base, motif=motif)
            a = pixels(spec)
            np.testing.assert_array_equal(a, pixels(spec))
            patterns.append(a.tobytes())
        self.assertEqual(len(set(patterns)), 4)
