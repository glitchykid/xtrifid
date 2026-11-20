import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cairo
import numpy as np

from xtrifid.geometry import build
from xtrifid.planner import Planner
from xtrifid.storage import load_project, save_project


def drawing(path, *, hole=False, side=False, opaque=False):
    image = cairo.ImageSurface(cairo.FORMAT_ARGB32, 160, 160)
    c = cairo.Context(image)
    if opaque:
        c.set_source_rgb(0.95, 0.94, 0.91)
        c.paint()
    c.set_source_rgb(0.36, 0.23, 0.15)
    c.rectangle(50 if side else 20, 20, 60 if side else 120, 120)
    if hole:
        c.new_sub_path()
        c.arc(80, 80, 26, 0, 2 * np.pi)
        c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    c.fill()
    image.write_to_png(str(path))


class ReferenceTests(unittest.TestCase):
    def test_input_memory_is_checked_before_processing_the_compressed_data(self):
        from xtrifid.reference import read_image
        from xtrifid.resources import Capacity

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.png"
            drawing(path)
            with (
                patch("xtrifid.reference.capacity", return_value=Capacity(0, 1)),
                patch("xtrifid.reference._dimensions") as dimensions,
                self.assertRaisesRegex(ValueError, "памяти"),
            ):
                read_image(path)
            dimensions.assert_not_called()

    def test_two_photos_fit_one_vision_input_and_preserve_both_views(self):
        import base64

        from xtrifid.reference import GdkPixbuf, _encoded_photo, pixels

        with tempfile.TemporaryDirectory() as tmp:
            front, side = Path(tmp) / "front.png", Path(tmp) / "side.png"
            drawing(front, hole=True)
            drawing(side, side=True)
            loader = GdkPixbuf.PixbufLoader.new_with_type("jpeg")
            loader.write(base64.b64decode(_encoded_photo(front, side)))
            loader.close()
            raster = pixels(loader.get_pixbuf())
            self.assertEqual(raster.shape[:2], (160, 328))
            self.assertGreater(raster[80, 80].min(), 200)
            self.assertLess(raster[80, 248].max(), 140)

    def test_rounded_sections_are_a_saved_explicit_choice(self):
        from xtrifid.model import Part
        from xtrifid.reference import reference_scene

        with tempfile.TemporaryDirectory() as tmp:
            front = Path(tmp) / "front.png"
            drawing(front)
            project = Path(tmp) / "round.xtri"
            self.assertFalse(reference_scene(front).parts[0].rounded)
            scene = reference_scene(front, rounded=True)
            save_project(project, scene, "game")
            self.assertTrue(load_project(project)[0].parts[0].rounded)
            with self.assertRaises(ValueError):
                Part.parse(scene.parts[0].data() | {"rounded": "yes"})

    def test_two_silhouettes_encode_the_hole_and_dimensions_without_source_paths(self):
        from xtrifid.reference import reference_scene

        with tempfile.TemporaryDirectory() as tmp:
            front, side = Path(tmp) / "front.png", Path(tmp) / "side.png"
            drawing(front, hole=True)
            drawing(side, side=True, opaque=True)
            scene = reference_scene(front, side, height_m=0.4, detail="high")
            p = scene.parts[0]
            self.assertEqual((p.shape, len(p.masks)), ("reference", 2))
            np.testing.assert_allclose(p.size, (0.4, 0.2, 0.4), atol=0.004)
            m = p.masks[0]
            self.assertEqual(m[len(m) // 2][len(m[0]) // 2], "0")
            target = Path(tmp) / "object.xtri"
            save_project(target, scene, "render")
            front.unlink()
            side.unlink()
            restored, _ = load_project(target)
            self.assertEqual(restored, scene)
            mesh = build(restored.parts[0], "render", "high")
            np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), p.size, atol=0.004)

    def test_single_uses_depth_and_rejects_ambiguous_or_excessive_input(self):
        from xtrifid.reference import reference_scene

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subject.png"
            drawing(path)
            self.assertAlmostEqual(reference_scene(path, depth_m=0.17).parts[0].size[1], 0.17)
            with self.assertRaises(ValueError):
                reference_scene(path, height_m=float("nan"))
            blank = cairo.ImageSurface(cairo.FORMAT_ARGB32, 32, 32)
            blank.write_to_png(str(path))
            with self.assertRaises(ValueError):
                reference_scene(path)
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                reference_scene(path)
            os.truncate(path, 20 * 1024 * 1024)
            with self.assertRaises(ValueError):
                reference_scene(path)

    def test_photograph_mode_supplies_actual_image_bytes_and_normalizes_height(self):
        from test_model import spec

        from xtrifid.model import Scene
        from xtrifid.reference import photo_scene

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "look.png"
            drawing(image)
            source = Scene.parse({"parts": [spec("box")]})
            planner = Planner()
            with (
                patch.object(
                    planner,
                    "_chat",
                    side_effect=[
                        json.dumps(
                            {"description": "A rectangular ceramic piece with a cutout."}
                        ),
                        json.dumps({"route": "semantic", "parts": []}),
                    ],
                ) as vision,
                patch.object(planner, "generate", return_value=source) as plan,
            ):
                result = photo_scene(
                    planner,
                    image,
                    None,
                    height_m=0.21,
                    detail="medium",
                    prompt="With the same outline",
                )
            images = [
                img
                for message in vision.call_args_list[0].args[0]
                for img in message.get("images", [])
            ]
            self.assertEqual(len(images), 1)
            self.assertGreater(len(images[0]), 100)
            self.assertIn("ceramic", str(plan.call_args))
            self.assertEqual(result.parts[0].size[2], 0.21)
