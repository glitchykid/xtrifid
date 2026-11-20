import hashlib
import struct
import tempfile
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import cairo
import numpy as np

from xtrifid.character import build_character
from xtrifid.geometry import Mesh, build
from xtrifid.model import Part, Scene
from xtrifid.storage import export_obj, load_project, save_project
from xtrifid.textures import (
    TextureSpec,
    pixels,
    texture_from_prompt,
    texture_working_bytes,
    uv_layout,
)


def box(name="Sample"):
    return Part.parse(
        dict(shape="box", name=name, size=[1, 2, 3], center=[0, 0, 1.5], rotation=[0, 0, 0])
    )


class TextureTests(unittest.TestCase):
    def test_bounded_materials_are_repeatable_and_palette_seed_and_scale_affect_raster(self):
        patterns = []
        for material in ("wood", "skin", "stone", "fabric", "metal"):
            spec = TextureSpec.parse(
                {
                    "material": material,
                    "resolution": 128,
                    "color_a": "#161400",
                    "color_b": "#e96000",
                }
            )
            image = pixels(spec)
            self.assertEqual((image.shape, image.dtype), ((128, 128, 3), np.uint8))
            self.assertEqual(image[:, :, 2].max(), 0)
            self.assertGreater(image[:, :, 0].std(), 5)
            np.testing.assert_array_equal(image, pixels(TextureSpec.parse(spec.data())))
            self.assertFalse(np.array_equal(image, pixels(replace(spec, seed=9))))
            self.assertFalse(np.array_equal(image, pixels(replace(spec, scale=2))))
            patterns.append(image.tobytes())
        self.assertEqual(len(set(patterns)), 5)
        for invalid in (
            {},
            {"material": []},
            {"material": "wood", "seed": True},
            {"material": "wood", "resolution": 8192},
            {"material": "wood", "scale": float("nan")},
            {"material": "wood", "color_a": "white"},
            {"material": "wood", "other": "x"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TextureSpec.parse(invalid)

    def test_offline_description_and_resource_estimate(self):
        spec = texture_from_prompt("синяя ткань в мелкое плетение", seed=4, resolution=128)
        self.assertEqual(spec.material, "fabric")
        self.assertEqual(spec.seed, 4)
        self.assertNotEqual(spec.color_a, TextureSpec.parse({"material": "fabric"}).color_a)
        with self.assertRaises(ValueError):
            texture_from_prompt("неизвестная поверхность")
        part = replace(box(), texture=spec)
        self.assertGreater(texture_working_bytes([part]), spec.resolution**2 * 3)
        self.assertEqual(texture_working_bytes([part]), texture_working_bytes([part, part]))
        self.assertEqual(texture_working_bytes([box()]), 0)

    def test_uv_has_nondegenerate_triangles_and_follows_local_space_including_human(self):
        for part in (box(), build_character({}).parts[0]):
            a = build(part, "render", "low")
            uv, faces = uv_layout(a, part)
            self.assertLessEqual(np.abs(uv).max(), 1 + 1e-6)
            self.assertEqual([len(f) for f in faces], [len(f) for f in a.faces])
            self.assertTrue(np.isfinite(uv).all())
            tri = uv[[[f[0], f[i], f[i + 1]] for f in faces for i in range(1, len(f) - 1)]]
            u, v = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            self.assertGreater(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).min(), 1e-16)
            rotated = replace(part, rotation=(15, 37, -12), center=(1, -2, 1))
            uv2, faces2 = uv_layout(build(rotated, "render", "low"), rotated)
            for a, b in zip(faces[:20], faces2[:20]):
                np.testing.assert_allclose(uv[list(a)], uv2[list(b)], atol=1e-6)

    def test_cylindrical_side_has_one_unbroken_wrap_and_caps_have_area(self):
        part = replace(box(), shape="cylinder")
        mesh = build(part, "render", "medium")
        uv, uv_faces = uv_layout(mesh, part)
        widths = []
        for face, indices in zip(mesh.faces, uv_faces):
            coords = uv[list(indices)]
            if np.ptp(mesh.vertices[list(face), 2]) > 0.1:
                widths.append(np.ptp(coords[:, 0]))
                self.assertAlmostEqual(np.ptp(coords[:, 1]), 1, places=6)
        self.assertAlmostEqual(sum(widths), 1, places=6)
        self.assertLess(max(widths), 0.1)

    def test_nonplanar_cloth_fold_maps_both_sides_of_the_quad(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)
        mesh = Mesh(points, ((0, 1, 2, 3),), (True,))
        uv, faces = uv_layout(mesh, box())
        a, b, c, d = uv[list(faces[0])]
        for first, second in ((b - a, c - a), (c - a, d - a)):
            self.assertGreater(abs(first[0] * second[1] - first[1] * second[0]), 1e-4)

    def test_obj_is_published_after_png_and_mtl_and_preserves_untextured_faces(self):
        texture = TextureSpec.parse({"material": "wood", "resolution": 128})
        one = replace(box("Frame"), texture=texture)
        scene = Scene((one, box("Clay")))
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            saved = folder / "example.xtri"
            obj = folder / "a name.obj"
            save_project(saved, scene, "render")
            self.assertEqual(load_project(saved), (scene, "render"))
            export_obj(obj, scene, "render")
            old = obj.read_bytes()
            lines = old.decode().splitlines()
            mtl = folder / next(line[7:] for line in lines if line.startswith("mtllib "))
            image = mtl.parent / next(
                line[7:] for line in mtl.read_text().splitlines() if line.startswith("map_Kd ")
            )
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            decoded = cairo.ImageSurface.create_from_png(str(image))
            self.assertEqual((decoded.get_width(), decoded.get_height()), (128, 128))
            word = struct.unpack_from(
                "=I", decoded.get_data(), 17 * decoded.get_stride() + 31 * 4
            )[0]
            self.assertEqual(
                ((word >> 16) & 255, (word >> 8) & 255, word & 255),
                tuple(pixels(texture)[17, 31]),
            )
            faces = [line for line in lines if line.startswith("f ")]
            self.assertEqual(Counter("/" in line for line in faces), {True: 6, False: 6})
            export_obj(obj, scene, "render")
            self.assertEqual(obj.read_bytes(), old)
            changed = Scene((replace(one, texture=replace(texture, seed=17)),))
            with (
                patch("xtrifid.storage.os.replace", side_effect=OSError("full")),
                self.assertRaises(OSError),
            ):
                export_obj(obj, changed, "render")
            self.assertEqual(obj.read_bytes(), old)
            self.assertTrue(mtl.exists() and image.exists())

    def test_concurrent_exports_publish_only_complete_sets_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            target = folder / "shared.obj"
            barrier = Barrier(3)

            def run(seed):
                texture = TextureSpec.parse(
                    {"material": "stone", "seed": seed, "resolution": 128}
                )
                scene = Scene((replace(box(), texture=texture),))
                barrier.wait(timeout=3)
                export_obj(target, scene, "game")

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(run, (2, 2, 5)))
            libraries = list(folder.glob("xtrifid-materials-*/surface.mtl"))
            self.assertEqual(len(libraries), 2)
            committed = target.read_bytes()
            active = folder / next(
                line[7:]
                for line in committed.decode().splitlines()
                if line.startswith("mtllib ")
            )
            self.assertIn(active, libraries)
            for lib in libraries:
                png = lib.parent / next(
                    line[7:]
                    for line in lib.read_text().splitlines()
                    if line.startswith("map_Kd ")
                )
                self.assertEqual(hashlib.sha256(png.read_bytes()).hexdigest(), png.stem)
            # A user-modified content-addressed file is neither overwritten nor reused.
            lib = libraries[0]
            image = next(lib.parent.glob("*.png"))
            image.write_bytes(b"broken")
            failures = 0
            for seed in (2, 5):
                spec = TextureSpec.parse({"material": "stone", "seed": seed, "resolution": 128})
                try:
                    export_obj(
                        folder / "retry.obj", Scene((replace(box(), texture=spec),)), "render"
                    )
                except OSError:
                    failures += 1
            self.assertEqual(failures, 1)
            self.assertEqual(image.read_bytes(), b"broken")
            self.assertEqual(target.read_bytes(), committed)
