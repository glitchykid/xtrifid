import math
import struct
import unittest
import zlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from xtrifid.material_maps import normals_from_height
from xtrifid.textures import (
    PALETTES,
    PBRMaterial,
    TextureSpec,
    metallic_roughness_pixels,
    normal_pixels,
    pbr_preset,
    pixels,
    png_bytes,
    texture_working_bytes,
)


def material(name="wood", **options):
    spec = TextureSpec.parse({"material": name, "seed": 17, "resolution": 128})
    return replace(spec, pbr=replace(pbr_preset(name), **options))


class MaterialMapsTests(unittest.TestCase):
    def test_material_recipe_round_trip_and_legacy_colour(self):
        for name in PALETTES:
            with self.subTest(name=name):
                spec = material(name)
                legacy = replace(spec, pbr=None)
                self.assertNotIn("pbr", legacy.data())
                self.assertEqual(TextureSpec.parse(legacy.data()), legacy)
                self.assertEqual(TextureSpec.parse(spec.data()), spec)
                self.assertEqual(PBRMaterial.parse(spec.data()["pbr"]), spec.pbr)
                np.testing.assert_array_equal(pixels(spec), pixels(legacy))
                self.assertEqual(spec.pbr.metallic, float(name == "metal"))
        self.assertLess(pbr_preset("metal").roughness, pbr_preset("fabric").roughness)

    def test_invalid_or_partial_physical_settings_fail_at_the_boundary(self):
        good = {"roughness": 0.4, "metallic": 0, "normal_strength": 0.7}
        bad = [True, {}, [], {"roughness": 1}, {**good, "normal_file": "/tmp/file"}]
        for key, limit in (("roughness", 1), ("metallic", 1), ("normal_strength", 2)):
            bad.extend(
                {**good, key: number}
                for number in (-0.01, limit + 0.01, math.nan, math.inf, False, "1", None)
            )
        for settings in bad:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TextureSpec.parse({"material": "skin", "pbr": settings})
        with self.assertRaises(ValueError):
            replace(material(), pbr=good)
        for value in ("unknown", None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pbr_preset(value)
        legacy = TextureSpec.parse({"material": "paint", "pbr": None})
        for generate in (normal_pixels, metallic_roughness_pixels):
            with self.assertRaises(ValueError):
                generate(legacy)

    def test_sinusoidal_height_has_the_analytic_normal_across_both_wrapped_edges(self):
        n = 128
        axis = np.arange(n) * 2 * np.pi / n
        height = 0.08 * np.sin(axis)[None, :] + 0.04 * np.sin(axis)[:, None]
        up = np.ones_like(height)
        # Increasing column is +U. Increasing PNG row points against +V.
        dx = -0.08 * n * np.sin(2 * np.pi / n) * np.cos(axis)[None, :] * up
        dy = 0.04 * n * np.sin(2 * np.pi / n) * np.cos(axis)[:, None] * up
        expected = np.stack((dx, dy, up), axis=-1)
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        image = normals_from_height(height.astype(np.float32), 1)
        np.testing.assert_allclose(image / 127.5 - 1, expected, atol=1 / 255 + 1e-6)

    def test_actual_surfaces_produce_repeatable_unit_normals_independent_of_pigment(self):
        tiles = set()
        for name in PALETTES:
            with self.subTest(name=name):
                spec = material(name)
                image = normal_pixels(spec)
                self.assertEqual((image.shape, image.dtype), ((128, 128, 3), np.uint8))
                self.assertGreater(image[:, :, :2].std(), 2)
                self.assertGreater(image[:, :, 2].min(), 127)
                decoded = image.astype(float) / 127.5 - 1
                np.testing.assert_allclose(np.linalg.norm(decoded, axis=-1), 1, atol=0.007)
                dyed = replace(spec, color_a="#444444", color_b="#444444", motif="spots")
                np.testing.assert_array_equal(normal_pixels(dyed), image)
                np.testing.assert_array_equal(
                    normal_pixels(TextureSpec.parse(spec.data())), image
                )
                self.assertFalse(np.array_equal(normal_pixels(replace(spec, seed=25)), image))
                self.assertFalse(np.array_equal(normal_pixels(replace(spec, scale=0.5)), image))
                for axis in (0, 1):
                    delta = np.abs(np.roll(decoded, 1, axis=axis) - decoded)
                    self.assertLess(delta.take(0, axis=axis).mean(), 3 * delta.mean() + 0.005)
                tiles.add(image.tobytes())
        self.assertEqual(len(tiles), len(PALETTES))
        wood = normal_pixels(material("wood")).std(axis=(0, 1))
        metal = normal_pixels(material("metal")).std(axis=(0, 1))
        self.assertGreater(wood[0], 1.6 * wood[1])
        self.assertGreater(metal[1], 1.6 * metal[0])

    def test_strength_and_roughness_endpoints_keep_the_requested_finish(self):
        neutral = normal_pixels(material(normal_strength=0))
        self.assertTrue(np.all(neutral == [128, 128, 255]))
        weak = normal_pixels(material("fabric", normal_strength=0.5))
        strong = normal_pixels(material("fabric", normal_strength=2))
        self.assertGreater(strong[:, :, :2].std(), weak[:, :, :2].std() * 1.4)
        for value in (0, 1):
            image = metallic_roughness_pixels(material(roughness=value, metallic=1 - value))
            self.assertTrue(np.all(image == [255, 255 * value, 255 * (1 - value)]))
        spec = material("stone", roughness=0.42, metallic=0.25)
        image = metallic_roughness_pixels(spec)
        self.assertEqual((image.shape, image.dtype), ((128, 128, 3), np.uint8))
        self.assertTrue(np.all(image[:, :, 0] == 255))
        self.assertTrue(np.all(image[:, :, 2] == 64))
        self.assertAlmostEqual(image[:, :, 1].mean() / 255, 0.42, delta=0.01)
        self.assertGreater(image[:, :, 1].std(), 2.5)
        np.testing.assert_array_equal(
            image, metallic_roughness_pixels(replace(spec, color_b="#000000"))
        )

    def test_three_resident_maps_are_budgeted_once_for_identical_recipes(self):
        one = SimpleNamespace(texture=material())
        legacy = SimpleNamespace(texture=replace(one.texture, pbr=None))
        size = one.texture.resolution**2
        self.assertGreaterEqual(
            texture_working_bytes([one]) - texture_working_bytes([legacy]), 8 * size
        )
        self.assertEqual(texture_working_bytes([one, one]), texture_working_bytes([one]))
        second = SimpleNamespace(texture=replace(one.texture, seed=91))
        self.assertGreaterEqual(
            texture_working_bytes([one, second]) - texture_working_bytes([one]), 3 * 4 * size
        )

    def test_grayscale_png_encodes_single_linear_channel_even_from_strided_view(self):
        packed = metallic_roughness_pixels(material("wood"))
        gray = packed[:, :, 1]
        payload = png_bytes(gray)
        width, height, bits, mode = struct.unpack_from(">IIBB", payload, 16)
        self.assertEqual((width, height, bits, mode), (128, 128, 8, 0))
        self.assertEqual(payload[37:41], b"IDAT")
        size = struct.unpack_from(">I", payload, 33)[0]
        scanlines = np.frombuffer(zlib.decompress(payload[41 : 41 + size]), np.uint8).reshape(
            128, 129
        )
        np.testing.assert_array_equal(scanlines[:, 1:], gray)
        self.assertTrue(np.all(scanlines[:, 0] == 0))
        self.assertNotIn(b"gAMA", payload[:33])
