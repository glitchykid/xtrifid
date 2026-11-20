import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_textures import box

from xtrifid.geometry import Mesh
from xtrifid.gltf import export_glb
from xtrifid.model import Scene
from xtrifid.resources import GIB, Capacity
from xtrifid.surface import baked_part
from xtrifid.textures import (
    TextureSpec,
    metallic_roughness_pixels,
    normal_pixels,
    pixels,
    png_bytes,
)


def decode(path):
    raw = path.read_bytes()
    magic, version, total, json_size, json_tag = struct.unpack_from("<4s3I4s", raw)
    assert (magic, version, total, json_tag) == (b"glTF", 2, len(raw), b"JSON")
    bin_size, bin_tag = struct.unpack_from("<I4s", raw, 20 + json_size)
    body = raw[28 + json_size :]
    assert (
        bin_tag == b"BIN\x00" and bin_size == len(body) and not (json_size % 4 or bin_size % 4)
    )
    doc = json.loads(raw[20 : 20 + json_size])
    assert 0 <= bin_size - doc["buffers"][0]["byteLength"] <= 3
    return doc, body


def read_accessor(doc, body, index):
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype({5126: "<f4", 5123: "<u2", 5125: "<u4"}[accessor["componentType"]])
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    return np.ndarray(
        (accessor["count"], width),
        dtype,
        body,
        view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
        (view.get("byteStride", width * dtype.itemsize), dtype.itemsize),
    )


def image_bytes(doc, body, reference):
    texture = doc["textures"][reference["index"]]
    image = doc["images"][texture["source"]]
    assert image["mimeType"] == "image/png" and "uri" not in image
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return body[start : start + view["byteLength"]]


def physical_spec():
    return TextureSpec.parse(
        {
            "material": "metal",
            "resolution": 128,
            "pbr": {"roughness": 0.35, "metallic": 0.8, "normal_strength": 1.75},
        }
    )


class GltfTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name, "моя деталь.glb")

    def test_local_indexed_geometry_and_pivots_preserve_oriented_instances_in_metres(self):
        first = box()
        second = replace(first, name="Другой куб", center=(3, 4, 5), rotation=(0, 0, 90))
        export_glb(self.path, Scene((first, second)))
        doc, body = decode(self.path)
        self.assertEqual(doc["asset"]["version"], "2.0")
        self.assertEqual(doc["scenes"][doc["scene"]]["nodes"], [0, 1])
        self.assertEqual([n["name"] for n in doc["nodes"]], [first.name, second.name])
        self.assertEqual(len(doc["meshes"]), 1)
        self.assertEqual(doc["nodes"][0]["mesh"], doc["nodes"][1]["mesh"])
        primitive = doc["meshes"][0]["primitives"][0]
        self.assertEqual(primitive.get("mode", 4), 4)
        self.assertEqual(set(primitive["attributes"]), {"POSITION", "NORMAL"})
        self.assertNotIn("images", doc)
        self.assertEqual(
            doc["materials"][primitive["material"]]["pbrMetallicRoughness"]["metallicFactor"], 0
        )
        points, normals = (
            read_accessor(doc, body, primitive["attributes"][key])
            for key in ("POSITION", "NORMAL")
        )
        indices = read_accessor(doc, body, primitive["indices"]).reshape(-1, 3)
        self.assertEqual(
            (len(points), len(indices), len(np.unique(points, axis=0))), (24, 12, 8)
        )
        bounds = doc["accessors"][primitive["attributes"]["POSITION"]]
        np.testing.assert_allclose(bounds["min"], points.min(axis=0))
        np.testing.assert_allclose(bounds["max"], points.max(axis=0))
        tri = points[indices]
        outward = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        outward /= np.linalg.norm(outward, axis=1, keepdims=True)
        np.testing.assert_allclose(normals[indices], np.repeat(outward[:, None], 3, axis=1))
        matrix = np.array(doc["nodes"][1]["matrix"]).reshape(4, 4).T
        np.testing.assert_allclose(matrix[:3, 3], [3, 5, -4])
        world = points @ matrix[:3, :3].T + matrix[:3, 3]
        np.testing.assert_allclose(world.min(axis=0), [2, 3.5, -4.5])
        np.testing.assert_allclose(world.max(axis=0), [4, 6.5, -3.5])

    def test_inline_images_have_correct_roles_values_and_share_between_materials(self):
        spec = physical_spec()
        legacy = replace(spec, pbr=None)
        parts = (
            replace(box("Metal"), texture=spec),
            replace(box("Old"), shape="sphere", texture=legacy),
            replace(box("Also metal"), shape="torus", texture=spec),
            box("Clay"),
        )
        scene = Scene(parts)
        export_glb(self.path, scene, "game")
        doc, body = decode(self.path)
        self.assertEqual(len(doc["images"]), 3)
        self.assertEqual(len(doc["materials"]), 3)
        self.assertNotIn("uri", doc["buffers"][0])
        primitives = [doc["meshes"][node["mesh"]]["primitives"][0] for node in doc["nodes"]]
        self.assertEqual(primitives[0]["material"], primitives[2]["material"])
        material = doc["materials"][primitives[0]["material"]]
        pbr = material["pbrMetallicRoughness"]
        for key in ("metallicFactor", "roughnessFactor"):
            self.assertEqual(pbr.get(key, 1), 1)
        self.assertEqual(material["normalTexture"].get("scale", 1), 1)
        self.assertEqual(
            image_bytes(doc, body, pbr["baseColorTexture"]), png_bytes(pixels(spec))
        )
        self.assertEqual(
            image_bytes(doc, body, material["normalTexture"]), png_bytes(normal_pixels(spec))
        )
        self.assertEqual(
            image_bytes(doc, body, pbr["metallicRoughnessTexture"]),
            png_bytes(metallic_roughness_pixels(spec)),
        )
        old_material = doc["materials"][primitives[1]["material"]]
        self.assertNotIn("normalTexture", old_material)
        self.assertNotIn("metallicRoughnessTexture", old_material["pbrMetallicRoughness"])
        self.assertEqual(
            old_material["pbrMetallicRoughness"]["baseColorTexture"], pbr["baseColorTexture"]
        )
        self.assertIn("TEXCOORD_0", primitives[1]["attributes"])
        self.assertNotIn("TANGENT", primitives[1]["attributes"])
        previous = self.path.read_bytes()
        export_glb(self.path, scene)
        self.assertEqual(previous, self.path.read_bytes())
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_flipped_gltf_uv_and_authored_positive_green_on_both_mirrored_cylinder_caps(self):
        part = replace(box(), shape="cylinder", texture=physical_spec())
        export_glb(self.path, Scene((part,), "low"))
        doc, body = decode(self.path)
        primitive = doc["meshes"][0]["primitives"][0]
        attrs = primitive["attributes"]
        points, normals, uv, tangent = (
            read_accessor(doc, body, attrs[key])
            for key in ("POSITION", "NORMAL", "TEXCOORD_0", "TANGENT")
        )
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(tangent[:, :3], axis=1), 1, atol=2e-6)
        np.testing.assert_allclose(np.sum(normals * tangent[:, :3], axis=1), 0, atol=2e-6)
        bitangent = np.cross(normals, tangent[:, :3]) * tangent[:, 3:4]
        for sign in (1, -1):
            cap = np.flatnonzero(normals[:, 1] * sign > 0.999)
            self.assertGreater(len(cap), 3)
            np.testing.assert_allclose(
                uv[cap, 0], points[cap, 0] / part.size[0] + 0.5, atol=1e-6
            )
            np.testing.assert_allclose(
                uv[cap, 1], points[cap, 2] / part.size[1] + 0.5, atol=1e-6
            )
            np.testing.assert_allclose(tangent[cap, 0], 1, atol=1e-6)
            np.testing.assert_allclose(tangent[cap, 3], sign)
            # On both caps, the positive green channel must tilt towards the top of the PNG.
            np.testing.assert_allclose(bitangent[cap, 2], -1, atol=1e-6)

    def test_dense_faceted_surface_uses_32_bit_indices_after_hard_edge_splits(self):
        count = 108
        y, x = np.mgrid[:count, :count]
        z = np.random.default_rng(19).uniform(-0.4, 0.4, size=x.shape)
        vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        a = (x + count * y)[:-1, :-1].ravel()
        faces = np.concatenate(
            (
                np.stack((a, a + 1, a + count), axis=1),
                np.stack((a + 1, a + count + 1, a + count), axis=1),
            )
        )
        mesh = Mesh(vertices, tuple(map(tuple, faces)), (False,) * len(faces))
        part = baked_part("Faceted", mesh)
        with patch("xtrifid.resources.capacity", return_value=Capacity(2 * GIB, 2)):
            export_glb(self.path, Scene((part,)))
        doc, body = decode(self.path)
        primitive = doc["meshes"][0]["primitives"][0]
        indices = read_accessor(doc, body, primitive["indices"])
        self.assertEqual(indices.dtype, np.dtype("<u4"))
        self.assertGreater(int(indices.max()), 65535)
        self.assertLess(
            int(indices.max()), doc["accessors"][primitive["attributes"]["POSITION"]]["count"]
        )

    def test_rejection_and_failed_replacement_leave_previous_file_intact(self):
        self.path.write_bytes(b"previous")
        scene = Scene((box(),))
        for value, mode in (
            (scene, "render"),
            (scene, "other"),
            (None, "game"),
            (Scene(()), "game"),
        ):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                export_glb(self.path, value, mode)
        with (
            patch("xtrifid.resources.capacity", return_value=Capacity(0, 1)),
            self.assertRaises(ValueError),
        ):
            export_glb(self.path, scene)
        with (
            patch("xtrifid.gltf.os.replace", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            export_glb(self.path, scene)
        self.assertEqual(self.path.read_bytes(), b"previous")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
