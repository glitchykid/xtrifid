import base64
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from test_gltf import decode, read_accessor
from test_studio import until

from xtrifid.character import build_character
from xtrifid.geometry import Mesh, build
from xtrifid.gltf import export_glb
from xtrifid.gtk import Gio, Gtk
from xtrifid.model import Scene
from xtrifid.storage import export_obj, load_project, save_project
from xtrifid.surface import Surface, baked_part
from xtrifid.tangents import corner_frames
from xtrifid.textures import TextureSpec, pbr_preset, uv_layout
from xtrifid.ui import Studio


def small_surface():
    mesh = Mesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]),
        ((0, 1, 2, 3), (0, 1, 4)),
        (False, True),
    )
    # Two equal coordinates have distinct chart identities at the geometric seam.
    uv = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1.25, 0.5]])
    return mesh, (uv, ((0, 1, 2, 3), (4, 5, 6)))


class BakedMaterialUVTests(unittest.TestCase):
    def test_legacy_diffuse_repeated_bake_and_quad_to_triangle_preserve_uv_and_seams(self):
        mesh, chart = small_surface()
        plain = baked_part("plain", mesh)
        self.assertNotIn("uv", plain.surface.data())
        first = baked_part("first", mesh, uv_chart=chart)
        first = replace(first, texture=TextureSpec.parse({"material": "stone"}))
        src = build(first, "render")
        uv, indices = uv_layout(src, first)
        np.testing.assert_array_equal(uv, chart[0])
        self.assertEqual(tuple(map(tuple, indices)), chart[1])
        moved = replace(src, vertices=src.vertices + (2, -3, 1))
        second = baked_part("second", moved, uv_chart=(uv, indices))
        second = replace(second, texture=first.texture)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cloth.xtri"
            save_project(path, Scene((second,)), "render")
            self.assertEqual(json.loads(path.read_text())["version"], 5)
            recovered = load_project(path)[0].parts[0]
        game_uv, triangles = uv_layout(build(recovered, "game"), recovered)
        np.testing.assert_array_equal(uv, game_uv)
        self.assertEqual(tuple(map(tuple, triangles)), ((0, 1, 2), (0, 2, 3), (4, 5, 6)))

    def test_partially_present_or_malformed_stored_chart_rejected_before_access(self):
        mesh, chart = small_surface()
        surface = baked_part("sample", mesh, uv_chart=chart).surface
        data = surface.data()
        valid = data["uv"]

        def encoded(array):
            return base64.b64encode(array.tobytes()).decode("ascii")

        coords = np.frombuffer(base64.b64decode(valid["coords"]), "<f4").copy()
        ids = np.frombuffer(base64.b64decode(valid["faces"]), "<i4").copy()
        invalid = [None, {}, {"coords": valid["coords"]}, {**valid, "faces": "@"}]
        for value in (np.nan, np.inf, 99):
            coords[0] = value
            invalid.append({**valid, "coords": encoded(coords)})
        for index, value in ((0, -1), (4, 99), (7, 5), (3, -1)):
            changed = ids.copy()
            changed[index] = value
            invalid.append({**valid, "faces": encoded(changed)})
        invalid.extend(
            (
                {**valid, "coords": encoded(np.zeros((9, 2), dtype="<f4"))},
                {**valid, "faces": encoded(ids[:-1])},
                {**valid, "extra": ""},
            )
        )
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Surface.parse({**data, "uv": bad})
        for invalid_layout in (
            (chart[0], ((0, 1, 2), (4, 5, 6))),
            (chart[0], ((0, 1, 2, 3), (4, 5, 99))),
        ):
            with self.subTest(invalid_layout=invalid_layout), self.assertRaises(ValueError):
                baked_part("bad", mesh, uv_chart=invalid_layout)

    @unittest.skipUnless(os.getenv("DISPLAY"), "requires GTK")
    def test_cloak_simulation_bake_reopen_and_both_game_exports_match_the_presented_frame(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.BakedMaterial",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.present()
        scene = build_character({"cloak": "knee"}, detail="low")
        cloak = scene.parts[-1]
        fabric = TextureSpec.parse({"material": "fabric", "resolution": 128})
        cloak = replace(cloak, texture=replace(fabric, pbr=pbr_preset("fabric")))
        scene = scene.with_part(len(scene.parts) - 1, cloak)
        window.install(scene, "render")
        window.select(len(scene.parts) - 1)
        panel = window.simulation
        panel.set_expanded(True)
        panel.wind.set_value(5)
        panel.timeline.set_value(8)
        until(lambda: panel.sim and panel.sim.frame == 8)
        current = panel.sim.mesh()
        expected = corner_frames(current, cloak, rest_mesh=window.meshes[-1])
        panel.bake.emit("clicked")
        baked = window.doc.scene.parts[-1]
        self.assertEqual(baked.shape, "surface")
        self.assertIsNotNone(baked.surface.uv)
        with tempfile.TemporaryDirectory() as folder:
            project, obj, glb = (
                Path(folder, "result." + ext) for ext in ("xtri", "obj", "glb")
            )
            save_project(project, Scene((baked,), detail="low"), "render")
            scene = load_project(project)[0]
            export_obj(obj, scene, "game")
            export_glb(glb, scene, "game")
            lines = obj.read_text().splitlines()
            uv = np.array(
                [line.split()[1:] for line in lines if line.startswith("vt ")], dtype=float
            )
            index = [
                [int(v.split("/")[1]) - 1 for v in line.split()[1:]]
                for line in lines
                if line.startswith("f ")
            ]
            np.testing.assert_allclose(uv[index], expected[2], atol=1e-6)
            doc, body = decode(glb)
            prim = doc["meshes"][0]["primitives"][0]
            ids = read_accessor(doc, body, prim["indices"]).reshape(-1, 3)
            uv = read_accessor(doc, body, prim["attributes"]["TEXCOORD_0"])[ids].copy()
            uv[..., 1] = 1 - uv[..., 1]
            np.testing.assert_allclose(uv, expected[2], atol=1e-6)
            tangent = read_accessor(doc, body, prim["attributes"]["TANGENT"])[ids]
            np.testing.assert_allclose(
                tangent[..., [0, 2, 1, 3]] * (1, -1, 1, 1), expected[3], atol=5e-5
            )
