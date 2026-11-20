import json
import os
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cairo
from test_model import spec
from test_planner import PlannerHTTPCase
from test_studio import until
from test_textures import box

from xtrifid.gtk import Gio, Gtk
from xtrifid.model import Scene
from xtrifid.ollama import PlannerError
from xtrifid.storage import export_obj, load_project, save_project
from xtrifid.texture_plan import text_to_texture
from xtrifid.textures import TextureSpec
from xtrifid.ui import Studio


def metal():
    return TextureSpec.parse(
        {
            "material": "metal",
            "resolution": 128,
            "pbr": {"roughness": 0.24, "metallic": 0.9, "normal_strength": 1.6},
        }
    )


def texel(path, x, y):
    image = cairo.ImageSurface.create_from_png(str(path))
    value = struct.unpack_from("=I", image.get_data(), y * image.get_stride() + x * 4)[0]
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


class GameMaterialFiles(unittest.TestCase):
    def test_version_five_persists_material_v4_preserves_diffuse_and_v5_rejects_bad_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.xtri"
            for texture, version in (
                (TextureSpec.parse({"material": "wood"}), 4),
                (metal(), 5),
            ):
                scene = Scene((replace(box(), texture=texture),))
                save_project(path, scene, "game")
                self.assertEqual(json.loads(path.read_bytes())["version"], version)
                self.assertEqual(load_project(path), (scene, "game"))
            invalid = json.loads(path.read_bytes())
            invalid["parts"][0]["texture"]["pbr"]["normal_strength"] = -1
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_project(path)

    def test_obj_exports_data_maps_in_linear_channels_and_corner_normals_only_for_game(self):
        from xtrifid.textures import metallic_roughness_pixels, normal_pixels

        tex = metal()
        scene = Scene((replace(box(), texture=tex), box("Clay")))
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "materials.obj"
            for mode in ("render", "game"):
                export_obj(obj, scene, mode)
                lines = obj.read_text().splitlines()
                mtl = obj.parent / next(
                    line[7:] for line in lines if line.startswith("mtllib ")
                )
                fields = dict(
                    line.split(maxsplit=1) for line in mtl.read_text().splitlines() if line
                )
                self.assertTrue((mtl.parent / fields["map_Kd"]).is_file())
                if mode == "render":
                    self.assertNotIn("norm", fields)
                    continue
                data = metallic_roughness_pixels(tex)[24, 13]
                self.assertTupleEqual(
                    texel(mtl.parent / fields["norm"], 13, 24),
                    tuple(normal_pixels(tex)[24, 13]),
                )
                for key, channel in (("map_Pr", 1), ("map_Pm", 2)):
                    self.assertTupleEqual(
                        texel(mtl.parent / fields[key], 13, 24), (data[channel],) * 3
                    )
                normals = [line for line in lines if line.startswith("vn ")]
                self.assertEqual(len(normals), 6)
                face = next(line for line in lines if line.startswith("f "))
                self.assertTrue(
                    all(1 <= int(v.split("/")[2]) <= len(normals) for v in face.split()[1:])
                )
                self.assertNotIn("bump", fields)


class GameMaterialPlanner(PlannerHTTPCase):
    def test_physical_recipe_repairs_invalid_response_and_preserves_user_resolution(self):
        answer = dict(
            material="metal",
            motif="none",
            color_a="#82724a",
            color_b="#ebc975",
            accent="#a49881",
            scale=1,
            pbr={"roughness": 0.11, "metallic": 1, "normal_strength": 0.34},
        )
        self.answer({**answer, "pbr": {**answer["pbr"], "roughness": 2}})
        self.answer(answer)
        result = text_to_texture(self.client, "полированная латунь", 17, 128, physical=True)
        self.assertEqual((result.seed, result.resolution), (17, 128))
        self.assertEqual(result.data()["pbr"], answer["pbr"])
        self.assertIn("pbr", self.calls[0]["format"]["required"])
        self.assertEqual(len(self.calls), 2)
        for _ in range(2):
            self.answer({**answer, "pbr": None})
        with self.assertRaises(PlannerError):
            text_to_texture(self.client, "латунь", physical=True)


@unittest.skipUnless(os.getenv("DISPLAY"), "requires GTK")
class GameMaterialWindow(unittest.TestCase):
    def test_choose_assign_save_game_asset_switch_profiles_remove_and_undo(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.MaterialWorkflow",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.present()
        self.assertFalse(window.glb_exporter.get_sensitive())
        window.install(Scene.parse({"parts": [spec(), spec("sphere")]}), "game")
        m = window.material
        self.assertTrue(m.physical.get_active())
        m.material.set_selected(4)
        self.assertEqual(m.metallic.get_value(), 1)
        m.roughness.set_value(0.23)
        m.normal_strength.set_value(1.4)
        m.apply_button.emit("clicked")
        assigned = window.doc.scene.parts[0].texture
        self.assertAlmostEqual(assigned.pbr.roughness, 0.23)
        self.assertAlmostEqual(assigned.pbr.normal_strength, 1.4)
        self.assertIsNone(window.doc.scene.parts[1].texture)
        window.mode.set_selected(1)
        self.assertFalse(window.glb_exporter.get_sensitive())
        window.mode.set_selected(0)
        self.assertTrue(window.glb_exporter.get_sensitive())
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"XTRIFID_FILES": tmp}),
        ):
            window.glb_exporter.emit("clicked")

            def chooser():
                return next(
                    (
                        w
                        for w in Gtk.Window.list_toplevels()
                        if isinstance(w, Gtk.FileChooserDialog) and w.get_visible()
                    ),
                    None,
                )

            until(
                lambda: (
                    chooser()
                    and chooser().get_current_folder()
                    and chooser().get_current_folder().get_path() == tmp
                )
            )
            chooser().set_current_name("деталь.glb")
            chooser().response(Gtk.ResponseType.ACCEPT)
            path = Path(tmp) / "деталь.glb"
            until(lambda: path.is_file() and not window.busy)
            self.assertEqual(path.read_bytes()[:4], b"glTF")
        m.remove_button.emit("clicked")
        self.assertIsNone(window.doc.scene.parts[0].texture)
        window.undo()
        self.assertEqual(window.doc.scene.parts[0].texture, assigned)
        window.apply_texture(TextureSpec.parse({"material": "wood"}))
        self.assertFalse(m.physical.get_active())
        self.assertFalse(m.normal_strength.is_sensitive())
