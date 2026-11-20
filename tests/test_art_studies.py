import tempfile
import unittest
from pathlib import Path

from test_gltf import decode, image_bytes

from xtrifid.gltf import export_glb
from xtrifid.storage import load_project, save_project

STUDIES = Path(__file__).resolve().parents[1] / "examples" / "art-studies"


class ArtStudiesTests(unittest.TestCase):
    def test_game_projects_remain_editable_and_export_their_materials(self):
        files = sorted(STUDIES.glob("*.xtri"))
        expected = {
            "moon-sword",
            "prism-blaster",
            "parade-sidearm",
            "fan-relic",
            "marsh-creature",
            "wayfarer-lantern",
            "orbital-spell",
            "tidal-crest",
        }
        self.assertTrue(expected <= {p.stem for p in files}, "Missing art study")
        with tempfile.TemporaryDirectory() as temporary:
            saved = Path(temporary) / "edited.xtri"
            output = saved.with_suffix(".glb")
            for path in files:
                with self.subTest(study=path.stem):
                    scene, mode = load_project(path)
                    self.assertEqual(mode, "game")
                    names = [part.name for part in scene.parts]
                    self.assertEqual(len(names), len(set(names)))
                    save_project(saved, scene, mode)
                    self.assertEqual(load_project(saved), (scene, mode))
                    export_glb(output, scene, mode)
                    document, body = decode(output)
                    self.assertEqual([node["name"] for node in document["nodes"]], names)
                    materials = document["materials"]
                    self.assertGreater(len(materials), 1)
                    for mesh in document["meshes"]:
                        for primitive in mesh["primitives"]:
                            self.assertIn("TANGENT", primitive["attributes"])
                            material = materials[primitive["material"]]
                            pbr = material["pbrMetallicRoughness"]
                            for texture in (
                                pbr["baseColorTexture"],
                                pbr["metallicRoughnessTexture"],
                                material["normalTexture"],
                            ):
                                self.assertTrue(
                                    image_bytes(document, body, texture).startswith(
                                        b"\x89PNG\r\n\x1a\n"
                                    )
                                )
