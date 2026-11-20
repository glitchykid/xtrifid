import os
import unittest

import numpy as np

from xtrifid.character import build_character, character_anchor
from xtrifid.geometry import build, rotation_matrix, triangle_bound
from xtrifid.model import Part, Scene


class PhysiqueTests(unittest.TestCase):
    def test_authored_adult_variants_share_topology_and_budget_at_each_level(self):
        for mode in ("game", "render"):
            for detail in ("low", "medium", "high"):
                with self.subTest(mode=mode, detail=detail):
                    people = [
                        build_character({"height_m": 1.8, "physique": kind}, detail)
                        for kind in ("neutral", "male", "female")
                    ]
                    self.assertEqual(people[0], build_character({"height_m": 1.8}, detail))
                    parts = [p.parts[0] for p in people]
                    meshes = [build(p, mode, detail) for p in parts]
                    for p, m in zip(parts, meshes):
                        self.assertEqual(m.faces, meshes[0].faces)
                        self.assertEqual(
                            triangle_bound(p, mode, detail),
                            triangle_bound(parts[0], mode, detail),
                        )
                        self.assertAlmostEqual(m.vertices[:, 2].max(), 1.8, places=10)
                        self.assertAlmostEqual(m.vertices[:, 2].min(), 0, places=10)
                    a, b = [m.vertices for m in meshes[1:]]
                    for z_range in ((0.63, 0.79), (0.88, 0.97)):
                        mask = (meshes[0].vertices[:, 2] > z_range[0] * 1.8) & (
                            meshes[0].vertices[:, 2] < z_range[1] * 1.8
                        )
                        self.assertGreater(
                            np.linalg.norm(a[mask] - b[mask], axis=1).mean(), 0.009
                        )
                    self.assertEqual(Scene.parse(people[-1].data()), people[-1])

    def test_hands_and_eyes_follow_adult_variants_in_all_poses(self):
        for kind in ("male", "female"):
            for pose in ("standing", "walking", "arms_out"):
                scene = build_character({"physique": kind, "pose": pose})
                v = build(scene.parts[0], "game", "medium").vertices
                with self.subTest(kind=kind, pose=pose):
                    for name in ("Кисть левая", "Кисть правая", "Голова"):
                        location = character_anchor(scene, name)
                        self.assertLess(np.linalg.norm(v - location, axis=1).min(), 0.045)
                    self.assertAlmostEqual(v[:, 2].min(), 0, places=10)
                    self.assertAlmostEqual(np.ptp(v[:, 2]), 1.75, places=10)

    def test_every_pose_preserves_stature_and_ground_plane_in_local_coordinates(self):
        tilt = (13, -27, 41)
        axis = rotation_matrix(tilt)[:, 2]
        for physique in ("male", "female", "neutral"):
            for pose in ("walking", "arms_out", "standing"):
                part = build_character(
                    {"physique": physique, "pose": pose, "height_m": 1.75}
                ).parts[0]
                rotated = Part.parse(part.data() | {"rotation": tilt})
                for mode in ("game", "render"):
                    for detail in ("low", "medium", "high"):
                        with self.subTest(
                            physique=physique, pose=pose, mode=mode, detail=detail
                        ):
                            v = build(rotated, mode, detail).vertices
                            h = (v - rotated.center) @ axis + rotated.center[2]
                            np.testing.assert_allclose(
                                [h.min(), h.max()], [0, 1.75], atol=1e-10, rtol=0
                            )

    def test_lod_calibration_preserves_explicit_extremity_edits(self):
        scene = build_character(
            {"physique": "female", "pose": "walking", "height_m": 1.8}, "low"
        )
        part = scene.parts[0]
        v = build(part, "game", scene.detail).vertices
        for region in ("head", "feet"):
            changed = Part.parse(part.data() | {"body_scales": {region: 1.18}})
            edited = build(changed, "game", scene.detail).vertices
            unaffected = (
                v[:, 2] < part.size[2] * 0.77
                if region == "head"
                else v[:, 2] > part.size[2] * 0.19
            )
            np.testing.assert_array_equal(edited[unaffected], v[unaffected])
            self.assertGreater(np.linalg.norm(edited - v, axis=1).max(), 0.015)
            if region == "head":
                self.assertGreater(edited[:, 2].max(), 1.83)
            else:
                left = character_anchor(scene.with_part(0, changed), "Кисть левая")
                np.testing.assert_array_equal(left, character_anchor(scene, "Кисть левая"))

    def test_invalid_or_missing_physique(self):
        part = build_character({}).parts[0]
        for value in (True, None, "unknown", []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Part.parse(part.data() | {"physique": value})
        old = part.data()
        old.pop("physique", None)
        self.assertEqual(Part.parse(old).physique, "neutral")

    @unittest.skipUnless(os.getenv("DISPLAY"), "requires GTK display")
    def test_generation_from_gendered_words_through_the_gtk_button(self):
        from test_studio import until

        from xtrifid.gtk import Gio, Gtk
        from xtrifid.ui import Studio

        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.AnatomyTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Studio(app)
        self.addCleanup(window.close)
        window.present()
        window.detail.set_selected(2)
        for prompt, expected in (
            ("Сделай мужчину", "male"),
            ("Сделай женщину", "female"),
            ("Человек", "neutral"),
        ):
            window.prompt.set_text(prompt)
            window.generate.emit("clicked")
            until(lambda: not window.busy)
            self.assertIsNotNone(window.doc.scene, window.status.get_text())
            self.assertEqual(len(window.doc.scene.parts), 1)
            self.assertEqual(window.doc.scene.parts[0].physique, expected)
        self.assertFalse(window.view.problem)
