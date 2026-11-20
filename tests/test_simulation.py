import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from xtrifid.character import build_character
from xtrifid.geometry import build, triangles
from xtrifid.model import Part
from xtrifid.simulation import MAX_FRAME, Simulation


def support():
    return Part.parse(
        {
            "name": "Опора",
            "shape": "box",
            "size": [0.50, 0.34, 0.18],
            "center": [0.15, -0.1, 0.31],
            "rotation": [0, 0, 0],
        }
    )


class SimulationTests(unittest.TestCase):
    def test_cloth_hangs_over_its_support_and_replays_without_moving_pins(self):
        part = support()
        sim = Simulation("cloth", part, build(part, "render"), "medium", wind=0)
        initial = sim.mesh().vertices.copy()
        sim.seek(45)
        evolved = sim.mesh().vertices.copy()
        self.assertGreater(np.max(np.linalg.norm(evolved - initial, axis=1)), 0.035)
        self.assertGreater(np.sum(np.linalg.norm(evolved - initial, axis=1) < 1e-9), 4)
        self.assertGreaterEqual(evolved[:, 2].min(), -1e-7)
        # After the fabric drapes, the box's interior remains empty.
        inside = np.abs((evolved - np.array(part.center)) / (np.array(part.size) * 0.5)) < 0.998
        self.assertFalse(inside.all(axis=1).any())
        sim.seek(17)
        sim.seek(45)
        np.testing.assert_array_equal(evolved, sim.mesh().vertices)
        sim.reset()
        np.testing.assert_allclose(sim.mesh().vertices, initial, atol=1e-12)

    def test_water_rebounds_inside_walls_while_conserving_volume(self):
        part = support()
        sim = Simulation("water", part, build(part, "game"), "high", wind=0)
        before = sim.mesh().vertices.copy()
        sim.seek(64)
        after = sim.mesh().vertices.copy()
        np.testing.assert_array_equal(after[:, :2], before[:, :2])
        half = len(before) // 2
        self.assertLess(abs((after[:half, 2] - before[:half, 2]).sum()), 1e-8)
        self.assertGreater(np.ptp(after[:half, 2] - before[:half, 2]), 0.001)
        self.assertGreater(after[:half, 2].min(), after[half:, 2].max())
        np.testing.assert_array_equal(after[half:], before[half:])

    def test_existing_cloak_keeps_its_neckline_and_topology_in_motion(self):
        scene = build_character({"cloak": "knee"}, "low")
        torso, cloak = scene.parts
        m = build(cloak, "render", scene.detail)
        sim = Simulation(
            "cloth",
            cloak,
            m,
            "low",
            colliders=((torso, build(torso, "render", "low")),),
            wind=3,
        )
        self.assertTrue(sim.replace_selected)
        np.testing.assert_allclose(sim.mesh().vertices, m.vertices, atol=1e-10)
        self.assertEqual(sim.mesh().faces, m.faces)
        sim.seek(14)
        out = sim.mesh()
        self.assertGreater(abs(out.vertices - m.vertices).max(), 0.005)
        top = m.vertices[:, 2] > m.vertices[:, 2].max() - 0.001
        np.testing.assert_allclose(out.vertices[top], m.vertices[top], atol=0.002)

    def test_hair_grows_from_the_selected_scalp_and_remains_a_closed_visible_mesh(self):
        human = build_character({}).parts[0]
        mesh = build(human, "game", "medium")
        sim = Simulation("hair", human, mesh, "low", wind=4)
        initial = sim.mesh()
        self.assertFalse(sim.replace_selected)
        self.assertGreater(initial.vertices[:, 2].max(), 1.70)
        sim.seek(32)
        after = sim.mesh()
        self.assertGreater(abs(after.vertices - initial.vertices).max(), 0.008)
        self.assertTrue(np.isfinite(after.vertices).all())
        edges = [(a, b) for f in after.faces for a, b in zip(f, (*f[1:], f[0]))]
        self.assertEqual(len(set(edges)), len(edges))
        self.assertEqual(set(edges), {(b, a) for a, b in edges})
        tri = after.vertices[list(triangles(after.faces))]
        self.assertGreater(
            np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            ).min(),
            1e-11,
        )

    def test_scattered_locks_leave_the_default_portrait_eyes_and_mouth_visible(self):
        part = build_character({}).parts[0]
        skin = build(part, "game", "medium")
        height = np.ptp(skin.vertices[:, 2])
        crown = skin.vertices[:, 2].max()
        face = skin.vertices[skin.vertices[:, 2] > crown - 0.14 * height]
        front = face[:, 1].min() + 0.2 * np.ptp(face[:, 1])
        sim = Simulation("hair", part, skin, "high", wind=0)
        for frame in (0, 35):
            sim.seek(frame)
            x, y, z = sim.mesh().vertices.T
            obscured = (
                (abs(x) < 0.025 * height)
                & (y < front)
                & (z < crown - 0.066 * height)
                & (z > crown - 0.136 * height)
            )
            self.assertFalse(obscured.any(), frame)

    def test_wind_stays_within_the_shallow_wave_regime_for_large_basins(self):
        part = replace(support(), size=(1000, 1000, 1))
        sim = Simulation("water", part, build(part, "game"), "high", wind=8, strength=0.1)
        for frame in range(15, MAX_FRAME + 1, 15):
            sim.seek(frame)
            z = sim.mesh().vertices[:, 2]
            depth = z[: len(z) // 2] - z[-1]
            self.assertGreater(depth.min() / depth.mean(), 0.50)

    def test_weight_of_cloak_does_not_pull_it_through_the_shoulders(self):
        body, cloak = build_character({"cloak": "knee"}).parts
        skin, garment = (build(p, "render", "medium") for p in (body, cloak))
        sim = Simulation("cloth", cloak, garment, "medium", colliders=((body, skin),), wind=0)
        sim.seek(38)
        # The projected shoulder extremity stays under fabric on each side of the neck.
        vertices = sim.mesh().vertices
        mids = np.ptp(skin.vertices[:, 2]) * 0.80 + skin.vertices[:, 2].min()
        upper = vertices[np.abs(vertices[:, 2] - mids) < 0.032]
        body_band = skin.vertices[np.abs(skin.vertices[:, 2] - mids) < 0.003]
        self.assertGreaterEqual(np.ptp(upper[:, 0]), 0.96 * np.ptp(body_band[:, 0]))

    def test_cape_covers_the_back_and_roots_cover_the_crown_in_portrait(self):
        body, cloak = build_character({"cloak": "knee"}).parts
        skin = build(body, "render", "medium")
        mantle = Simulation(
            "cloth", cloak, build(cloak, "render", "medium"), "medium", ((body, skin),), wind=5
        )
        mantle.seek(8)

        def exterior(mesh, columns, x, y):
            tris = mesh.vertices[list(triangles(mesh.faces))][:, :, columns]
            a = tris[:, 1] - tris[:, 0]
            b = tris[:, 2] - tris[:, 0]
            det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            valid = np.abs(det) > 1e-13
            a, b, det, origin = a[valid], b[valid], det[valid], tris[valid, 0]
            dx, dy = x - origin[:, 0], y - origin[:, 1]
            u = (dx * b[:, 1] - dy * b[:, 0]) / det
            v = (dy * a[:, 0] - dx * a[:, 1]) / det
            hit = (u >= 0) & (v >= 0) & (u + v <= 1)
            return np.max((origin[:, 2] + a[:, 2] * u + b[:, 2] * v)[hit], initial=-np.inf)

        h = np.ptp(skin.vertices[:, 2])
        for z in (0.72, 0.78, 0.808):
            for x in (-0.075, 0, 0.075):
                with self.subTest(z=z, x=x):
                    self.assertGreater(
                        exterior(mantle.mesh(), [0, 2, 1], x * h, z * h) + 0.001,
                        exterior(skin, [0, 2, 1], x * h, z * h),
                    )
        mantle.seek(28)
        for z in (0.49, 0.53, 0.55):
            for x in (-0.035, 0.035):
                with self.subTest(hip_z=z, hip_x=x):
                    self.assertGreater(
                        exterior(mantle.mesh(), [0, 2, 1], x * h, z * h) + 0.001,
                        exterior(skin, [0, 2, 1], x * h, z * h),
                    )
        hair = Simulation("hair", body, skin, "high")
        for y in (-0.072, -0.035, 0.013):
            with self.subTest(crown_y=y):
                self.assertGreater(
                    exterior(hair.mesh(), [0, 1, 2], 0, y) + 0.001,
                    exterior(skin, [0, 1, 2], 0, y),
                )

    def test_invalid_request_and_tight_resources_fail_before_starting(self):
        part = support()
        mesh = build(part, "render")
        for kind, detail, kw in (
            ("smoke", "low", {}),
            ("cloth", "ultra", {}),
            ("water", "low", {"strength": float("nan")}),
            ("hair", "low", {"wind": 99}),
        ):
            with self.subTest(kind=kind, detail=detail, kw=kw), self.assertRaises(ValueError):
                Simulation(kind, part, mesh, detail, **kw)
        with (
            patch("xtrifid.simulation.mesh_triangle_budget", return_value=20),
            self.assertRaises(ValueError),
        ):
            Simulation("water", part, mesh, "low")
        sim = Simulation("cloth", part, mesh, "low")
        for frame in (-1, MAX_FRAME + 1, True):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                sim.seek(frame)
        self.assertEqual(sim.frame, 0)
