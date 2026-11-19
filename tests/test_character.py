import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from xtrifid.character import build_character
from xtrifid.geometry import build, triangles
from xtrifid.model import Part, Scene
from xtrifid.organic import ORGANIC_SHAPES, organic_mesh, organic_triangle_bound
from xtrifid.storage import load_project, save_project


def depths_at(mesh, x, z):
    """Intersect a ray through (x, z) with the actual mesh triangles."""
    a, b, c = mesh.vertices[list(triangles(mesh.faces))].transpose(1, 0, 2)
    u, v, point = b - a, c - a, np.array([x, z]) - a[:, [0, 2]]
    det = u[:, 0] * v[:, 2] - u[:, 2] * v[:, 0]
    valid = abs(det) > 1e-12
    det = np.where(valid, det, 1)
    s = (point[:, 0] * v[:, 2] - point[:, 1] * v[:, 0]) / det
    t = (u[:, 0] * point[:, 1] - u[:, 2] * point[:, 0]) / det
    hit = valid & (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9)
    return (a[:, 1] + u[:, 1] * s + v[:, 1] * t)[hit]


class CharacterTests(unittest.TestCase):
    def test_low_poly_fabric_uses_matching_diagonals_on_both_sides_of_each_panel(self):
        scene = build_character({"hood": True, "cloak": "knee"})
        for part in scene.parts:
            if part.shape not in ("cloak", "hood"):
                continue
            mesh = build(part, "game", "low")
            offset = len(mesh.vertices) // 2
            outer = {tuple(sorted(f)) for f in mesh.faces if max(f) < offset}
            inner = {
                tuple(sorted(i - offset for i in f)) for f in mesh.faces if min(f) >= offset
            }
            with self.subTest(shape=part.shape):
                self.assertTrue(outer == inner)

    def test_outstretched_arms_clear_the_front_edges_of_the_cape(self):
        scene = build_character({"pose": "arms_out", "cloak": "knee", "height_m": 1})
        parts = {p.name: p for p in scene.parts}
        garment = build(parts["Плащ"], "render")
        arm = build(parts["Рука левая"], "render")
        torso = build(parts["Туловище"], "render")
        shoulder = [depths_at(m, 0.114, 0.82) for m in (arm, torso)]
        self.assertTrue(all(len(hits) for hits in shoulder))
        self.assertLess(
            max(hits.min() for hits in shoulder), min(hits.max() for hits in shoulder)
        )
        for x in (0.14, 0.15, 0.16, 0.17):
            z = 0.823
            arm_depths = depths_at(arm, x, z)
            cape_depths = depths_at(garment, x, z)
            self.assertGreater(arm_depths.size, 0)
            self.assertTrue((cape_depths > arm_depths.max()).all())
        self.assertEqual(Scene.parse(scene.data()), scene)

    def test_tunic_covers_the_leading_thigh_above_the_hem_when_walking(self):
        parts = {
            p.name: p
            for p in build_character({"height_m": 1, "pose": "walking", "cloak": "knee"}).parts
        }
        garment = build(parts["Туловище"], "render")
        thigh = build(parts["Нога левая"], "render")
        self.assertLess(
            depths_at(garment, 0.057, 0.442).min(), depths_at(thigh, 0.057, 0.442).min()
        )

    def test_back_of_torso_is_beneath_the_cape_from_neck_to_waist(self):
        for length in ("knee", "ankle", "short"):
            parts = {p.name: p for p in build_character({"height_m": 1, "cloak": length}).parts}
            for z in (0.65, 0.72, 0.78, 0.82, 0.84, 0.853):
                rears = []
                for name in ("Туловище", "Плащ"):
                    mesh = build(parts[name], "render")
                    rears.append(depths_at(mesh, 0, z).max())
                with self.subTest(length=length, z=z):
                    self.assertGreater(rears[1], rears[0])

    def test_cape_covers_both_shoulders_and_clasp_reaches_its_open_edges(self):
        for length in ("short", "knee", "ankle"):
            scene = build_character({"height_m": 1, "cloak": length})
            parts = {p.name: p for p in scene.parts}
            cape = build(parts["Плащ"], "render")
            clasp = parts["Застёжка плаща"]
            crossings = []
            for mesh in (cape, build(parts["Рука левая"], "render")):
                tri = mesh.vertices[list(triangles(mesh.faces))]
                # Intersect the triangles with a horizontal plane passing through the shoulder.
                edge = tri[:, [1, 2, 0]] - tri
                dz = edge[:, :, 2]
                t = (0.819 - tri[:, :, 2]) / np.where(dz != 0, dz, 1)
                section = (tri + t[:, :, None] * edge)[(dz != 0) & (t >= 0) & (t <= 1)]
                crossings.append(section[:, 0].max())
            with self.subTest(length=length):
                self.assertGreater(crossings[0], crossings[1])
                # Sample the large front panels between rings; a brooch holds both flaps.
                tri = cape.vertices[list(triangles(cape.faces))]
                a, b = tri, tri[:, [1, 2, 0]]
                samples = np.concatenate([a + t * (b - a) for t in np.linspace(0, 1, 16)])
                inside = (((samples - clasp.center) / (np.asarray(clasp.size) / 2)) ** 2).sum(
                    axis=2
                ) < 1
                self.assertLess(samples[inside][:, 0].min(initial=0), 0)
                self.assertGreater(samples[inside][:, 0].max(initial=0), 0)
            self.assertIn("Пояс", parts)

    def test_cloak_hood_proportions_and_height_follow_request(self):
        character = build_character({"height_m": 1.6, "cloak": "knee", "hood": True})
        parts = {p.name: p for p in character.parts}
        self.assertEqual(len(parts), len(character.parts))
        self.assertEqual(parts["Плащ"].shape, "cloak")
        self.assertEqual(parts["Капюшон"].shape, "hood")
        self.assertGreater(parts["Туловище"].size[0], parts["Голова"].size[0])
        self.assertGreater(parts["Голова"].center[2], parts["Туловище"].center[2])
        self.assertAlmostEqual(max(p.center[2] + p.size[2] / 2 for p in parts.values()), 1.6)
        self.assertAlmostEqual(min(p.center[2] - p.size[2] / 2 for p in parts.values()), 0)
        plain = build_character({})
        self.assertFalse({"cloak", "hood"} & {p.shape for p in plain.parts})
        for length in ("short", "knee", "ankle"):
            coat = next(
                p for p in build_character({"cloak": length}).parts if p.shape == "cloak"
            )
            self.assertAlmostEqual(coat.size[2] + 2 * coat.center[2], 1.75 * 2 * 0.86)
        widths = [
            build_character({"build": b}).parts[0].size[0] for b in ("slim", "average", "broad")
        ]
        self.assertTrue(widths[0] < widths[1] < widths[2])
        pose = {p.name: p for p in build_character({"pose": "arms_out"}).parts}
        self.assertGreater(
            abs(pose["Кисть левая"].center[0]), abs(parts["Кисть левая"].center[0]) * 2
        )

    def test_character_part_controls_round_trip_and_specs_are_bounded(self):
        scene = build_character({"pose": "walking", "cloak": "ankle"}, "high")
        self.assertEqual(scene.detail, "high")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "character.xtri"
            save_project(path, scene, "render")
            self.assertEqual(load_project(path), (scene, "render"))
        for spec in (
            [],
            {"height_m": float("nan")},
            {"height_m": True},
            {"height_m": 3.1},
            {"hood": "yes"},
            {"hood": True, "cloak": "none"},
            {"cloak": []},
            {"pose": "unknown"},
            {"build": False},
            {"extra": 1},
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                build_character(spec)
        for h in (0.3, 3.0):
            scene = build_character(
                {
                    "height_m": h,
                    "pose": "walking",
                    "build": "broad",
                    "cloak": "ankle",
                    "hood": True,
                }
            )
            self.assertEqual(Scene.parse(json.loads(json.dumps(scene.data()))), scene)

    def test_closed_oriented_meshes_on_actual_anatomy_and_garments(self):
        scene = build_character({"cloak": "ankle", "hood": True})
        thrown_back = next(
            p
            for p in build_character({"cloak": "knee", "pose": "arms_out"}).parts
            if p.shape == "cloak"
        )
        shapes = {p.shape for p in scene.parts}
        self.assertTrue(set(ORGANIC_SHAPES) <= shapes)
        for n in (8, 16, 24, 32, 48, 64):
            for p in (*scene.parts, thrown_back):
                if p.shape not in ORGANIC_SHAPES:
                    continue
                with self.subTest(part=p.name, segments=n):
                    v, faces, smooth = organic_mesh(p, n)
                    v = np.asarray(v)
                    self.assertEqual(len(faces), len(smooth))
                    self.assertEqual(len(set(map(tuple, v))), len(v))
                    edges = Counter((a, b) for f in faces for a, b in zip(f, (*f[1:], f[0])))
                    self.assertTrue(
                        all(num == 1 and edges[b, a] == 1 for (a, b), num in edges.items())
                    )
                    tri = v[list(triangles(faces))]
                    self.assertLessEqual(len(tri), organic_triangle_bound(p, n))
                    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    self.assertGreater(np.linalg.norm(cross, axis=1).min(), 1e-10)
                    self.assertGreater((tri[:, 0] * cross).sum() / 6, 0)
                    np.testing.assert_allclose(v.min(axis=0), [-0.5] * 3, atol=1e-9)
                    np.testing.assert_allclose(v.max(axis=0), [0.5] * 3, atol=1e-9)

    def test_garments_leave_open_space_at_front(self):
        parts = build_character({"cloak": "knee", "hood": True}).parts
        for shape in ("cloak", "hood"):
            part = next(p for p in parts if p.shape == shape)
            v, faces, _ = organic_mesh(part, 32)
            tri = np.asarray(v)[list(triangles(faces))]
            from_front = tri[(tri[:, :, 1].mean(axis=1) < -0.12)]
            # The central front of the figure must be visible through the opening.
            for test_x in (-0.04, 0.04):
                for test_z in (-0.1, 0.1):
                    p = np.array([test_x, test_z])
                    a, b, c = from_front[:, :, [0, 2]].transpose(1, 0, 2)

                    def det(v, w):
                        return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

                    sides = np.array([det(b - a, p - a), det(c - b, p - b), det(a - c, p - c)])
                    self.assertFalse(((sides > 0).all(axis=0) | (sides < 0).all(axis=0)).any())

    def test_invalid_loft_sections_rejected_before_meshing(self):
        loft = next(p for p in build_character({}).parts if p.shape == "loft").data()
        for rows in (
            None,
            [],
            [[0, 0.5, 0.5, 0, 0]] * 3,
            [[0, 0.5, 0.5, 0, 0], [1e-300, 0.5, 0.5, 0, 0], [1, 0, 0, 0, 0]],
            [[0, 1e-300, 1e-300, 1, 1], [0.5, 1e-300, 1e-300, 1, 1], [1, 0, 0, 1, 1]],
            [[0, 0, 0.5, 0, 0], [0.5, 0.5, 0.5, 0, 0], [1, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0], [1, 0.5, 0.5, 0, 0]],
            [[0, 0.5, 0.5, 0, 0], [0.5, float("nan"), 0.5, 0, 0], [1, 0.5, 0.5, 0, 0]],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Part.parse(loft | {"sections": rows})
