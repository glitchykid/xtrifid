import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from xtrifid.sculpt import mesh, parse_masks, parse_nodes, triangle_bound

TREE = (
    ("основание", -1, 0, 0, -0.8, 0.18),
    ("развилка", 0, 0, 0, 0.05, 0.14),
    ("левый побег", 1, -0.6, 0.12, 0.65, 0.09),
    ("правый побег", 1, 0.65, -0.2, 0.8, 0.075),
)
RING = ("0011100", "0111110", "1110111", "1100011", "1110111", "0111110", "0011100")
SIDE = ("0011100", "0111110", "1111111", "1111111", "0111110", "0011100", "0001000")


def surface(part, mode="render", detail="low", **options):
    vertices, faces, smooth = mesh(part, mode, detail, **options)
    assert len(smooth) == len(faces)
    return np.asarray(vertices), faces


class SculptTests(unittest.TestCase):
    def test_rounded_body_has_smooth_normals_across_raster_scan_lines(self):
        from xtrifid.geometry import triangles

        sample = (np.arange(63) + 0.5) / 63 - 0.5
        mask = tuple(
            "".join("1" if x * x + z * z < 0.25 else "0" for x in sample) for z in sample
        )
        part = SimpleNamespace(
            shape="reference", rounded=True, masks=(mask, mask), size=(1.0, 1.0, 1.0)
        )
        vertices, faces = surface(part, "render", "high")
        points = vertices[np.array(list(triangles(faces)))]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        weights = np.linalg.norm(normals, axis=1)
        centres = points.mean(axis=1)
        radii = np.linalg.norm(centres, axis=1)
        cosine = (normals * centres).sum(axis=1) / (weights * radii)
        error = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        self.assertLess(np.average(error, weights=weights), 5.2)
        self.assertLess(np.max(abs(radii - 0.5)), 0.025)

    def test_optional_rounded_sections_turn_a_rectangular_projection_into_a_round_body(self):
        masks = parse_masks([["1" * 16] * 16, ["1" * 16] * 16])
        for views in (masks[:1], masks):
            part = SimpleNamespace(
                shape="reference", rounded=True, masks=views, size=(1.0, 1.0, 1.0)
            )
            vertices, faces = surface(part, "render", "medium")
            belt = vertices[abs(vertices[:, 2]) < 0.05]
            self.assertGreater(len(belt), 20)
            distances = np.linalg.norm(belt[:, :2], axis=1)
            np.testing.assert_allclose(distances, 0.5, atol=0.024)
            self.assertEqual(self.assert_closed(vertices, faces), 2)
            self.assertEqual(
                sum(len(f) - 2 for f in faces), triangle_bound(part, "render", "medium")
            )

    def test_rounded_sections_keep_through_holes_and_limit_a_secondary_lobes_depth(self):
        full = "1" * 20
        split = "1" * 10 + "0" * 7 + "1" * 3
        pierced = (full,) * 3 + (split,) * 10 + (full,) * 3
        wide = (full,) * 16
        for masks, secondary_axis in [((pierced, wide), 0), ((wide, pierced), 1)]:
            part = SimpleNamespace(
                shape="reference", rounded=True, masks=masks, size=(1.0, 1.0, 1.0)
            )
            vertices, faces = surface(part, "render", "medium")
            belt = vertices[abs(vertices[:, 2]) < 0.09]
            appendage = belt[belt[:, secondary_axis] > 0.34]
            self.assertGreater(len(appendage), 8)
            self.assertLess(np.ptp(appendage[:, 1 - secondary_axis]), 0.22)
            self.assertFalse(
                np.any((belt[:, secondary_axis] > 0.08) & (belt[:, secondary_axis] < 0.28))
            )
            self.assertEqual(self.assert_closed(vertices, faces), 0)

    def test_low_poly_keeps_a_live_three_way_junction_lighter_than_standard(self):
        part = SimpleNamespace(
            shape="sculpt",
            nodes=parse_nodes(
                [
                    ("ствол", -1, -0.1, 0, -0.8, 0.25),
                    ("лево", 0, -0.6, 0.1, -0.4, 0.12),
                    ("право", 0, 0.6, 0.1, -0.4, 0.12),
                    ("назад", 0, -0.1, 0.3, -0.4, 0.12),
                ]
            ),
        )
        standard = triangle_bound(part, "render", "medium")
        vertices, faces = surface(part, "render", "medium", low_poly=True)
        self.assertLess(sum(len(f) - 2 for f in faces), standard)
        self.assertEqual(self.assert_closed(vertices, faces), 2)

    def test_solid_rectangular_silhouettes_need_only_six_planar_quads(self):
        part = SimpleNamespace(
            shape="reference", rounded=False, masks=parse_masks([["1111"] * 3])
        )
        for low_poly in (False, True):
            vertices, faces = surface(part, "render", "high", low_poly=low_poly)
            self.assertEqual((len(vertices), len(faces)), (8, 6))
            self.assertTrue(all(len(f) == 4 for f in faces))
            self.assertEqual(self.assert_closed(vertices, faces), 2)
            self.assertEqual(triangle_bound(part, "render", "high", low_poly), 12)

    def test_an_unambiguous_tunnel_is_built_entirely_with_shared_quads(self):
        part = SimpleNamespace(
            shape="reference", rounded=False, masks=parse_masks((RING, SIDE))
        )
        vertices, faces = surface(part, "render", "medium")
        self.assertTrue(all(len(f) == 4 for f in faces))
        self.assertEqual(self.assert_closed(vertices, faces), 0)
        self.assertLess(len(faces), 7500)

    def test_simulated_scalar_field_keeps_index_coordinates_and_ignores_empty_frames(self):
        from xtrifid.sculpt import isosurface, isosurface_triangle_count

        x, y, z = np.ogrid[:9, :11, :13]
        field = 1.5 - np.sqrt((x - 3) ** 2 + (y - 4) ** 2 + (z - 7) ** 2)
        vertices, faces = isosurface(field)
        np.testing.assert_allclose(vertices.min(0), [1.5, 2.5, 5.5])
        np.testing.assert_allclose(vertices.max(0), [4.5, 5.5, 8.5])
        self.assertEqual(isosurface_triangle_count(field), sum(len(f) - 2 for f in faces))
        self.assertEqual(self.assert_closed(vertices, faces), 2)
        empty = np.full((5, 6, 7), -1.0)
        vertices, faces = isosurface(empty)
        self.assertEqual(vertices.shape, (0, 3))
        self.assertEqual(faces, [])
        with self.assertRaises(ValueError):
            isosurface(field, max_triangles=1)

    def test_scene_round_trip_and_real_obj_export_follow_mode_and_style(self):
        from xtrifid.geometry import build
        from xtrifid.model import Part, Scene
        from xtrifid.storage import export_obj, load_project, save_project

        part = Part.parse(
            {
                "name": "Побег",
                "shape": "sculpt",
                "nodes": TREE,
                "size": [0.6, 0.2, 0.5],
                "center": [0, 0, 0.25],
                "rotation": [0, 0, 0],
            }
        )
        scene = Scene((part,), "medium")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "organic.xtri"
            save_project(path, scene, "render")
            restored, mode = load_project(path)
            self.assertEqual((restored, mode), (scene, "render"))
            render = build(part, "render", scene.detail)
            self.assertTrue(any(len(f) == 4 for f in render.faces))
            game = build(part, "game", scene.detail, "low_poly")
            self.assertTrue(all(len(f) == 3 for f in game.faces))
            self.assertFalse(any(game.smooth))
            self.assertLess(len(game.faces), len(render.faces))
            obj = Path(directory) / "organic.obj"
            export_obj(obj, scene, "render")
            records = obj.read_text().splitlines()
            self.assertEqual(sum(line.startswith("f ") for line in records), len(render.faces))

    def assert_closed(self, vertices, faces):
        directed = Counter((a, b) for face in faces for a, b in zip(face, (*face[1:], face[0])))
        self.assertTrue(
            all(count == 1 and directed[(b, a)] == 1 for (a, b), count in directed.items())
        )
        volume = 0
        for face in faces:
            a = vertices[face[0]]
            for i in range(1, len(face) - 1):
                b, c = vertices[face[i]], vertices[face[i + 1]]
                self.assertGreater(np.linalg.norm(np.cross(b - a, c - a)), 0)
                volume += a @ np.cross(b, c) / 6
        self.assertGreater(volume, 0)
        return len(vertices) - len(directed) // 2 + len(faces)

    def test_a_branch_is_a_single_closed_oriented_surface_at_both_levels(self):
        part = SimpleNamespace(shape="sculpt", nodes=parse_nodes(TREE))
        counts = []
        for mode, detail in [("game", "low"), ("render", "medium")]:
            vertices, faces = surface(part, mode, detail)
            np.testing.assert_allclose(np.ptp(vertices, axis=0), [1, 1, 1])
            self.assertEqual(self.assert_closed(vertices, faces), 2)
            count = sum(len(f) - 2 for f in faces)
            self.assertLessEqual(count, triangle_bound(part, mode, detail))
            counts.append(count)
            tips = vertices[vertices[:, 2] > 0.32]
            self.assertTrue(np.any(tips[:, 0] < -0.24) and np.any(tips[:, 0] > 0.24))
            self.assertFalse(np.any(abs(tips[:, 0]) < 0.12))
        self.assertGreater(counts[1], counts[0])

    def test_front_hole_survives_single_and_paired_views(self):
        for masks in [(RING,), (RING, SIDE)]:
            part = SimpleNamespace(shape="reference", rounded=False, masks=parse_masks(masks))
            vertices, faces = surface(part, "render", "low")
            self.assertEqual(self.assert_closed(vertices, faces), 0)
            gpu_positions = (vertices + 1).astype("f4")
            self.assertEqual(len(np.unique(gpu_positions, axis=0)), len(vertices))
            self.assertFalse(
                np.any((abs(vertices[:, 0]) < 0.065) & (abs(vertices[:, 2]) < 0.065))
            )
            if len(masks) == 2:
                crown = vertices[vertices[:, 2] > 0.46]
                self.assertLess(np.ptp(crown[:, 1]), 0.65)

    def test_reject_damaged_or_unbounded_recipes(self):
        for raw in [
            [TREE[0]],
            [TREE[0], ("B", 1, 0, 0, 0.5, 0.1)],
            [TREE[0], ("B", -1, 0, 0, 0.5, 0.1)],
            [TREE[0], ("B", 0, 0, 0, 0.5, float("nan"))],
            TREE * 7,
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_nodes(raw)
        for raw in [[], [("00", "01x")], [("1" * 97,) * 2], [("00", "00")], [("101",) * 98]]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_masks(raw)
