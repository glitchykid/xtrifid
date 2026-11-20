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


def depths_at(mesh, x, z, *, vertical=False):
    """Intersect a ray through (x, z) with the actual mesh triangles."""
    vertices = mesh.vertices[:, (0, 2, 1)] if vertical else mesh.vertices
    a, b, c = vertices[list(triangles(mesh.faces))].transpose(1, 0, 2)
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

    def test_skin_is_the_same_with_or_without_a_requested_cloak(self):
        for pose in ("standing", "walking", "arms_out"):
            with self.subTest(pose=pose):
                bare = build_character({"pose": pose})
                caped = build_character({"pose": pose, "cloak": "knee"})
                self.assertEqual([p.shape for p in bare.parts], ["human"])
                self.assertEqual(caped.parts[:-1], bare.parts)
                self.assertEqual(caped.parts[-1].shape, "cloak")

    def test_pose_changes_arm_span_and_walking_stride_while_keeping_the_height(self):
        from xtrifid.character import character_anchor

        people = {
            pose: build_character({"pose": pose, "height_m": 1.6}, "low")
            for pose in ("standing", "walking", "arms_out")
        }
        hands = {
            pose: character_anchor(person, "Кисть левая") for pose, person in people.items()
        }
        self.assertGreater(hands["arms_out"][0], hands["standing"][0] * 2)
        self.assertGreater(hands["arms_out"][2], hands["standing"][2] + 0.25)
        soles = {}
        for pose, person in people.items():
            body = build(person.parts[0], "game", "low").vertices
            self.assertAlmostEqual(np.ptp(body[:, 2]), 1.6, delta=0.012)
            self.assertAlmostEqual(body[:, 2].min(), 0, delta=0.012)
            soles[pose] = [
                body[(body[:, 0] * side > 0) & (body[:, 2] < 0.1), 1].mean() for side in (-1, 1)
            ]
            # The reference point lies inside the palm for placing accessories.
            distances = np.linalg.norm(body - hands[pose], axis=1)
            self.assertLess(distances.min(), 0.035)
        self.assertGreater(abs(np.diff(soles["walking"])[0]), 0.15)
        self.assertLess(abs(np.diff(soles["standing"])[0]), 0.015)

    def test_local_ear_edit_leaves_the_body_and_pose_unchanged(self):
        person = build_character({"pose": "walking"}).parts[0]
        changed = Part.parse(person.data() | {"body_scales": {"ears": 1.24}})
        original = build(person, "render").vertices
        edited = build(changed, "render").vertices
        off_face = original[:, 2] < person.size[2] * 0.86
        np.testing.assert_array_equal(original[off_face], edited[off_face])
        self.assertGreater(abs(original - edited).max(), 0.001)
        for scales in ({"head": 1.3}, {"head": True}, {"arms": 2}, [], {"waist": None}):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                Part.parse(person.data() | {"body_scales": scales})

    def test_cloak_hood_proportions_and_height_follow_request(self):
        character = build_character({"height_m": 1.6, "cloak": "knee", "hood": True})
        self.assertEqual([p.shape for p in character.parts], ["human", "cloak", "hood"])
        self.assertAlmostEqual(max(p.center[2] + p.size[2] / 2 for p in character.parts), 1.6)
        self.assertAlmostEqual(min(p.center[2] - p.size[2] / 2 for p in character.parts), 0)
        for length in ("short", "knee", "ankle"):
            cape = build_character({"height_m": 1, "cloak": length}).parts[-1]
            mesh = build(cape, "render")
            skin = build(build_character({"height_m": 1}).parts[0], "render")
            for z in (0.63, 0.68, 0.73, 0.77, 0.8):
                self.assertGreater(depths_at(mesh, 0, z).max(), depths_at(skin, 0, z).max())
        widths = [
            build_character({"build": b}).parts[0].size[0] for b in ("slim", "average", "broad")
        ]
        self.assertTrue(widths[0] < widths[1] < widths[2])

    def test_reposing_preserves_placement_and_rejects_oversize_results(self):
        from xtrifid.character import with_character_pose

        base = build_character({"height_m": 1.9, "build": "broad"}).parts[0]
        posed = with_character_pose(base, "walking")
        generated = build_character(
            {"height_m": 1.9, "build": "broad", "pose": "walking"}
        ).parts[0]
        np.testing.assert_allclose(posed.size, generated.size)
        np.testing.assert_allclose(posed.center, generated.center)
        moved = Part.parse(base.data() | {"center": [3, 4, 5], "rotation": [10, 35, 20]})
        restored = with_character_pose(with_character_pose(moved, "arms_out"), "standing")
        np.testing.assert_allclose(restored.size, moved.size)
        np.testing.assert_allclose(restored.center, moved.center)
        huge = Part.parse(base.data() | {"size": [999, 1, 1]})
        with self.assertRaises(ValueError):
            with_character_pose(huge, "arms_out")

    def test_shoulder_skin_stays_under_the_cape_for_each_physique(self):
        for physique in ("neutral", "male", "female"):
            scene = build_character(
                {"height_m": 1, "cloak": "knee", "hood": True, "physique": physique}
            )
            skin, cape = [build(p, "render") for p in scene.parts[:2]]
            for x in (0.061, 0.084, 0.100):
                for y in (0.0, 0.017, 0.031):
                    with self.subTest(physique=physique, x=x, y=y):
                        self.assertGreater(
                            depths_at(cape, x, y, vertical=True).max(),
                            depths_at(skin, x, y, vertical=True).max(),
                        )

    def test_collar_rests_on_both_shoulders(self):
        for hood in (False, True):
            scene = build_character({"height_m": 1, "cloak": "knee", "hood": hood})
            skin, cloth = [build(part, "render").vertices for part in scene.parts[:2]]
            neck = skin[(skin[:, 2] > 0.72) & (skin[:, 2] < 0.88)]
            for side in (-1, 1):
                collar = cloth[
                    (cloth[:, 2] > cloth[:, 2].max() - 0.025) & (side * cloth[:, 0] > 0.025)
                ]
                distances = np.linalg.norm(collar[:, None] - neck[None], axis=2).min(axis=1)
                with self.subTest(hood=hood, side=side):
                    self.assertLess(np.percentile(distances, 25), 0.008)

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
        self.assertEqual({"human", "cloak", "hood"}, shapes)
        for n in (8, 16, 24, 32, 48, 64):
            for p in (*scene.parts, thrown_back):
                if p.shape == "human":
                    continue  # Topology of the authored anatomy is checked at each LOD separately.
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
        loft = {
            "name": "Старая деталь",
            "shape": "loft",
            "size": [1, 1, 1],
            "rotation": [0, 0, 0],
            "center": [0, 0, 0],
        }
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
