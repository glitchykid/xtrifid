import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from xtrifid.geometry import build, triangles
from xtrifid.model import Part, Scene
from xtrifid.profile import PROFILE_SHAPES, move_profile_point, profile_guide, profile_points
from xtrifid.storage import export_obj, load_project, save_project


def part(shape, **options):
    return Part.parse(
        {
            "shape": shape,
            "name": "Корпус",
            "size": [2, 4, 6],
            "center": [3, 6, 9],
            "rotation": [90, 0, 90],
            **options,
        }
    )


class ProfileTests(unittest.TestCase):
    def test_physical_points_match_local_dimensions_for_every_editable_shape(self):
        square = ((-1, -2), (1, -2), (1, 2), (-1, 2))
        axis_section = ((0, -3), (1, -3), (1, 3), (0, 3))
        tube_section = ((0.25, -3), (1, -3), (1, 3), (0.25, 3))
        cases = [
            (part("box"), square),
            (
                part("extrusion", outline=[[-0.8, 0.1], [0.1, 0.1], [0.1, 0.5], [-0.8, 0.5]]),
                square,
            ),
            (part("cone"), ((0, -3), (1, -3), (0, 3))),
            (part("cylinder"), axis_section),
            (part("tube", inner_diameter=0.5), tube_section),
            (
                part("lathe", outline=[[0.125, -0.2], [0.5, -0.2], [0.5, 0.6], [0.125, 0.6]]),
                tube_section,
            ),
        ]
        self.assertEqual({p.shape for p, _ in cases}, set(PROFILE_SHAPES))
        for p, expected in cases:
            with self.subTest(shape=p.shape):
                self.assertEqual(profile_points(p), expected)
                for i, point in enumerate(expected):
                    self.assertIs(move_profile_point(p, i, point), p)

    def test_changed_extreme_moves_only_one_edge_of_rotated_box(self):
        old = part("box")
        new = move_profile_point(old, 0, (-1.5, -3.25))
        self.assertEqual(new.shape, "extrusion")
        self.assertEqual(new.name, old.name)
        self.assertEqual(new.rotation, old.rotation)
        self.assertEqual(new.size, (2.5, 5.25, 6))
        np.testing.assert_allclose(new.center, (3, 5.75, 8.375), atol=1e-14)
        for mode in ("game", "render"):
            expected = build(old, mode).vertices.copy()
            # A 90-degree turn around X followed by one around Z maps (x,y,z) to (z,x,y).
            expected[[0, 4]] += (0, -0.5, -1.25)
            np.testing.assert_allclose(build(new, mode).vertices, expected, atol=3e-14)

    def test_changing_one_elliptical_tube_ring_retains_bore_and_other_world_rings(self):
        old = part("tube", inner_diameter=0.5)
        new = move_profile_point(old, 2, (1.25, 4.0))
        self.assertEqual(new.shape, "lathe")
        self.assertIsNone(new.inner_diameter)
        self.assertEqual(new.size, (2.5, 5, 7))
        np.testing.assert_allclose(new.center, (3.5, 6, 9))
        for mode, n in (("game", 16), ("render", 48)):
            before = build(old, mode).vertices.reshape(4, n, 3)
            after = build(new, mode).vertices.reshape(4, n, 3)
            np.testing.assert_allclose(after[[0, 1, 3]], before[[0, 1, 3]], atol=3e-14)
            np.testing.assert_allclose(after[2, 0], (7, 7.25, 9), atol=3e-14)
            np.testing.assert_allclose(after[2, n // 4], (7, 6, 11.5), atol=3e-14)

    def test_deformation_persists_and_exports_a_closed_surface_in_both_modes(self):
        old = part(
            "extrusion", outline=[[0, 0], [1, 0], [1, 0.3], [0.3, 0.3], [0.3, 1], [0, 1]]
        )
        points = profile_points(old)
        new = move_profile_point(old, 3, (-0.25, -0.5))
        expected = list(points)
        expected[3] = (-0.25, -0.5)
        np.testing.assert_allclose(profile_points(new), expected)
        self.assertEqual(new.center, old.center)
        scene = Scene((new, old))
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("game", "render"):
                file = Path(tmp) / "profile.xtri"
                save_project(file, scene, mode)
                loaded, restored_mode = load_project(file)
                self.assertEqual((loaded, restored_mode), (scene, mode))
                mesh = build(loaded.parts[0], mode)
                edges = Counter(
                    (a, b) for face in mesh.faces for a, b in zip(face, face[1:] + face[:1])
                )
                self.assertTrue(
                    all(count == 1 and edges[b, a] == 1 for (a, b), count in edges.items())
                )
                tri = mesh.vertices[list(triangles(mesh.faces))]
                normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                self.assertGreater(np.linalg.norm(normals, axis=1).min(), 0)
                self.assertGreater((tri[:, 0] * normals).sum(), 0)
                obj = Path(tmp) / "profile.obj"
                export_obj(obj, loaded, mode)
                data = obj.read_text().split("o 02_", 1)[0].splitlines()
                verts = [
                    tuple(map(float, line.split()[1:]))
                    for line in data
                    if line.startswith("v ")
                ]
                np.testing.assert_allclose(verts, mesh.vertices)
                faces = [
                    tuple(int(k) - 1 for k in line.split()[1:])
                    for line in data
                    if line.startswith("f ")
                ]
                self.assertEqual(tuple(faces), mesh.faces)

    def test_rejects_malformed_candidates_and_invalid_geometry(self):
        box = part("box")
        triangle = part("extrusion", outline=[[0, 0], [1, 0], [0, 1]])
        invalid = [
            (box, False, (0, 0)),
            (box, -1, (0, 0)),
            (box, 4, (0, 0)),
            (box, 1, None),
            (box, 0, (True, 0)),
            (box, 0, (0, float("nan"))),
            (box, 0, (0, float("inf"))),
            (box, 0, (-1, 2)),
            (box, 0, (1.5, 0)),
            (box, 0, (-1100, 0)),
            (triangle, 2, (-1, -3)),  # Would reverse winding.
            (part("cone"), 1, (-0.1, 0)),
            (part("cylinder"), 0, (0.2, -3)),
            (part("box", center=[1000, 0, 0], rotation=[0, 0, 0]), 2, (2, 2)),
        ]
        for p, index, point in invalid:
            with (
                self.subTest(shape=p.shape, index=index, point=point),
                self.assertRaises(ValueError),
            ):
                move_profile_point(p, index, point)
        for shape in ("sphere", "torus"):
            with self.assertRaises(ValueError):
                profile_points(part(shape))

    def test_guide_draws_actual_rotated_edges_rings_and_poles_in_both_modes(self):
        for mode, n in (("game", 16), ("render", 48)):
            box = move_profile_point(part("box"), 1, (1.2, -2.5))
            mesh = build(box, mode)
            np.testing.assert_array_equal(profile_guide(box, 1, mode), mesh.vertices[[1, 5]])
            for turned in (
                part("cone"),
                part("cylinder"),
                part("tube", inner_diameter=0.5),
                part("lathe", outline=[[0.5, 0], [1, 0], [1, 0.5], [0.5, 1]]),
            ):
                mesh = build(turned, mode)
                offset = 0
                for i, (radius, _) in enumerate(profile_points(turned)):
                    with self.subTest(mode=mode, shape=turned.shape, index=i):
                        guide = profile_guide(turned, i, mode)
                        if radius == 0:
                            self.assertEqual(guide.shape, (1, 3))
                            np.testing.assert_array_equal(guide[0], mesh.vertices[offset])
                            offset += 1
                        else:
                            self.assertEqual(guide.shape, (n + 1, 3))
                            np.testing.assert_array_equal(
                                guide[:-1], mesh.vertices[offset : offset + n]
                            )
                            np.testing.assert_array_equal(guide[0], guide[-1])
                            offset += n
                self.assertEqual(offset, len(mesh.vertices))
        for index, mode in ((-1, "render"), (True, "game"), (0, "unknown")):
            with self.assertRaises(ValueError):
                profile_guide(part("box"), index, mode)

    def test_micro_scale_parts_still_validate_in_normalized_profile_space(self):
        old = part("cone", size=(0.001, 0.003, 0.001))
        new = move_profile_point(old, 1, (0.0006, -0.0004))
        self.assertEqual(new.shape, "lathe")
        np.testing.assert_allclose(new.size, (0.0012, 0.0036, 0.001))
        self.assertEqual(new.center, old.center)
