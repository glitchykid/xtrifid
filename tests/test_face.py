import unittest
from collections import defaultdict

import numpy as np

from xtrifid.character import build_character
from xtrifid.geometry import build, triangles
from xtrifid.human_asset import rest_mesh, rest_triangle_count
from xtrifid.human_body import character_points
from xtrifid.model import Part


def components(faces):
    adjacent = defaultdict(set)
    for face in faces:
        for a, b in zip(face, (*face[1:], face[0])):
            adjacent[a].add(b)
            adjacent[b].add(a)
    groups = []
    while adjacent:
        todo = [next(iter(adjacent))]
        group = []
        while todo:
            vertex = todo.pop()
            if vertex in adjacent:
                todo.extend(adjacent.pop(vertex))
                group.append(vertex)
        groups.append(group)
    return sorted(groups, key=len, reverse=True)


def ray_depths(vertices, faces, xs, zs):
    """Intersect frontal rays with the nearest facial surface."""
    tri = vertices[list(triangles(faces))]
    tri = tri[(tri[:, :, 2].max(axis=1) >= min(zs)) & (tri[:, :, 2].min(axis=1) <= max(zs))]
    a, b, c = tri.transpose(1, 0, 2)
    u, w = b - a, c - a
    det = u[:, 0] * w[:, 2] - u[:, 2] * w[:, 0]
    valid = abs(det) > 1e-12
    denominator = np.where(valid, det, 1)
    samples = []
    for z in zs:
        for x in xs:
            dx, dz = x - a[:, 0], z - a[:, 2]
            s = (dx * w[:, 2] - dz * w[:, 0]) / denominator
            t = (u[:, 0] * dz - u[:, 2] * dx) / denominator
            hit = valid & (s >= 0) & (t >= 0) & (s + t <= 1)
            samples.append((a[:, 1] + s * u[:, 1] + t * w[:, 1])[hit].min(initial=np.inf))
    return np.array(samples).reshape(len(zs), len(xs))


def frontal_depths(level):
    v, faces = rest_mesh(level)
    return ray_depths(v, faces, np.linspace(-0.036, 0.036, 24), np.linspace(0.874, 0.947, 24))


class AnatomicalFaceTests(unittest.TestCase):
    def test_ears_face_and_feet_share_skin_with_two_fitted_eyeballs_at_every_lod(self):
        for level in (8, 16, 24, 32, 48, 64):
            with self.subTest(level=level):
                vertices, faces = rest_mesh(level)
                skin, left, right = components(faces)
                body = vertices[skin]
                self.assertGreater(np.ptp(body[:, 2]), 0.99)
                for side in (-1, 1):
                    ear = body[
                        (side * body[:, 0] > 0.04) & (body[:, 2] > 0.90) & (body[:, 2] < 0.95)
                    ]
                    self.assertGreater(len(ear), 12)
                    self.assertGreater(np.ptp(ear[:, 1]), 0.014)
                    self.assertGreater(np.ptp(ear[:, 2]), 0.025)
                eyes = sorted((vertices[left], vertices[right]), key=lambda v: v[:, 0].mean())
                for side, eye in zip((-1, 1), eyes):
                    center = (eye.max(axis=0) + eye.min(axis=0)) / 2
                    self.assertTrue(0.014 < side * center[0] < 0.025)
                    self.assertTrue(0.91 < center[2] < 0.94)
                    self.assertTrue(0.012 < np.ptp(eye[:, 0]) < 0.019)
                self.assertEqual(sum(len(f) - 2 for f in faces), rest_triangle_count(level))

    def test_detail_increases_and_render_preserves_editable_facial_quads(self):
        for levels in ((8, 16, 32), (24, 48, 64)):
            counts = [rest_triangle_count(level) for level in levels]
            self.assertTrue(counts[0] < counts[1] < counts[2] < 150_000)
        for level in (48, 64):
            vertices, faces = rest_mesh(level)
            cheeks_and_lips = [f for f in faces if 0.865 < vertices[f[0], 2] < 0.903]
            quads = sum(len(face) == 4 for face in cheeks_and_lips)
            self.assertGreater(quads, len(cheeks_and_lips) * 0.95)

    def test_game_reduction_preserves_visible_facial_relief(self):
        reference = frontal_depths(48)
        for level, tolerance_mm in ((8, 1.5), (16, 0.6), (32, 0.3)):
            with self.subTest(level=level):
                reduced = frontal_depths(level)
                hit = np.isfinite(reference) & np.isfinite(reduced)
                self.assertGreater(hit.mean(), 0.94)
                error_mm = 1750 * np.abs(reduced[hit] - reference[hit])
                self.assertLess(np.percentile(error_mm, 95), tolerance_mm)

    def test_neutral_reduced_skin_remains_bilaterally_symmetric(self):
        for n in (8, 16, 24, 32):
            with self.subTest(n=n):
                v, _ = rest_mesh(n)
                points = set(map(tuple, np.rint(v * 1e7).astype(int)))
                self.assertTrue(all((-x, y, z) in points for x, y, z in points))

    def test_regional_face_edits_are_local_and_do_not_turn_triangles_inside_out(self):
        # Check displacement, topology and orientation on both sides of the
        # control, including the coarsest mesh and the subdivided render mesh.
        for mode, detail, n in (("game", "low", 8), ("render", "high", 64)):
            base = build_character({}).parts[0]
            mesh = build(base, mode, detail)
            rest, _ = rest_mesh(n)
            x, y, z = rest.T
            indices = np.array(list(triangles(mesh.faces)))
            tri = mesh.vertices[indices]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            for zone, limits in (
                ("nose", (0.893, 0.927)),
                ("lips", (0.876, 0.898)),
                ("chin", (0.846, 0.883)),
            ):
                permitted = (abs(x) < 0.036) & (y < 0.012) & (z > limits[0]) & (z < limits[1])
                for scale in (0.75, 1.25):
                    with self.subTest(zone=zone, scale=scale, n=n):
                        part = build_character({"body_scales": {zone: scale}}).parts[0]
                        self.assertEqual(Part.parse(part.data()), part)
                        edited = build(part, mode, detail)
                        self.assertEqual(mesh.faces, edited.faces)
                        displacement = edited.vertices - mesh.vertices
                        np.testing.assert_array_equal(displacement[~permitted], 0)
                        # Larger features project further from the face and vice versa.
                        forward = -displacement[:, 1] * np.sign(scale - 1)
                        self.assertGreater(forward.max(), 0.001)
                        self.assertLess(np.linalg.norm(displacement, axis=1).max(), 0.010)
                        moved = edited.vertices[indices]
                        new_normals = np.cross(
                            moved[:, 1] - moved[:, 0], moved[:, 2] - moved[:, 0]
                        )
                        self.assertGreater(np.einsum("ij,ij->i", normals, new_normals).min(), 0)

    def test_closed_nondegenerate_outward_surfaces_after_reduction(self):
        for n in (8, 16, 24, 32, 48, 64):
            with self.subTest(n=n):
                vertices, faces = rest_mesh(n)
                tri = vertices[list(triangles(faces))]
                normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                self.assertGreater(np.min(np.linalg.norm(normal, axis=1)), 1e-13)
                self.assertGreater(np.sum(tri[:, 0] * normal), 0)
                edges = [(a, b) for face in faces for a, b in zip(face, (*face[1:], face[0]))]
                directed = set(edges)
                self.assertEqual(len(directed), len(edges))
                self.assertTrue(all((b, a) in directed for a, b in directed))

    def test_resting_lips_meet_softly_and_eyeballs_fill_the_lid_openings(self):
        # Inspect physical occlusion in ray space, independent of viewport lighting.
        for level in (48, 64):
            rest, faces = rest_mesh(level)
            for physique in ("neutral", "male", "female"):
                with self.subTest(level=level, physique=physique):
                    part = build_character({"physique": physique}).parts[0]
                    v = character_points(part, rest, f"lod{level}")
                    for x in (0.002, 0.006, 0.010, 0.013):
                        rim = (x, -0.025, 0.8871)
                        vertex = np.argmin(np.sum((rest - rim) ** 2, axis=1))
                        offset = v[vertex] - rest[vertex]
                        zs = np.linspace(0.8841, 0.8901, 181) + offset[2]
                        lips = ray_depths(v, faces, [x + offset[0]], zs)
                        opening_mm = np.sum(lips > 0.005 + offset[1]) * (zs[1] - zs[0]) * 1750
                        self.assertLessEqual(opening_mm, 0.22)
                    for side in (-1, 1):
                        eye_center = (side * 0.018, -0.016, 0.928)
                        vertex = np.argmin(np.sum((rest - eye_center) ** 2, axis=1))
                        offset = v[vertex] - rest[vertex]
                        eyes = ray_depths(
                            v,
                            faces,
                            side * np.linspace(0.01, 0.027, 18) + offset[0],
                            np.linspace(0.924, 0.931, 18) + offset[2],
                        )
                        self.assertLess(eyes.max(), -0.009 + offset[1])

    def test_unknown_level_fails_before_reading_geometry(self):
        with self.assertRaises(ValueError):
            rest_mesh(0)
