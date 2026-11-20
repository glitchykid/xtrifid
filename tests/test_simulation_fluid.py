import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from test_simulation import support

from xtrifid.geometry import build, triangles
from xtrifid.model import Scene
from xtrifid.simulation import Simulation
from xtrifid.storage import export_obj, load_project, save_project
from xtrifid.surface import baked_part


class FluidTests(unittest.TestCase):
    def test_volume_drops_as_one_mass_hits_the_support_and_spreads(self):
        part = support()
        sim = Simulation("fluid", part, build(part, "render"), "medium", wind=0)
        start = sim.mesh()
        xyz = start.vertices
        edges = {}
        for face in start.faces:
            for a, b in zip(face, (*face[1:], face[0])):
                edges.setdefault(a, set()).add(b)
                edges.setdefault(b, set()).add(a)
        visited, queue = set(), [0]
        while queue:
            vertex = queue.pop()
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(edges[vertex] - visited)
        self.assertEqual(len(visited), len(xyz))
        top = part.center[2] + part.size[2] / 2
        self.assertGreater(xyz[:, 2].min(), top + 0.035)
        sim.seek(6)
        self.assertLess(sim.mesh().vertices[:, 2].mean(), xyz[:, 2].mean() - 0.045)
        sim.seek(25)
        splat = sim.mesh()
        self.assertGreater(np.ptp(splat.vertices[:, 0]), 1.3 * np.ptp(xyz[:, 0]))
        self.assertLess(splat.vertices[:, 2].min(), top + 0.006)
        # Runoff can drop below the tabletop along its edges; the interior
        # stays outside the solid throughout the collision.
        over_table = (np.abs(splat.vertices[:, 0] - part.center[0]) < part.size[0] * 0.45) & (
            np.abs(splat.vertices[:, 1] - part.center[1]) < part.size[1] * 0.45
        )
        self.assertGreater(splat.vertices[over_table, 2].min(), top - 0.003)
        self.assertGreater(splat.vertices[:, 2].min(), part.center[2] - part.size[2] / 2)
        tri = splat.vertices[list(triangles(splat.faces))]
        self.assertGreater(
            np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            ).min(),
            0,
        )
        sim.seek(10)
        sim.seek(25)
        np.testing.assert_array_equal(sim.mesh().vertices, splat.vertices)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "splash.xtri"
            scene = Scene((part, baked_part("Жидкость", splat)), "medium")
            save_project(path, scene, "render")
            loaded, _ = load_project(path)
            recovered = build(loaded.parts[1], "render", "medium")
            np.testing.assert_allclose(recovered.vertices, splat.vertices, atol=1e-10)
            self.assertEqual(recovered.faces, splat.faces)
            obj = Path(temp) / "splash.obj"
            export_obj(obj, loaded, "render")
            self.assertIn("o 02_Жидкость", obj.read_text())

    def test_preflight_rejects_an_intractably_small_droplet_before_play(self):
        target = replace(support(), size=(0.001, 0.001, 0.001))
        with self.assertRaisesRegex(ValueError, "мал"):
            Simulation("fluid", target, build(target, "game"), "high", strength=2)
