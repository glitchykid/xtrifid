import os
import unittest
from dataclasses import replace
from threading import Event, get_ident
from unittest.mock import patch

import numpy as np
from OpenGL import GL
from test_studio import until
from test_textures import box

from xtrifid.geometry import Mesh, build, rotation_matrix
from xtrifid.gtk import Gio, Gtk
from xtrifid.textures import TextureSpec
from xtrifid.viewport import Viewport


def material_part(shape="cylinder", *, roughness=0.5, metallic=0.0, strength=1.0):
    recipe = TextureSpec.parse(
        {
            "material": "fabric",
            "color_a": "#997548",
            "color_b": "#997548",
            "resolution": 128,
            "pbr": {"roughness": roughness, "metallic": metallic, "normal_strength": strength},
        }
    )
    return replace(box(), shape=shape, size=(1, 1, 1), center=(0, 0, 0), texture=recipe)


class CornerFrameTests(unittest.TestCase):
    def test_end_caps_have_opposite_handedness_and_identical_image_up(self):
        from xtrifid.tangents import corner_frames

        part = material_part()
        mesh = build(part, "game", "low")
        indices, normals, uv, tangents = corner_frames(mesh, part)
        points = mesh.vertices[indices]
        caps = np.ptp(points[:, :, 2], axis=1) == 0
        self.assertEqual(set(np.unique(tangents[caps, :, 3])), {-1.0, 1.0})
        self.assertTrue((tangents[:, :, 3] == tangents[:, :1, 3]).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(tangents[:, :, :3], axis=-1), 1, atol=1e-6)
        np.testing.assert_allclose((normals * tangents[:, :, :3]).sum(axis=-1), 0, atol=1e-6)
        b = np.cross(normals, tangents[:, :, :3]) * tangents[:, :, 3:]
        np.testing.assert_allclose(b[caps] - (0, 1, 0), 0, atol=1e-6)
        np.testing.assert_allclose(tangents[caps, :, :3] - (1, 0, 0), 0, atol=1e-6)
        np.testing.assert_allclose(uv[caps], points[caps, :, :2] + 0.5)
        # Smooth side quads share vertex normals even along their texture seam.
        for vertex in np.unique(indices[~caps]):
            corners = (indices == vertex) & ~caps[:, None]
            ns = normals[corners]
            np.testing.assert_allclose(ns - ns[0], 0, atol=1e-6)

    def test_cloth_keeps_rest_uvs_and_rotates_both_normal_and_tangent(self):
        from xtrifid.tangents import corner_frames

        part = material_part("box")
        mesh = build(part, "render")
        rot = rotation_matrix((33, -29, 71))
        _, normals, uv, tangents = corner_frames(mesh, part)
        moved = replace(mesh, vertices=mesh.vertices @ rot.T + (4, -3, 2))
        _, new_normals, new_uv, new_tangents = corner_frames(moved, part, rest_mesh=mesh)
        np.testing.assert_array_equal(uv, new_uv)
        np.testing.assert_allclose(new_normals, normals @ rot.T, atol=1e-6)
        np.testing.assert_allclose(
            new_tangents[:, :, :3], tangents[:, :, :3] @ rot.T, atol=1e-6
        )
        np.testing.assert_array_equal(tangents[:, :, 3], new_tangents[:, :, 3])
        with self.assertRaises(ValueError):
            corner_frames(replace(mesh, faces=mesh.faces[::-1]), part, rest_mesh=mesh)

    def test_empty_mesh_collapsed_uv_and_degenerate_geometry_produce_finite_frames(self):
        from xtrifid.tangents import corner_frames

        part = material_part("box")
        mesh = Mesh(np.empty((0, 3)), (), ())
        self.assertEqual(
            [a.shape for a in corner_frames(mesh, part)],
            [(0, 3), (0, 3, 3), (0, 3, 2), (0, 3, 4)],
        )
        mesh = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]), ((0, 1, 2),), (True,))
        with np.errstate(invalid="raise", divide="raise"):
            _, normals, uv, tangents = corner_frames(mesh, part)
        self.assertTrue(np.isfinite(uv).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1)
        np.testing.assert_allclose(np.linalg.norm(tangents[:, :, :3], axis=-1), 1)
        np.testing.assert_allclose((normals * tangents[:, :, :3]).sum(axis=-1), 0)


@unittest.skipUnless(os.getenv("DISPLAY"), "requires X11/OpenGL")
class MaterialPreviewTests(unittest.TestCase):
    def setUp(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid." + self._testMethodName,
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        self.window = Gtk.ApplicationWindow(
            application=app, default_width=380, default_height=380
        )
        self.addCleanup(self.window.close)
        self.errors = []
        self.view = Viewport(lambda *_: None, self.errors.append)
        self.window.set_child(self.view)
        self.window.present()
        until(lambda: self.view.drawn or self.errors)
        self.assertEqual(self.errors, [])

    def render(self, part):
        view = self.view
        mesh = build(part, "game", "high")
        view.set_meshes([mesh], [part])
        view.selected = 0
        view.yaw = view.pitch = 0
        view.zoom = 1.45
        return self.read()

    def read(self):
        view = self.view
        view.make_current()
        view.attach_buffers()
        view.render_mesh(view, view.get_context())
        w, h = map(int, GL.glGetIntegerv(GL.GL_VIEWPORT)[2:])
        data = GL.glReadPixels(
            w // 2 - 60, h // 2 - 60, 120, 120, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        self.assertEqual(self.errors, [])
        return np.frombuffer(data, np.uint8).reshape(120, 120, 3).astype(int)

    def test_real_material_relief_metal_and_roughness_change_frame(self):
        flat = self.render(material_part("sphere", strength=0))
        relief = self.render(material_part("sphere", strength=2))
        metal = self.render(material_part("sphere", strength=0, metallic=1, roughness=0.14))
        rough = self.render(material_part("sphere", strength=0, metallic=1, roughness=0.90))
        for a, b in ((flat, relief), (flat, metal), (metal, rough)):
            with self.subTest(difference=np.abs(a - b).mean()):
                self.assertGreater(np.abs(a - b).mean(), 1)
        part = material_part("sphere", strength=2)
        mesh = build(part, "game", "high")
        self.render(part)
        self.view.set_simulation_preview(mesh, True)
        np.testing.assert_array_equal(self.read(), relief)
        self.view.set_simulation_preview(None)
        np.testing.assert_array_equal(self.read(), relief)

    def test_positive_green_tilts_front_face_toward_overhead_light(self):
        def tilted(green, seed):
            part = material_part(strength=1, roughness=0.75)
            part = replace(part, texture=replace(part.texture, seed=seed))
            image = np.empty((128, 128, 3), dtype=np.uint8)
            image[:] = (128, green, 224)
            with patch("xtrifid.viewport.normal_pixels", return_value=image):
                return self.render(part)[45:75, 45:75].mean()

        self.assertGreater(tilted(212, 1), tilted(44, 2) + 10)

    def test_large_map_can_be_replaced_while_gtk_remains_responsive(self):
        from xtrifid.textures import normal_pixels

        large = material_part()
        large = replace(large, texture=replace(large.texture, resolution=512))
        replacement = material_part("sphere", strength=0)
        expected = self.render(replacement)
        entered, resume = Event(), Event()
        gui_thread = get_ident()

        def slow_map(spec):
            if spec == replacement.texture:
                return normal_pixels(spec)
            self.assertNotEqual(get_ident(), gui_thread)
            entered.set()
            self.assertTrue(resume.wait(5))
            return np.full((spec.resolution, spec.resolution, 3), 128, np.uint8)

        with patch("xtrifid.viewport.normal_pixels", side_effect=slow_map):
            try:
                self.render(large)
                until(entered.is_set)
                self.render(replacement)
            finally:
                resume.set()
            until(lambda: replacement.texture in self.view.textures or self.errors)
        np.testing.assert_array_equal(self.read(), expected)
        self.assertNotIn(large.texture, self.view.textures)

    def test_partial_driver_upload_releases_names_and_an_explicit_assignment_can_retry(self):
        part = material_part()
        mesh = build(part, "game", "high")
        view = self.view
        view.set_meshes([mesh], [part])
        view.make_current()
        view.attach_buffers()
        allocate, upload = GL.glGenTextures, GL.glTexImage2D
        names = []

        def track(*args):
            name = allocate(*args)
            names.append(name)
            return name

        def fail_normal(*args):
            if len(names) == 2:
                raise MemoryError("test GPU has no room for the normal map")
            return upload(*args)

        with (
            patch.object(GL, "glGenTextures", side_effect=track),
            patch.object(GL, "glTexImage2D", side_effect=fail_normal),
        ):
            for _ in range(2):
                view.render_mesh(view, view.get_context())
        self.assertEqual(len(self.errors), 1)
        self.assertEqual(len(names), 2)
        self.assertFalse(any(GL.glIsTexture(name) for name in names))
        self.assertNotIn(part.texture, view.textures)
        self.errors.clear()
        # Setting the material again is a deliberate retry after memory is freed.
        view.set_meshes([mesh], [part])
        self.read()
        self.assertIn(part.texture, view.textures)
