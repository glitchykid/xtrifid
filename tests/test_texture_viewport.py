import os
import unittest
from dataclasses import replace

import numpy as np
from OpenGL import GL
from test_studio import until
from test_textures import box

from xtrifid.geometry import build
from xtrifid.gtk import Gio, Gtk
from xtrifid.texture_panel import TexturePanel
from xtrifid.textures import TextureSpec
from xtrifid.viewport import Viewport


@unittest.skipUnless(os.getenv("DISPLAY"), "requires X11/OpenGL")
class TextureViewportTests(unittest.TestCase):
    def test_pixels_on_actual_model_simulation_overlay_and_panel_remove(self):
        app = Gtk.Application(
            application_id="io.github.glitchykid.Xtrifid.TextureTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        app.register(None)
        window = Gtk.ApplicationWindow(application=app, default_width=370, default_height=370)
        self.addCleanup(window.close)
        errors = []
        view = Viewport(lambda *_: None, errors.append)
        window.set_child(view)
        window.present()
        until(lambda: view.drawn)
        spec = TextureSpec.parse(
            {
                "material": "fabric",
                "color_a": "#800000",
                "color_b": "#ff1200",
                "resolution": 128,
            }
        )
        part = replace(box(), texture=spec)
        mesh = build(part, "render")
        view.set_meshes([mesh], [part])
        view.selected = 0
        view.yaw = view.pitch = 0
        view.zoom = 1.5

        def sample():
            view.make_current()
            view.attach_buffers()
            view.render_mesh(view, view.get_context())
            w, h = map(int, GL.glGetIntegerv(GL.GL_VIEWPORT)[2:])
            return (
                np.frombuffer(
                    GL.glReadPixels(
                        w // 2 - 12, h // 2 - 12, 24, 24, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
                    ),
                    np.uint8,
                )
                .reshape(-1, 3)
                .astype(int)
            )

        pixels = sample()
        self.assertTrue((pixels[:, 0] > pixels[:, 1] * 3).all())
        self.assertGreater(np.ptp(pixels[:, 0]), 10)
        shifted = replace(mesh, vertices=mesh.vertices + np.array([2, 0, 0]))
        before = view.buffer
        view.set_simulation_preview(mesh, True)
        np.testing.assert_array_equal(sample(), pixels)
        view.set_simulation_preview(shifted, True)
        self.assertLess(sample()[:, 0].mean(), 40)
        self.assertIs(before, view.buffer)
        view.set_simulation_preview(None, False)
        np.testing.assert_array_equal(sample(), pixels)
        self.assertEqual(errors, [])
        assigned = []
        panel = TexturePanel(assigned.append, errors.append)
        panel.set_part(part)
        panel.remove_button.emit("clicked")
        self.assertEqual(assigned, [None])
        panel.set_part(None)
        self.assertFalse(panel.get_sensitive())
