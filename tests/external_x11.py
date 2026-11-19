"""Check model pixels as presented by the external X server in external_x11.sh."""

import struct
from pathlib import Path

import numpy as np
from test_model import spec
from test_studio import until

from xtrifid.gtk import Gio, Gtk
from xtrifid.model import Scene
from xtrifid.ui import Studio


def framebuffer():
    # Xvfb -fbdir exposes a live XWD version 7 image. Header fields are network order.
    path = Path("/display/Xvfb_screen0")
    with path.open("rb") as f:
        header = struct.unpack(">25I", f.read(100))
    size, version, format_, depth, width, height = header[:6]
    byte_order, bpp, stride, ncolors = header[7], header[11], header[12], header[19]
    assert (version, format_, depth, bpp) == (7, 2, 24, 32)
    assert header[14:17] == (0xFF0000, 0xFF00, 0xFF)
    pixels = np.memmap(
        path,
        ">u4" if byte_order else "<u4",
        "r",
        offset=size + 12 * ncolors,
        shape=(height, stride // 4),
    )
    return pixels[:, :width]


def main():
    app = Gtk.Application(
        application_id="io.github.glitchykid.Xtrifid.ExternalX11",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )
    app.register(None)
    window = Studio(app)
    # This Xvfb has no window manager. Its only undecorated toplevel is at (0, 0).
    window.set_decorated(False)
    window.present()
    try:
        part = spec()
        window.install(Scene.parse({"parts": [part]}), "game")
        until(lambda: window.view.drawn or window.view.problem or window.view.get_error())
        assert not window.view.problem and not window.view.get_error()
        ok, bounds = window.view.compute_bounds(window)
        assert ok
        dx, dy = window.view.screen_position(part["center"])
        x, y = int(bounds.get_x() + dx), int(bounds.get_y() + dy)
        screen = framebuffer()
        sample = screen[y - 10 : y + 10, x - 10 : x + 10]
        assert sample.shape == (20, 20)

        def has_selected_mesh():
            red, green, blue = [(sample >> n) & 255 for n in (16, 8, 0)]
            mint = (green > 105) & (green > 1.35 * red) & (blue > 1.3 * red)
            return np.count_nonzero(mint) > sample.size * 0.9

        try:
            until(has_selected_mesh)
        finally:
            print(f"Presented model center RGB at ({x}, {y}): #{screen[y, x] & 0xFFFFFF:06x}")
    finally:
        window.close()


if __name__ == "__main__":
    main()
