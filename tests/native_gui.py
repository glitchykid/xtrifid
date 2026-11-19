"""Exercise a release ELF from an unrelated directory on a private X11 display.

Usage: python3 tests/native_gui.py /absolute/path/to/xtrifid-linux-x86_64 [--generate]
The external driver uses xvfb-run, dbus-run-session, python3-pyatspi, python3-xlib
and python3-pil. For --generate, select a running Ollama with OLLAMA_URL or
point OLLAMA_MODELS at the existing model cache for the bundled server.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROMPT = (
    "Создай один прямоугольный брусок без других деталей. Размер по X 40 см, "
    "по Y 20 см, по Z 10 см. Без поворота, нижняя грань лежит на полу."
)


def wait_for(probe, description, process, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Client exited with status {process.returncode}: {description}")
        if result := probe():
            return result
        time.sleep(0.1)
    raise TimeoutError(description)


def descendants(widget):
    for child in widget:
        if child is not None:
            yield child
            yield from descendants(child)


def lookup(container, role, name):
    return next(
        (w for w in descendants(container) if w.getRoleName() == role and w.name == name),
        None,
    )


def check_obj(path, expected_size, expected_faces):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields[:1] == ["v"]:
            vertices.append(tuple(map(float, fields[1:])))
        if fields[:1] == ["f"]:
            faces.append(tuple(map(int, fields[1:])))
    assert len(faces) == expected_faces, len(faces)
    assert {len(f) for f in faces} == {3}
    assert all(1 <= i <= len(vertices) for f in faces for i in f)
    edges = Counter((a, b) for f in faces for a, b in zip(f, (*f[1:], f[0])))
    assert all(a != b and n == 1 and edges[b, a] == 1 for (a, b), n in edges.items())
    size = [round(max(axis) - min(axis), 5) for axis in zip(*vertices)]
    assert size == expected_size, size
    print(
        f"OBJ {path.name}: {len(vertices)} vertices, {len(faces)} triangles, {size} m",
        flush=True,
    )


def stage_session(binary, generate):
    with tempfile.TemporaryDirectory(
        prefix="xtrifid-native-", ignore_cleanup_errors=True
    ) as tmp:
        work = Path(tmp)
        for sub in ("home", "runtime", "working directory", "projects", "exports"):
            (work / sub).mkdir(mode=0o700)
        env = os.environ.copy()
        for key in (
            "AT_SPI_BUS_ADDRESS",
            "DBUS_SESSION_BUS_ADDRESS",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "PYTHONPATH",
            "PYTHONHOME",
            "GTK_IM_MODULE",
        ):
            env.pop(key, None)
        env.update(
            HOME=str(work / "home"),
            XDG_RUNTIME_DIR=str(work / "runtime"),
            XDG_CONFIG_HOME=str(work / "home/config"),
            XDG_DATA_HOME=str(work / "home/data"),
            XDG_CACHE_HOME=str(work / "home/cache"),
            XDG_STATE_HOME=str(work / "home/state"),
            GDK_BACKEND="x11",
            GTK_A11Y="atspi",
            GSETTINGS_BACKEND="memory",
            XTRIFID_FILES=str(work / "exports"),
            LC_ALL="C.UTF-8",
            LANGUAGE="en",
        )
        return subprocess.run(
            [
                "xvfb-run",
                "-a",
                "-s",
                "-screen 0 1280x900x24 -nolisten tcp",
                "dbus-run-session",
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                str(binary),
                "--session",
                str(work),
                *(["--generate"] if generate else []),
            ],
            cwd=work / "working directory",
            env=env,
            check=False,
        ).returncode


def exercise(binary, work, generate):
    import pyatspi as at
    from PIL import ImageGrab
    from Xlib import X, display
    from Xlib.ext import xtest

    fixture = work / "projects" / "Test bushing with spaces.xtri"
    fixture.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "render",
                "parts": [
                    {
                        "name": "Test bushing",
                        "shape": "tube",
                        "size": [0.08, 0.08, 0.06],
                        "center": [0, 0, 0.03],
                        "rotation": [0, 0, 0],
                        "inner_diameter": 0.04,
                    }
                ],
            }
        )
    )
    process = subprocess.Popen([binary, fixture], stdin=subprocess.DEVNULL)
    x = display.Display()

    def click_menu_item(popover, item):
        # GTK4 exposes the popup as a separate X11 surface; its AT-SPI child
        # positions are relative to the containing GTK window.
        surface = next(
            w
            for w in reversed(x.screen().root.query_tree().children)
            if w.get_attributes().map_state == X.IsViewable
            and w.get_attributes().override_redirect
        )
        origin = surface.get_geometry()
        parent = popover.queryComponent().getExtents(at.WINDOW_COORDS)
        bounds = item.queryComponent().getExtents(at.WINDOW_COORDS)
        xtest.fake_input(
            x,
            X.MotionNotify,
            x=origin.x + bounds.x - parent.x + bounds.width // 2,
            y=origin.y + bounds.y - parent.y + bounds.height // 2,
        )
        for event in (X.ButtonPress, X.ButtonRelease):
            xtest.fake_input(x, event, 1)
        x.sync()

    def click(widget):
        actions = widget.queryAction()
        assert actions.nActions > 0 and actions.doAction(0), widget.name

    def expect(root, role, name, seconds=20):
        return wait_for(lambda: lookup(root, role, name), f"{role} {name}", process, seconds)

    def mint_pixels():
        sample = ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).crop((500, 185, 1010, 560))
        return sum(g > 105 and g > 1.35 * r and b > 1.3 * r for r, g, b in sample.getdata())

    def preview():
        count = wait_for(lambda: c if (c := mint_pixels()) > 1500 else 0, "3D preview", process)
        print(f"Preview: {count} mesh-coloured pixels in the viewport", flush=True)

    def export(name, size, faces):
        click(expect(frame, "push button", "Экспорт OBJ"))
        click(expect(desktop, "push button", "Save"))
        output = work / "exports/model.obj"
        wait_for(lambda: output.is_file() and output.stat().st_size > 0, str(output), process)
        target = output.rename(output.with_name(name))
        check_obj(target, size, faces)

    try:
        desktop = at.Registry.getDesktop(0)
        frame = expect(desktop, "frame", "Xtrifid")
        expect(frame, "combo box", "1. Test bushing")
        preview()
        click(expect(frame, "toggle button", "Для рендера"))
        popover = expect(frame, "list box", "")
        click_menu_item(popover, expect(popover, "label", "Game Ready"))
        expect(frame, "combo box", "Game Ready")
        export("bushing.obj", [0.08, 0.08, 0.06], 128)
        if generate:
            field = expect(frame, "text", "Описание новой модели")
            field.queryEditableText().setTextContents(PROMPT)
            click(expect(frame, "push button", "Создать модель"))
            expect(
                frame,
                "label",
                "Модель готова. Выберите деталь в окне просмотра или в списке.",
                260,
            )
            preview()
            export("block.obj", [0.4, 0.2, 0.1], 12)
        click(expect(frame, "push button", "Close"))
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        x.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--session", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.session:
        exercise(args.binary, args.session, args.generate)
    else:
        sys.exit(stage_session(args.binary.resolve(strict=True), args.generate))
