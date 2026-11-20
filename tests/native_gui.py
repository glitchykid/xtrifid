"""Exercise a release ELF from an unrelated directory on a private X11 display.

Usage: python3 tests/native_gui.py /absolute/path/to/xtrifid-linux-x86_64 [--generate]
The external driver uses xvfb-run, dbus-run-session, python3-pyatspi, python3-xlib
and python3-pil. The default run covers the reference image chooser, textured
Low Poly OBJ export and a baked cloth frame entirely offline.
For --generate, select a running Ollama with OLLAMA_URL or
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
    from gi.repository import GLib

    try:
        return next(
            (w for w in descendants(container) if w.getRole() == role and w.name == name),
            None,
        )
    except GLib.Error as error:
        # GTK may destroy a popup row between enumeration and the remote AT-SPI read.
        if "does not exist" in error.message:
            return None
        raise


def check_obj(path, expected_size=None, expected_faces=None, *, expected_euler=None):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields[:1] == ["v"]:
            vertices.append(tuple(map(float, fields[1:])))
        if fields[:1] == ["f"]:
            faces.append(tuple(int(corner.split("/")[0]) for corner in fields[1:]))
    if expected_faces is not None:
        assert len(faces) == expected_faces, len(faces)
    assert {len(f) for f in faces} == {3}
    assert all(1 <= i <= len(vertices) for f in faces for i in f)
    edges = Counter((a, b) for f in faces for a, b in zip(f, (*f[1:], f[0])))
    assert all(a != b and n == 1 and edges[b, a] == 1 for (a, b), n in edges.items())
    if expected_euler is not None:
        assert len(vertices) - len(edges) // 2 + len(faces) == expected_euler
    size = [round(max(axis) - min(axis), 5) for axis in zip(*vertices)]
    if expected_size is not None:
        assert size == expected_size, size
    print(
        f"OBJ {path.name}: {len(vertices)} vertices, {len(faces)} triangles, {size} m",
        flush=True,
    )
    return size


def check_material(obj, resolution):
    from PIL import Image

    records = [line.split() for line in obj.read_text().splitlines() if line]
    uvs = [tuple(map(float, r[1:])) for r in records if r[0] == "vt"]
    faces = [r[1:] for r in records if r[0] == "f"]
    assert (
        faces
        and uvs
        and all(1 <= int(corner.split("/")[1]) <= len(uvs) for face in faces for corner in face)
    )
    assert {r[1] for r in records if r[0] == "s"} == {"off"}
    (library,) = [obj.parent / r[1] for r in records if r[0] == "mtllib"]
    used = {r[1] for r in records if r[0] == "usemtl"}
    maps = {}
    for line in library.read_text().splitlines():
        if line.startswith("newmtl "):
            current = line.split()[1]
        elif line.startswith("map_Kd "):
            maps[current] = library.parent / line[7:]
    assert len(used) == 1
    with Image.open(maps[used.pop()]) as rgb:
        assert (rgb.format, rgb.size) == ("PNG", (resolution, resolution))
        red, green, blue = rgb.convert("RGB").getextrema()
        assert red[1] - red[0] > 12 and red[0] > max(green[1], blue[1])
    print(
        f"Material verified: {len(uvs)} UVs, {resolution} × {resolution} red fabric", flush=True
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
    from PIL import Image, ImageDraw, ImageGrab
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

    def pointer_click(widget):
        # Some GTK4 controls have bounds but no remote action. Resolve their window
        # on our private X display and click the same visible target as a user.
        top = widget
        while top.getRole() not in (at.ROLE_FRAME, at.ROLE_DIALOG):
            top = top.parent
        root = x.screen().root
        wm_name = x.intern_atom("_NET_WM_NAME")
        window = next(
            win
            for win in reversed(root.query_tree().children)
            if win.get_attributes().map_state == X.IsViewable
            and (title := win.get_full_property(wm_name, X.AnyPropertyType))
            and title.value.decode("utf-8") == top.name
        )
        origin = root.translate_coords(window, 0, 0)
        bounds = widget.queryComponent().getExtents(at.WINDOW_COORDS)
        xtest.fake_input(
            x,
            X.MotionNotify,
            x=origin.x + bounds.x + bounds.width // 2,
            y=origin.y + bounds.y + bounds.height // 2,
        )
        for event in (X.ButtonPress, X.ButtonRelease):
            xtest.fake_input(x, event, 1)
        x.sync()

    def click(widget):
        actions = widget.queryAction()
        if not (actions.nActions and actions.doAction(0)):
            pointer_click(widget)

    def expect(root, role, name, seconds=20):
        return wait_for(lambda: lookup(root, role, name), f"{role} {name}", process, seconds)

    def mesh_pixels(portrait):
        sample = ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).crop((500, 185, 1010, 560))
        if portrait:
            return sum(r > 95 and r >= g >= b and r - b > 3 for r, g, b in sample.getdata())
        return sum(g > 105 and g > 1.35 * r and b > 1.3 * r for r, g, b in sample.getdata())

    def preview(portrait=False):
        count = wait_for(
            lambda: c if (c := mesh_pixels(portrait)) > 1500 else 0, "3D preview", process
        )
        print(f"Preview: {count} mesh-coloured pixels in the viewport", flush=True)

    def save_as(name):
        ext = Path(name).suffix
        click(
            expect(frame, at.ROLE_PUSH_BUTTON, "Экспорт OBJ" if ext == ".obj" else "Сохранить")
        )
        click(expect(desktop, at.ROLE_PUSH_BUTTON, "Save"))
        output = work / "exports" / ("model" + ext)
        wait_for(lambda: output.is_file() and output.stat().st_size > 0, str(output), process)
        return output.rename(output.with_name(name))

    def reference_workshop():
        path = work / "exports" / "Front ring.png"
        image = Image.new("RGBA", (96, 96))
        drawing = ImageDraw.Draw(image)
        drawing.ellipse((16, 8, 79, 87), fill="navy")
        drawing.ellipse((34, 28, 61, 65), fill=(0, 0, 0, 0))
        image.save(path)
        click(expect(frame, at.ROLE_CHECK_BOX, "Low Poly"))
        click(expect(frame, at.ROLE_PUSH_BUTTON, "По референсам…"))
        dialog = expect(desktop, at.ROLE_FRAME, "Создать по референсам")
        click(expect(dialog, at.ROLE_PUSH_BUTTON, "Вид спереди"))
        chooser = expect(desktop, at.ROLE_DIALOG, "Выберите изображение PNG или JPEG")
        cell = expect(chooser, at.ROLE_TABLE_CELL, path.name)
        pointer_click(cell)
        click(expect(chooser, at.ROLE_PUSH_BUTTON, "Open"))
        expect(dialog, at.ROLE_LABEL, path.name)
        build = expect(dialog, at.ROLE_PUSH_BUTTON, "Построить модель")
        wait_for(lambda: build.getState().contains(at.STATE_SENSITIVE), "PNG decoded", process)
        click(build)
        expect(frame, at.ROLE_LABEL, "Модель по референсу готова")
        expect(frame, at.ROLE_COMBO_BOX, "1. Front ring")
        click(expect(frame, at.ROLE_PUSH_BUTTON, "Текстура детали"))
        field = expect(frame, at.ROLE_TEXT, "Описание текстуры")
        field.queryEditableText().setTextContents("Красная ткань #a80000 #ff1808")
        click(expect(frame, at.ROLE_PUSH_BUTTON, "Назначить"))
        expect(frame, at.ROLE_LABEL, "Текстура применена")

        def red_pixels():
            sample = ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).crop((500, 185, 1010, 560))
            return sum(r > 65 and r > 3 * max(g, b) for r, g, b in sample.getdata()) > 1500

        wait_for(red_pixels, "Textured model renders red in OpenGL", process)
        ring = save_as("textured-ring.obj")
        check_obj(ring, [0.8, 0.44, 1], expected_euler=0)
        check_material(ring, 256)
        click(expect(frame, at.ROLE_PUSH_BUTTON, "Физика"))
        slider = expect(frame, at.ROLE_SLIDER, "Кадр симуляции")
        slider.queryValue().currentValue = 6
        expect(frame, at.ROLE_LABEL, "Кадр 6 / 180    0.20 с")
        click(expect(frame, at.ROLE_PUSH_BUTTON, "Запечь кадр в модель"))
        expect(frame, at.ROLE_COMBO_BOX, "2. Ткань")
        saved = json.loads(save_as("textured-ring-and-cloth.xtri").read_text())
        assert saved["style"] == "low_poly"
        ring, cloth = saved["parts"]
        assert ring["shape"] == "reference" and [round(v, 5) for v in ring["size"]] == [
            0.8,
            0.44,
            1,
        ]
        (mask,) = ring["masks"]
        assert "1" in mask[len(mask) // 2] and mask[len(mask) // 2][len(mask[0]) // 2] == "0"
        assert ring["texture"]["material"] == "fabric" and cloth["shape"] == "surface"
        assert cloth["surface"]["vertices"] and cloth["surface"]["faces"]
        print("Front PNG hole, Low Poly and cloth bake preserved in the project", flush=True)

    try:
        desktop = at.Registry.getDesktop(0)
        frame = expect(desktop, at.ROLE_FRAME, "Xtrifid")
        expect(frame, at.ROLE_COMBO_BOX, "1. Test bushing")
        preview()
        click(expect(frame, at.ROLE_TOGGLE_BUTTON, "Для рендера"))
        popover = expect(frame, at.ROLE_LIST_BOX, "")
        click_menu_item(popover, expect(popover, at.ROLE_LABEL, "Game Ready"))
        expect(frame, at.ROLE_COMBO_BOX, "Game Ready")
        check_obj(save_as("bushing.obj"), [0.08, 0.08, 0.06], 128)
        click(expect(frame, at.ROLE_TOGGLE_BUTTON, "Средняя"))
        popover = expect(frame, at.ROLE_LIST_BOX, "")
        click_menu_item(popover, expect(popover, at.ROLE_LABEL, "Низкая"))
        expect(frame, at.ROLE_COMBO_BOX, "Низкая")
        field = expect(frame, at.ROLE_TEXT, "Описание новой модели")
        field.queryEditableText().setTextContents("сделай человека с плащом")
        click(expect(frame, at.ROLE_PUSH_BUTTON, "Создать модель"))
        expect(frame, at.ROLE_COMBO_BOX, "1. Человек")
        click(expect(frame, at.ROLE_PUSH_BUTTON, "К лицу"))
        preview(portrait=True)
        figure = save_as("character.obj")
        size = check_obj(figure)
        assert size[2] == 1.75 and any(
            line.startswith("o ") and line.endswith("_Плащ")
            for line in figure.read_text().splitlines()
        )
        if generate:
            field.queryEditableText().setTextContents(PROMPT)
            click(expect(frame, at.ROLE_PUSH_BUTTON, "Создать модель"))
            expect(
                frame,
                at.ROLE_LABEL,
                "Модель готова. Выберите деталь в окне просмотра или в списке.",
                260,
            )
            preview()
            check_obj(save_as("block.obj"), [0.4, 0.2, 0.1], 12)
        reference_workshop()
        click(expect(frame, at.ROLE_PUSH_BUTTON, "Close"))
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
