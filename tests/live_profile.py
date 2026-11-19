"""Exercise a local profile edit via GTK and Ollama using an existing v1 project."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from test_studio import until

from xtrifid.gtk import Gio, Gtk
from xtrifid.storage import export_obj, load_project, save_project
from xtrifid.ui import Studio


def main(folder):
    app = Gtk.Application(
        application_id="io.github.glitchykid.Xtrifid.ProfileAcceptance",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )
    app.register(None)
    window = Studio(app)
    window.present()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "v1.xtri"
            source.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "mode": "render",
                        "parts": [
                            {
                                "name": "Втулка",
                                "shape": "tube",
                                "size": [0.08, 0.08, 0.06],
                                "inner_diameter": 0.04,
                                "center": [0, 0, 0.03],
                                "rotation": [0, 0, 0],
                            }
                        ],
                    }
                )
            )
            original, mode = load_project(source)
        window.install(original, mode)
        window.profile_section.set_expanded(True)
        editor = window.point_editor
        editor.selector.set_value(4)
        editor.prompt.set_text("Увеличь радиус только этой вершины на 5 мм. Высоту сохрани.")
        editor.prompt.emit("activate")
        until(lambda: not window.busy, 600)
        edited = window.doc.scene
        assert edited.parts[0].shape == "lathe", window.status.get_text()
        assert window.view.drawn and not window.view.problem
        project = folder / "flared-bushing.xtri"
        save_project(project, edited, mode)
        assert load_project(project) == (edited, mode)
        for i, (profile, segments) in enumerate((("game", 16), ("render", 48))):
            window.mode.set_selected(i)
            target = folder / ("flared-bushing-" + profile + ".obj")
            export_obj(target, window.doc.scene, window.doc.mode)
            lines = target.read_text().splitlines()
            verts = np.array(
                [list(map(float, line.split()[1:])) for line in lines if line.startswith("v ")]
            ).reshape(4, segments, 3)
            for ring, radius, height in zip(
                verts, (0.020, 0.040, 0.040, 0.025), (0, 0, 0.060, 0.060)
            ):
                np.testing.assert_allclose(
                    np.linalg.norm(ring[:, :2], axis=1), radius, rtol=0, atol=1e-10
                )
                np.testing.assert_allclose(ring[:, 2], height, rtol=0, atol=1e-10)
            faces = [line for line in lines if line.startswith("f ")]
            assert {len(f.split()) - 1 for f in faces} == ({3} if profile == "game" else {4})
            print(
                f"{profile}: {len(faces)} faces; bore 40 mm at base, 50 mm at top", flush=True
            )
        while window.doc.history and window.doc.scene != original:
            window.undo()
        assert window.doc.scene == original
    finally:
        window.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(Path(sys.argv[1] if len(sys.argv) > 1 else tmp))
