"""Optional acceptance run against a real Ollama model on the Compose network."""

import json
import sys
import tempfile
from pathlib import Path

from test_studio import until

from xtrifid.gtk import Gio, Gtk
from xtrifid.storage import export_obj, load_project, save_project
from xtrifid.ui import Studio


def main():
    app = Gtk.Application(
        application_id="io.github.glitchykid.Xtrifid.Acceptance",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )
    app.register(None)
    window = Studio(app)
    window.present()
    window.prompt.set_text(
        "Сделай табурет из пяти отдельных деталей: квадратное сиденье 40 на 40 см толщиной 4 см и четыре прямые ножки высотой 40 см по углам. Пол находится на Z=0."
    )
    window.generate.emit("clicked")
    until(lambda: not window.busy, 600)
    initial = window.doc.scene
    assert initial, window.status.get_text()
    print(json.dumps(initial.data(), ensure_ascii=False), flush=True)
    assert len(initial.parts) == 5, "Expected a seat and four legs"
    seat = max(range(5), key=lambda i: initial.parts[i].size[0] * initial.parts[i].size[1])
    assert all(abs(a - b) < 0.0001 for a, b in zip(initial.parts[seat].size, (0.4, 0.4, 0.04)))
    assert len({p.center[:2] for i, p in enumerate(initial.parts) if i != seat}) == 4
    legs = [p for i, p in enumerate(initial.parts) if i != seat]
    seat_bottom = initial.parts[seat].center[2] - initial.parts[seat].size[2] / 2
    for leg in legs:
        assert abs(leg.center[2] - leg.size[2] / 2) < 1e-6, "A foot must touch the floor"
        assert abs(leg.center[2] + leg.size[2] / 2 - seat_bottom) < 1e-6, (
            "Seat must rest on all legs"
        )
    window.select(seat)
    window.adjust.set_text(
        "Измени только толщину сиденья по Z на 8 см. Сохрани центр и остальные размеры."
    )
    window.adjust.emit("activate")
    until(lambda: not window.busy, 600)
    final = window.doc.scene
    assert all(p == final.parts[i] for i, p in enumerate(initial.parts) if i != seat)
    assert final.parts[seat].size == (0.4, 0.4, 0.08), window.status.get_text()
    assert final.parts[seat].center == initial.parts[seat].center
    assert window.view.drawn and not window.view.problem
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(sys.argv[1] if len(sys.argv) > 1 else temp)
        for i, mode in enumerate(("game", "render")):
            window.mode.set_selected(i)
            path = folder / f"stool-{mode}.obj"
            export_obj(path, final, mode)
            faces = [line for line in path.read_text().splitlines() if line.startswith("f ")]
            assert {len(line.split()) - 1 for line in faces} == ({3} if mode == "game" else {4})
            print(mode, len(faces), "faces", flush=True)
        project = folder / "stool.xtri"
        save_project(project, final, "render")
        assert load_project(project) == (final, "render")
        print(
            "OK: selected seat thickness changed from 4 to 8 cm; four legs preserved.",
            flush=True,
        )
    window.close()


if __name__ == "__main__":
    main()
