"""Exercise cloak generation, isolation of a local edit, and both exports in real GTK."""

import argparse
import tempfile
from pathlib import Path
from time import perf_counter

from test_studio import until

from xtrifid.gtk import Gio, Gtk
from xtrifid.storage import export_obj, load_project, save_project
from xtrifid.ui import Studio


def exercise(folder, constrained):
    app = Gtk.Application(
        application_id="io.github.glitchykid.Xtrifid.CloakAcceptance",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )
    app.register(None)
    w = Studio(app)
    w.present()
    try:
        w.prompt.set_text("сделай человека с плащом")
        assert not w.generate.get_sensitive()
        w.detail.set_selected(2)
        started = perf_counter()
        w.prompt.emit("activate")
        until(lambda: not w.busy)
        assert w.doc.scene, w.status.get_text()
        until(lambda: w.view.drawn or w.view.problem)
        assert not w.view.problem and not w.view.get_error()
        original = w.doc.scene
        print(
            f"GTK generated and displayed {len(original.parts)} parts in {perf_counter() - started:.2f} s",
            flush=True,
        )
        cloak = next(i for i, p in enumerate(original.parts) if p.shape == "cloak")
        w.part_list.set_selected(cloak)
        w.fields["size"][2].set_value(0.8)
        w.apply_numeric()
        assert w.doc.scene.parts[cloak].size[2] == 0.8
        assert all(
            w.doc.scene.parts[i] == p for i, p in enumerate(original.parts) if i != cloak
        )
        w.undo()
        assert w.doc.scene == original
        if constrained:
            w.detail.set_selected(1)
            w.mode.set_selected(1)
            assert w.doc.mode == "render" and w.doc.scene.detail == "low"
            previous = w.doc.scene
            w.detail.set_selected(3)
            assert w.doc.scene is previous and w.detail.get_selected() == 1
            assert "лимит" in w.status.get_text(), w.status.get_text()
            print(w.status.get_text(), flush=True)
            export_obj(folder / "constrained.obj", previous, "render")
            return
        for mode_index, mode in enumerate(("game", "render")):
            w.mode.set_selected(mode_index)
            samples = []
            for selector in (1, 2, 3):
                w.detail.set_selected(selector)
                assert w.selected == cloak
                samples.append(len(w.view.owners))
            assert 0 < samples[0] < samples[1] < samples[2] <= 150_000
            w.detail.set_selected(2)
            obj = folder / f"cloaked-person-{mode}.obj"
            export_obj(obj, w.doc.scene, mode)
            records = obj.read_text().splitlines()
            faces = [len(line.split()) - 1 for line in records if line.startswith("f ")]
            assert set(faces) == ({3} if mode == "game" else {3, 4})
            assert sum(n - 2 for n in faces) == samples[1]
            assert sum(s.startswith("o ") for s in records) == len(original.parts)
            print(mode, samples, len(faces), "medium-detail faces", flush=True)
        project = folder / "cloaked-person.xtri"
        save_project(project, original, "render")
        assert load_project(project) == (original, "render")
        print(f"OK: {len(original.parts)} editable parts; independent cloak edit and undo.")
    finally:
        w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path)
    parser.add_argument("--constrained", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        exercise(args.folder or Path(tmp), args.constrained)
