import json
import os
import tempfile
from pathlib import Path

from .geometry import build
from .model import MODES, Scene


def atomic_write(path, write):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        try:
            write(file)
            file.flush()
            os.fsync(file.fileno())
            file.close()
            os.replace(file.name, path)
        finally:
            Path(file.name).unlink(missing_ok=True)


def save_project(path, scene, mode):
    payload = {"version": 1, "mode": mode, **scene.data()}
    atomic_write(path, lambda file: json.dump(payload, file, ensure_ascii=False, indent=2))


def load_project(path):
    with open(path, "rb") as file:
        data = file.read(524289)
    if len(data) > 524288:
        raise ValueError("Слишком большой файл проекта")
    data = json.loads(data)
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
        or data.get("mode") not in MODES
    ):
        raise ValueError("Неизвестная версия или профиль проекта")
    return Scene.parse(data), data["mode"]


def export_obj(path, scene, mode):
    def write(file):
        file.write("# Xtrifid · Z up · metres\n")
        offset = 1
        for index, part in enumerate(scene.parts, 1):
            mesh = build(part, mode)
            file.write(f"o {index:02}_{part.name.replace(' ', '_')}\n")
            for x, y, z in mesh.vertices:
                file.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
            last = None
            for face, smooth in zip(mesh.faces, mesh.smooth):
                if last != smooth:
                    file.write("s 1\n" if smooth else "s off\n")
                    last = smooth
                file.write("f " + " ".join(str(v + offset) for v in face) + "\n")
            offset += len(mesh.vertices)

    atomic_write(path, write)
