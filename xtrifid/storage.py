import json
import os
import tempfile
from pathlib import Path

from .geometry import build, check_scene_budget
from .model import DETAILS, MODES, Scene


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
    payload = {"version": 2, "mode": mode, **scene.data()}
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
        or data["version"] not in (1, 2)
        or data.get("mode") not in MODES
    ):
        raise ValueError("Неизвестная версия или профиль проекта")
    if data["version"] == 1:
        data["detail"] = "medium"
    elif data.get("detail") not in DETAILS:
        raise ValueError("Неизвестный уровень детализации проекта")
    return Scene.parse(data), data["mode"]


def export_obj(path, scene, mode):
    check_scene_budget(scene, mode)

    def write(file):
        file.write("# Xtrifid · Z up · metres\n")
        offset = 1
        for index, part in enumerate(scene.parts, 1):
            mesh = build(part, mode, scene.detail)
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
