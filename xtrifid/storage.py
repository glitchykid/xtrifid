import errno
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import numpy as np

from .geometry import build, check_scene_budget
from .model import DETAILS, MODES, Scene
from .resources import MIB, capacity
from .textures import metallic_roughness_pixels, normal_pixels, pixels, png_bytes, uv_layout


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


MAX_PROJECT_BYTES = 16 * 1024 * 1024


def save_project(path, scene, mode):
    def write(file):
        has_material_data = any(
            (p.texture and p.texture.pbr) or (p.surface and p.surface.uv is not None)
            for p in scene.parts
        )
        version = 5 if has_material_data else 4
        payload = {"version": version, "mode": mode, **scene.data()}
        json.dump(payload, file, ensure_ascii=False, indent=2)
        if file.tell() > MAX_PROJECT_BYTES:
            raise ValueError("Слишком большой файл проекта")

    atomic_write(path, write)


def load_project(path):
    with os.fdopen(os.open(os.fspath(path), os.O_RDONLY | os.O_NONBLOCK), "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Выберите обычный файл проекта")
        if info.st_size > MAX_PROJECT_BYTES:
            raise ValueError("Слишком большой файл проекта")
        if capacity().headroom < 4 * MIB + 24 * info.st_size:
            raise ValueError("Недостаточно доступной памяти для открытия проекта")
        data = file.read(info.st_size + 1)
        if len(data) != info.st_size:
            raise ValueError("Файл проекта изменился во время чтения")
    data = json.loads(data)
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] not in (1, 2, 3, 4, 5)
        or data.get("mode") not in MODES
    ):
        raise ValueError("Неизвестная версия или профиль проекта")
    if data["version"] == 1:
        data["detail"] = "medium"
    elif data.get("detail") not in DETAILS:
        raise ValueError("Неизвестный уровень детализации проекта")
    return Scene.parse(data), data["mode"]


def _durable_bytes(path, payload):
    with path.open("xb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _material_library(parent, parts, mode):
    specs = tuple(dict.fromkeys(p.texture for p in parts if p.texture is not None))
    if not specs:
        return None, {}
    names = {spec: f"surface_{i:02}" for i, spec in enumerate(specs, 1)}
    # Publish a complete immutable set before replacing the OBJ commit point.
    # An earlier, still readable OBJ retains its original resources across a
    # retry, concurrent export or failure. Identical exports share their files.
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        stage = Path(temporary) / "maps"
        stage.mkdir()
        records = ["newmtl untextured\nKd 0.80 0.80 0.80\nillum 1\n"]
        hashes = {}

        def write_map(image, prefix=""):
            payload = png_bytes(image)
            digest = hashlib.sha256(payload).hexdigest()
            filename = f"{prefix}{digest}.png"
            if filename not in hashes:
                _durable_bytes(stage / filename, payload)
                hashes[filename] = (digest, len(payload))
            return filename

        for spec, name in names.items():
            colour = write_map(pixels(spec))
            record = f"newmtl {name}\nKd 1 1 1\nillum 1\nmap_Kd {colour}\n"
            if mode == "game" and spec.pbr is not None:
                normal = write_map(normal_pixels(spec), "normal-gl-")
                packed = metallic_roughness_pixels(spec)
                roughness = write_map(packed[:, :, 1], "roughness-")
                metallic = write_map(packed[:, :, 2], "metallic-")
                del packed
                record += f"norm {normal}\nPr 1\nmap_Pr {roughness}\nPm 1\nmap_Pm {metallic}\n"
            records.append(record)
        library = "\n".join(records).encode("ascii")
        digest = hashlib.sha256(library).hexdigest()
        hashes["surface.mtl"] = (digest, len(library))
        _durable_bytes(stage / "surface.mtl", library)
        _sync_directory(stage)
        directory = parent / f"xtrifid-materials-{digest[:24]}"
        try:
            os.rename(stage, directory)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if directory.is_symlink() or not directory.is_dir():
                raise OSError("Папка текстур занята другим файлом") from exc
            for name, (expected, size) in hashes.items():
                entry = directory / name
                if entry.is_symlink() or not entry.is_file() or entry.stat().st_size != size:
                    raise OSError("Папка текстур содержит повреждённые данные") from exc
                with entry.open("rb") as file:
                    if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                        raise OSError("Контрольная сумма файла текстуры не совпадает") from exc
        _sync_directory(parent)
        return f"{directory.name}/surface.mtl", names


def export_obj(path, scene, mode):
    check_scene_budget(scene, mode)
    path = Path(path)
    library, materials = _material_library(path.parent, scene.parts, mode)

    def write(file):
        file.write("# Xtrifid · Z up · metres\n")
        if library:
            file.write(f"mtllib {library}\n")
        offset = uv_offset = normal_offset = 1
        for index, part in enumerate(scene.parts, 1):
            mesh = build(part, mode, scene.detail, scene.style)
            file.write(f"o {index:02}_{part.name.replace(' ', '_')}\n")
            if library:
                file.write(f"usemtl {materials.get(part.texture, 'untextured')}\n")
            for x, y, z in mesh.vertices:
                file.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
            uv_faces = None
            normal_faces = None
            if part.texture is not None:
                uv, uv_faces = uv_layout(mesh, part)
                for u, v in uv:
                    file.write(f"vt {u:.12g} {v:.12g}\n")
                if mode == "game" and part.texture.pbr is not None:
                    from .tangents import corner_normals

                    _, corner_ns = corner_normals(mesh)
                    normals, inverse = np.unique(
                        corner_ns.reshape(-1, 3), axis=0, return_inverse=True
                    )
                    normal_faces = inverse.reshape(-1, 3)
                    for nx, ny, nz in normals:
                        file.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
            last = None
            for i, (face, smooth) in enumerate(zip(mesh.faces, mesh.smooth)):
                if last != smooth:
                    file.write("s 1\n" if smooth else "s off\n")
                    last = smooth
                if normal_faces is not None:
                    corners = (
                        f"{v + offset}/{t + uv_offset}/{n + normal_offset}"
                        for v, t, n in zip(face, uv_faces[i], normal_faces[i])
                    )
                elif uv_faces is None:
                    corners = (str(v + offset) for v in face)
                else:
                    corners = (
                        f"{v + offset}/{t + uv_offset}" for v, t in zip(face, uv_faces[i])
                    )
                file.write("f " + " ".join(corners) + "\n")
            offset += len(mesh.vertices)
            if uv_faces is not None:
                uv_offset += len(uv)
            if normal_faces is not None:
                normal_offset += len(normals)

    atomic_write(path, write)
    if library:
        _sync_directory(path.parent)
