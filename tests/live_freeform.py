"""Exercise the local model on unrelated subjects and measure the geometry it generated."""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from xtrifid.geometry import build, check_scene_budget, triangles
from xtrifid.planner import Planner
from xtrifid.storage import load_project, save_project

MOUSE = (
    "Сделай только цельный обтекаемый корпус компьютерной мыши. Ширина 7 см по X, длина 12 см по Y, "
    "высота 4 см по Z. Дно плоское. Верх плавно закруглён, его высшая точка смещена назад (+Y) "
    "от середины корпуса. Спереди корпус ниже. Без кнопок, швов и колеса."
)
MIRROR = (
    "Создай настольное зеркало для человека из трёх деталей. Овальное плоское основание "
    "шириной 14 см, глубиной 9 см и высотой 1 см лежит горизонтально в плоскости XY. "
    "По центру основания поставь вертикальную круглую стойку вдоль оси Z, высотой 15 см "
    "и диаметром 1 см. На стойку поставь круглое "
    "зеркальное стекло диаметром 20 см и толщиной 5 мм: оно стоит вертикально в плоскости XZ "
    "и смотрит вперёд. Габариты стекла в собранном виде: ширина по X 20 см, глубина по Y "
    "5 мм, высота по Z 20 см. Без рамы и дополнительных деталей."
)
MUSHROOM = "Сделай гриб с широкой шляпкой на толстой ножке"


def verify_mouse(scene):
    assert len(scene.parts) == 1
    part = scene.parts[0]
    assert part.shape == "loft", part
    vertices = build(part, "render", scene.detail).vertices
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [0.07, 0.12, 0.04], atol=1e-7)
    assert abs(vertices[:, 2].min()) < 1e-7
    sole = vertices[vertices[:, 2] < 1e-7]
    assert np.ptp(sole[:, 1]) > 0.05, "ground contact too narrow for the requested flat base"
    crown = vertices[vertices[:, 2] > 0.0399999]
    assert np.ptp(crown[:, 1]) < 0.005, "expected a rounded crest, found a wide flat top"
    assert crown[:, 1].mean() > (vertices[:, 1].min() + vertices[:, 1].max()) / 2 + 0.001, part


def verify_mirror(scene):
    assert len(scene.parts) == 3
    meshes = sorted(
        (build(part, "game", scene.detail).vertices for part in scene.parts),
        key=lambda vertices: vertices[:, 2].min(),
    )
    floor = 0
    for mesh, dimensions in zip(
        meshes, ([0.14, 0.09, 0.01], [0.01, 0.01, 0.15], [0.2, 0.005, 0.2])
    ):
        np.testing.assert_allclose(np.ptp(mesh, axis=0), dimensions, atol=0.00001)
        assert abs(mesh[:, 2].min() - floor) < 0.00001
        floor = mesh[:, 2].max()
    glass = meshes[-1]
    centre = (glass.max(axis=0) + glass.min(axis=0)) / 2
    radii = np.linalg.norm((glass - centre)[:, [0, 2]], axis=1)
    np.testing.assert_allclose(radii[radii > 1e-7], 0.1, atol=1e-7)


def verify_mushroom(scene):
    meshes = [build(p, "render", scene.detail) for p in scene.parts]
    boxes = sorted(
        ((m.vertices.min(axis=0), m.vertices.max(axis=0)) for m in meshes),
        key=lambda bounds: bounds[0][2],
    )
    lo, hi = boxes[0]
    if len(boxes) > 1:
        cap_box = max(boxes, key=lambda pair: np.prod((pair[1] - pair[0])[:2]))
        assert cap_box[0][2] > lo[2] + 0.01 * max(high[2] - lo[2] for _, high in boxes), (
            "cap also placed directly on floor"
        )
    for next_lo, next_hi in boxes[1:]:
        assert np.all(next_lo <= hi + 1e-7) and np.all(lo <= next_hi + 1e-7), (
            "detached mushroom geometry"
        )
        lo, hi = np.minimum(lo, next_lo), np.maximum(hi, next_hi)
    faces = np.concatenate([m.vertices[list(triangles(m.faces))] for m in meshes])
    a = faces.reshape(-1, 3)
    b = np.roll(faces, -1, axis=1).reshape(-1, 3)

    def section_width(fraction):
        z = lo[2] + fraction * (hi[2] - lo[2])
        cut = (a[:, 2] > z) != (b[:, 2] > z)
        start, end = a[cut], b[cut]
        assert len(start), "gap between stem and cap"
        t = (z - start[:, 2]) / (end[:, 2] - start[:, 2])
        section = start[:, :2] + t[:, None] * (end[:, :2] - start[:, :2])
        return np.ptp(section, axis=0)

    stem = section_width(0.20)
    cap = np.max([section_width(h) for h in np.linspace(0.60, 0.90, 7)], axis=0)
    for h in np.linspace(0.1, 0.95, 18):
        section_width(h)
    assert np.all(cap > 1.25 * stem) and np.all(stem > 0.12 * cap), (stem, cap)
    print(f"Stem widths {stem} m; cap widths {cap} m; continuous from the ground", flush=True)


def main(folder, selected=None):
    for name, prompt, detail, mode, verify in (
        ("mouse-shell", MOUSE, "high", "render", verify_mouse),
        ("table-mirror", MIRROR, "low", "game", verify_mirror),
        ("mushroom", MUSHROOM, "medium", "render", verify_mushroom),
    ):
        if selected and name != selected:
            continue
        print(f"Generating {name} at {detail} detail", flush=True)
        scene = Planner().generate(prompt, detail)
        print(
            [(part.name, part.shape, part.size, part.rotation) for part in scene.parts],
            flush=True,
        )
        check_scene_budget(scene, mode)
        try:
            verify(scene)
        except AssertionError:
            print(scene.data(), flush=True)
            raise
        path = folder / f"{name}.xtri"
        save_project(path, scene, mode)
        assert load_project(path) == (scene, mode)
        print(f"Verified {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?")
    parser.add_argument(
        "selected", choices=("mouse-shell", "table-mirror", "mushroom"), nargs="?"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        main(args.folder or Path(tmp), args.selected)
