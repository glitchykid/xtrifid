"""Check natural-language characters, a dimensioned object and attachments with real Ollama."""

import numpy as np

from xtrifid.character import build_character
from xtrifid.geometry import build
from xtrifid.planner import Planner


def check_accessory(client):
    scene = client.generate(
        "Человек ростом 160 см, обычного телосложения, без плаща и капюшона. "
        "Руки горизонтально разведены в стороны. На левой ладони одна кубическая коробка, "
        "длина каждого ребра 10 см. Больше ничего нет.",
        "low",
    )
    person = build_character({"height_m": 1.6, "pose": "arms_out"}, "low")
    assert scene.parts[:-1] == person.parts, "Expected just the posed figure and one extra box"
    box = scene.parts[-1]
    assert box.shape == "box"
    np.testing.assert_allclose(box.size, [0.1, 0.1, 0.1])
    left = next(p for p in person.parts if p.name == "Кисть левая")
    assert np.linalg.norm(np.subtract(box.center, left.center)) < 0.15, (
        box.center,
        left.center,
    )
    print("AI accessory: 10 cm box at the left hand of a 1.6 m figure in a T-pose", flush=True)


def check_character(client):
    figure = client.generate(
        "Создай широкоплечего путника ростом 192 см в шаге. На нём длинный плащ "
        "до щиколоток с надетым капюшоном. В руках ничего нет.",
        "medium",
    )
    expected = build_character(
        {"height_m": 1.92, "build": "broad", "pose": "walking", "cloak": "ankle", "hood": True},
        "medium",
    )
    assert figure == expected, (
        "The figure differs from the requested height, build, pose or clothing"
    )
    print(
        f"AI character: {len(figure.parts)} parts, 1.92 m, walking, ankle cloak, raised hood",
        flush=True,
    )
    return figure


def check_prop(client):
    prop = client.generate(
        "Сделай для человека в плаще прямоугольную подставку из одного бруска. Размер по X 40 см, "
        "по Y 20 см, по Z 10 см. Без поворота, нижняя грань лежит на полу.",
        "low",
    )
    assert len(prop.parts) == 1 and prop.detail == "low"
    mesh = build(prop.parts[0], "game", prop.detail)
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), [0.4, 0.2, 0.1])
    assert np.isclose(mesh.vertices[:, 2].min(), 0)
    assert len(mesh.vertices) == 8 and len(mesh.faces) == 12
    print("AI prop: 0.4 × 0.2 × 0.1 m, eight vertices and twelve triangles", flush=True)


if __name__ == "__main__":
    client = Planner()
    check_character(client)
    check_prop(client)
    check_accessory(client)
