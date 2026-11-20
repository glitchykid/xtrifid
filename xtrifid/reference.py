"""Turn calibrated silhouettes or a visual description of a photo into editable geometry."""

import base64
import os
import stat
from dataclasses import replace
from pathlib import Path

import gi
import numpy as np

from .geometry import build, check_scene_budget
from .model import DETAILS, MAX_LENGTH, MIN_SIZE, Part, Scene, vector
from .planner import MAX_PROMPT
from .resources import MIB, capacity

gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib  # noqa: E402

MAX_BYTES = 16 * MIB
MAX_PIXELS = 24_000_000
SAMPLE = 448
MASK_SIZE = 96


def _dimensions(raw):
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and raw[12:16] == b"IHDR" and len(raw) >= 24:
        return "png", int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw.startswith(b"\xff\xd8"):
        # Check the SOF before handing compressed data to the image decoder.
        pos = 2
        while pos + 4 <= len(raw) and raw[pos] == 255:
            pos += 1
            while pos < len(raw) and raw[pos] == 255:
                pos += 1
            if pos + 3 > len(raw):
                break
            marker = raw[pos]
            length = int.from_bytes(raw[pos + 1 : pos + 3], "big")
            if length < 2 or pos + 1 + length > len(raw) or marker in (0xD9, 0xDA):
                break
            if marker in (0xC0, 0xC1, 0xC2) and length >= 8:
                height = int.from_bytes(raw[pos + 4 : pos + 6], "big")
                width = int.from_bytes(raw[pos + 6 : pos + 8], "big")
                return "jpeg", width, height
            pos += length + 1
    raise ValueError("Ожидается изображение PNG или JPEG")


def read_image(path):
    try:
        with os.fdopen(os.open(os.fspath(path), os.O_RDONLY | os.O_NONBLOCK), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
                raise ValueError("Выберите файл PNG или JPEG размером до 16 МиБ")
            if capacity().headroom < 32 * MIB + 2 * info.st_size:
                raise ValueError("Недостаточно доступной памяти для открытия референса")
            raw = stream.read(info.st_size + 1)
    except OSError as exc:
        raise ValueError("Не удалось прочитать файл референса") from exc
    if len(raw) != info.st_size:
        raise ValueError("Файл референса изменился во время чтения")
    kind, w, h = _dimensions(raw)
    if min(w, h) < 2 or max(w, h) > 16384 or w * h > MAX_PIXELS:
        raise ValueError(
            "Референс должен содержать до 24 мегапикселей, со стороной до 16 384 пикселей"
        )
    if capacity().headroom < 32 * MIB + len(raw) + w * h * 8:
        raise ValueError("Недостаточно доступной памяти для открытия референса")
    ratio = min(1.0, SAMPLE / max(w, h))
    try:
        loader = GdkPixbuf.PixbufLoader.new_with_type(kind)
        loader.connect(
            "size-prepared",
            lambda source, *_: source.set_size(
                max(1, round(w * ratio)), max(1, round(h * ratio))
            ),
        )
        loader.write(raw)
        loader.close()
        buf = loader.get_pixbuf()
        if buf is not None:
            buf = buf.apply_embedded_orientation() or buf
    except GLib.Error as exc:
        raise ValueError("Не удалось декодировать изображение PNG или JPEG") from exc
    if buf is None or max(buf.get_width(), buf.get_height()) > SAMPLE:
        raise ValueError("Не удалось получить уменьшенную копию референса")
    return buf


def pixels(buf):
    # GdkPixbuf can omit right padding on the final row.
    return np.ndarray(
        (buf.get_height(), buf.get_width(), buf.get_n_channels()),
        dtype=np.uint8,
        buffer=buf.get_pixels(),
        strides=(buf.get_rowstride(), buf.get_n_channels(), 1),
    )


def silhouette(buf, tolerance=42):
    if type(tolerance) is not int or not 4 <= tolerance <= 128:
        raise ValueError("Допуск фона должен быть от 4 до 128")
    rgb = pixels(buf)
    if rgb.shape[2] == 4 and np.any(rgb[:, :, 3] < 32):
        mask = rgb[:, :, 3] >= 96
    else:
        border = np.concatenate(
            [rgb[0, :, :3], rgb[-1, :, :3], rgb[1:-1, 0, :3], rgb[1:-1, -1, :3]]
        )
        background = np.median(border, axis=0)
        delta = np.max(np.abs(rgb[:, :, :3].astype(np.int16) - background), axis=2)
        mask = delta > tolerance
        too_varied = np.mean(np.max(np.abs(border - background), axis=1) > tolerance)
        if too_varied > 0.25:
            raise ValueError(
                "Сложный фон. Выберите режим «Фото с ИИ» или подготовьте прозрачный PNG."
            )
    rows, cols = np.nonzero(mask)
    if len(rows) < 12 or min(np.ptp(rows), np.ptp(cols)) < 2:
        raise ValueError("Не удалось выделить силуэт. Проверьте фон, прозрачность и допуск.")
    mask = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    h, w = mask.shape
    scale = min(1.0, MASK_SIZE / max(h, w))
    # Sample pixel centres to keep holes and disconnected silhouettes.
    nh, nw = max(3, round(h * scale)), max(3, round(w * scale))
    y = np.minimum(((np.arange(nh) + 0.5) * h / nh).astype(int), h - 1)
    x = np.minimum(((np.arange(nw) + 0.5) * w / nw).astype(int), w - 1)
    sampled = mask[np.ix_(y, x)]
    return tuple(bytes(row + 48).decode("ascii") for row in sampled.astype("uint8")), w / h


def _height(height_m):
    return vector([height_m], 1, MIN_SIZE, MAX_LENGTH)[0]


def reference_scene(
    front,
    side=None,
    *,
    height_m=1.0,
    depth_m=None,
    detail="medium",
    tolerance=42,
    rounded=False,
):
    height = _height(height_m)
    if detail not in DETAILS:
        raise ValueError("Выберите детализацию модели")
    front_mask, ratio = silhouette(read_image(front), tolerance)
    masks = [front_mask]
    width = height * ratio
    if side:
        side_mask, side_ratio = silhouette(read_image(side), tolerance)
        masks.append(side_mask)
        depth = height * side_ratio
    else:
        depth = width * 0.55
    if depth_m is not None:
        depth = _height(depth_m)
    name = "".join(c for c in Path(front).stem if c.isprintable()).strip() or "Силуэт"
    part = Part.parse(
        {
            "name": name[:80],
            "shape": "reference",
            "size": [width, depth, height],
            "center": [0.0, 0.0, height / 2],
            "rotation": [0, 0, 0],
            "masks": masks,
            "rounded": rounded,
        }
    )
    return Scene((part,), detail)


def _encoded_photo(front, side=None):
    views = [read_image(path) for path in (front, side) if path is not None]
    height = max(v.get_height() for v in views)
    gap = 8
    width = sum(v.get_width() for v in views) + (len(views) - 1) * gap
    canvas = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, width, height)
    canvas.fill(0xF0F0F0FF)
    x = 0
    for view in views:
        y = (height - view.get_height()) // 2
        view.composite(
            canvas,
            x,
            y,
            view.get_width(),
            view.get_height(),
            x,
            y,
            1,
            1,
            GdkPixbuf.InterpType.NEAREST,
            255,
        )
        x += view.get_width() + gap
    ok, data = canvas.save_to_bufferv("jpeg", ["quality"], ["88"])
    if not ok:
        raise ValueError("Не удалось подготовить фото")
    return base64.b64encode(data).decode("ascii")


def photo_scene(
    planner,
    front,
    side=None,
    *,
    height_m=1.0,
    detail="medium",
    prompt="",
    mode="game",
    style="standard",
):
    height = _height(height_m)
    if detail not in DETAILS or len(prompt) > MAX_PROMPT:
        raise ValueError("Проверьте детализацию и длину описания")
    photo = _encoded_photo(front, side)
    from .photo_plan import from_photo

    scene = replace(
        from_photo(planner, photo, side is not None, prompt, height, detail), style=style
    )
    check_scene_budget(scene, mode)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for part in scene.parts:
        vertices = build(part, mode, detail, style).vertices
        lo = np.minimum(lo, vertices.min(axis=0))
        hi = np.maximum(hi, vertices.max(axis=0))
    if hi[2] <= lo[2]:
        raise ValueError("На фото не удалось определить объём предмета")
    factor = height / (hi[2] - lo[2])
    origin = np.array([(hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, lo[2]])
    parts = []
    for part in scene.parts:
        size = vector((np.array(part.size) * factor).tolist(), 3, MIN_SIZE, MAX_LENGTH)
        center = vector(((part.center - origin) * factor).tolist(), 3, -MAX_LENGTH, MAX_LENGTH)
        bore = part.inner_diameter * factor if part.inner_diameter else None
        if bore is not None and not 0.000001 <= bore < size[0]:
            raise ValueError("Отверстие слишком мало при выбранной высоте модели")
        parts.append(replace(part, size=size, center=center, inner_diameter=bore))
    return replace(scene, parts=tuple(parts))
