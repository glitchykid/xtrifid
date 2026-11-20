"""Choose image views and inspect the exact silhouettes used for reconstruction."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .gtk import Gdk, Gio, GLib, Gtk, Pango
from .planner import MAX_PROMPT
from .reference import read_image, silhouette


@dataclass(frozen=True)
class ReferenceRequest:
    front: Path
    side: Path | None
    photo: bool
    height: float
    depth: float | None
    tolerance: int
    prompt: str
    rounded: bool = False


class ReferenceDialog(Gtk.Window):
    def __init__(self, parent, on_generate, folder=None, prompt="", endpoint=""):
        super().__init__(
            title="Создать по референсам", transient_for=parent, modal=True, default_width=465
        )
        self.set_destroy_with_parent(True)
        self.on_generate = on_generate
        self.folder = folder
        self.paths = [None, None]
        self.buffers = [None, None]
        self.requests = [0, 0]
        self.loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reference")
        self.pending = [None, None]
        self.dead = False
        self.endpoint = urlsplit(endpoint).hostname if endpoint else None
        self.connect("close-request", self._closing)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        for edge in ("start", "end", "top", "bottom"):
            getattr(box, "set_margin_" + edge)(12)
        self.set_child(box)
        self.method = Gtk.DropDown.new_from_strings(["Силуэты", "Фото с ИИ"])
        self.method.update_property([Gtk.AccessibleProperty.LABEL], ["Способ реконструкции"])
        box.append(self.method)
        self.instructions = Gtk.Label(xalign=0, wrap=True)
        box.append(self.instructions)
        views = Gtk.Box(spacing=8, homogeneous=True)
        box.append(views)
        self.pictures = []
        self.names = []
        for i, title in enumerate(("Вид спереди", "Вид справа")):
            column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            views.append(column)
            picture = Gtk.Picture(can_shrink=True)
            picture.set_size_request(120, 95)
            picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            self.pictures.append(picture)
            column.append(picture)
            name = Gtk.Label(
                label="Обязательно" if i == 0 else "По желанию",
                ellipsize=Pango.EllipsizeMode.MIDDLE,
            )
            self.names.append(name)
            column.append(name)
            button = Gtk.Button(label=title)
            button.connect("clicked", lambda _, n=i: self.choose(n))
            column.append(button)
        self.clear_side = Gtk.Button(label="Убрать второй вид")
        self.clear_side.connect("clicked", lambda *_: self.set_path(1, None))
        box.append(self.clear_side)
        grid = Gtk.Grid(column_spacing=10, row_spacing=4)
        box.append(grid)

        def row(i, text, control):
            label = Gtk.Label(label=text, xalign=0, hexpand=True)
            grid.attach(label, 0, i, 1, 1)
            grid.attach(control, 1, i, 1, 1)
            control.update_property([Gtk.AccessibleProperty.LABEL], [text])
            return label

        self.height = Gtk.SpinButton.new_with_range(0.001, 1000, 0.01)
        self.height.set_digits(3)
        self.height.set_value(1.0)
        row(0, "Высота модели, м", self.height)
        self.depth_check = Gtk.CheckButton(label="Задать глубину, м")
        self.depth = Gtk.SpinButton.new_with_range(0.001, 1000, 0.01)
        self.depth.set_digits(3)
        self.depth.set_value(0.25)
        self.depth.update_property([Gtk.AccessibleProperty.LABEL], ["Глубина модели, м"])
        grid.attach(self.depth_check, 0, 1, 1, 1)
        grid.attach(self.depth, 1, 1, 1, 1)
        self.tolerance = Gtk.SpinButton.new_with_range(4, 128, 2)
        self.tolerance.set_value(42)
        self.tolerance_label = row(2, "Допуск цвета фона", self.tolerance)
        self.rounded = Gtk.CheckButton(label="Округлить сечения")
        self.rounded.set_tooltip_text("Приблизить глубину округлых частей эллипсами")
        box.append(self.rounded)
        self.hint = Gtk.Entry(
            placeholder_text="Какой объект построить с фото", max_length=MAX_PROMPT
        )
        self.hint.set_text(prompt)
        self.hint.update_property([Gtk.AccessibleProperty.LABEL], ["Уточнение к фото"])
        box.append(self.hint)
        self.status = Gtk.Label(xalign=0, wrap=True)
        box.append(self.status)
        buttons = Gtk.Box(spacing=6, halign=Gtk.Align.END)
        close = Gtk.Button(label="Отмена")
        close.connect("clicked", lambda *_: self.close())
        self.generate = Gtk.Button(label="Построить модель")
        self.generate.add_css_class("suggested-action")
        self.generate.connect("clicked", self.submit)
        buttons.append(close)
        buttons.append(self.generate)
        box.append(buttons)
        self.method.connect("notify::selected", lambda *_: self.refresh())
        self.depth_check.connect(
            "toggled", lambda *_: self.depth.set_sensitive(self.depth_check.get_active())
        )
        self.tolerance.connect("value-changed", lambda *_: self.refresh())
        self.refresh()

    def _closing(self, *_):
        self.dead = True
        self.loader.shutdown(wait=False, cancel_futures=True)
        return False

    def choose(self, index):
        dialog = Gtk.FileDialog(title="Выберите изображение PNG или JPEG")
        if self.folder:
            dialog.set_initial_folder(Gio.File.new_for_path(self.folder))
        formats = Gtk.FileFilter()
        formats.set_name("PNG, JPEG")
        formats.add_mime_type("image/png")
        formats.add_mime_type("image/jpeg")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(formats)
        dialog.set_filters(filters)
        dialog.set_default_filter(formats)

        def finish(sender, result):
            try:
                file = sender.open_finish(result)
                path = file.get_path()
                if path and not self.dead:
                    self.set_path(index, path)
            except GLib.Error as exc:
                if not self.dead and not exc.matches(
                    Gtk.dialog_error_quark(), Gtk.DialogError.DISMISSED
                ):
                    self.status.set_text(exc.message)

        dialog.open(self, None, finish)

    def set_path(self, index, path):
        if self.dead:
            return
        if self.pending[index] is not None:
            self.pending[index].cancel()
        self.requests[index] += 1
        request = self.requests[index]
        self.paths[index] = Path(path) if path is not None else None
        self.buffers[index] = None
        self.refresh()
        if path is None:
            return
        self.folder = os.fspath(Path(path).parent)

        def ready(buf, error):
            if not self.dead and request == self.requests[index]:
                self.buffers[index] = buf
                self.refresh()
                if error:
                    self.status.set_text(error)
            return False

        def load():
            try:
                buf = read_image(path)
            except (ValueError, OSError, GLib.Error, MemoryError) as exc:
                GLib.idle_add(ready, None, str(exc) or "Недостаточно памяти для фото")
            else:
                GLib.idle_add(ready, buf, None)

        self.pending[index] = self.loader.submit(load)

    def refresh(self):
        photo = self.method.get_selected() == 1
        server = self.endpoint or "локальный сервер Ollama"
        self.instructions.set_text(
            f"ИИ выделит главный объект и оценит его форму. Фото будут переданы в {server}."
            if photo
            else "Выберите PNG с прозрачностью или предмет на ровном фоне. Белым показан силуэт. Для двух видов используйте одну высоту и снимайте без наклона."
        )
        self.clear_side.set_visible(self.paths[1] is not None)
        self.hint.set_visible(photo)
        for control in (
            self.depth_check,
            self.depth,
            self.tolerance,
            self.tolerance_label,
            self.rounded,
        ):
            control.set_visible(not photo)
        self.depth.set_sensitive(self.depth_check.get_active())
        valid = self.buffers[0] is not None
        error = ""
        for i, (buf, path) in enumerate(zip(self.buffers, self.paths)):
            pic = self.pictures[i]
            self.names[i].set_label(
                path.name if path else ("Обязательно" if i == 0 else "По желанию")
            )
            paint = None
            if buf is None:
                if path:
                    valid = False
            elif photo:
                paint = Gdk.Texture.new_for_pixbuf(buf)
            else:
                try:
                    rows, _ = silhouette(buf, self.tolerance.get_value_as_int())
                    data = np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8).reshape(
                        len(rows), len(rows[0])
                    )
                    rgb = np.repeat(
                        np.where(data == 49, 235, 27).astype("uint8")[:, :, None], 3, axis=2
                    )
                    paint = Gdk.MemoryTexture.new(
                        rgb.shape[1],
                        rgb.shape[0],
                        Gdk.MemoryFormat.R8G8B8,
                        GLib.Bytes.new(rgb.tobytes()),
                        rgb.shape[1] * 3,
                    )
                except ValueError as exc:
                    error = str(exc)
                    valid = False
            pic.set_paintable(paint)
        self.status.set_label(error)
        self.generate.set_sensitive(valid)

    def submit(self, *_):
        if not self.generate.get_sensitive():
            return
        self.height.update()
        self.depth.update()
        request = ReferenceRequest(
            self.paths[0],
            self.paths[1],
            bool(self.method.get_selected()),
            self.height.get_value(),
            self.depth.get_value() if self.depth_check.get_active() else None,
            self.tolerance.get_value_as_int(),
            self.hint.get_text().strip(),
            self.rounded.get_active(),
        )
        self.close()
        self.on_generate(request)
