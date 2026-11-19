import os
import threading
from pathlib import Path

from xtrifid.gtk import Gio, GLib, Gtk, Pango

from .geometry import build
from .local_server import pull_model
from .model import MAX_LENGTH, MIN_SIZE, MODES, Document, Part
from .planner import MAX_PROMPT, ModelMissing, Planner, PlannerError
from .profile import PROFILE_SHAPES, profile_guide
from .profile_editor import ProfileEditor
from .storage import export_obj, load_project, save_project
from .viewport import Viewport


def label(text, css=None):
    widget = Gtk.Label(label=text, xalign=0)
    if css:
        widget.add_css_class(css)
    return widget


def button(title, callback, css=None):
    widget = Gtk.Button(label=title)
    widget.connect("clicked", lambda *_: callback())
    if css:
        widget.add_css_class(css)
    return widget


def accessible(widget, name):
    widget.update_property([Gtk.AccessibleProperty.LABEL], [name])


STYLE = """
window {background:#171b22;color:#e7eaf0;}
headerbar {background:#20252f;color:#e7eaf0;}
.brand {font-weight:800;letter-spacing:1.1px;color:#a7e6d8;}
.side {background:#20252f;padding:16px;}
.side entry,.side dropdown button,.side spinbutton {background:#161b23;color:#eef1f7;}
.section {font-weight:bold;font-size:14px;}
.muted {color:#abb5c5;font-size:12px;}
.primary {background:#8ad9c7;color:#102720;font-weight:bold;}
.compact spinbutton button {padding:0;min-width:15px;min-height:25px;}
.compact spinbutton text {padding-left:3px;padding-right:2px;min-width:40px;}
.toolbar {background:#20252f;padding:6px 10px;}
.status {background:#191e27;padding:7px 12px;}
.empty {background:#1e242e;color:#d8e0ec;border-radius:10px;padding:12px 16px;}
"""


class Studio(Gtk.ApplicationWindow):
    def __init__(self, app, local_server=None):
        super().__init__(
            application=app, title="Xtrifid", default_width=1140, default_height=740
        )
        Gtk.Settings.get_default().set_property("gtk-application-prefer-dark-theme", True)
        css = Gtk.CssProvider()
        css.load_from_string(STYLE)
        Gtk.StyleContext.add_provider_for_display(
            self.get_display(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self.doc = Document()
        self.planner = Planner(local_server=local_server)
        self.missing_model = None
        self.cancel_download = threading.Event()
        self.file_folder = None
        self.selected = 0
        self.busy = False
        self.closed = False
        self.syncing = False
        self.connect("close-request", self.closing)
        header = Gtk.HeaderBar()
        header.set_title_widget(label("XTRIFID", "brand"))
        self.opener = button("Открыть", lambda: self.file_dialog("open"))
        self.saver = button("Сохранить", lambda: self.file_dialog("save"))
        self.exporter = button("Экспорт OBJ", lambda: self.file_dialog("obj"))
        for widget in (self.opener, self.saver):
            header.pack_start(widget)
        header.pack_end(self.exporter)
        self.set_titlebar(header)
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(root)
        layout = Gtk.Paned(
            orientation=Gtk.Orientation.HORIZONTAL, wide_handle=True, vexpand=True
        )
        root.append(layout)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        layout.set_start_child(scroll)
        layout.set_resize_start_child(False)
        layout.set_shrink_start_child(False)
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=9)
        side.add_css_class("side")
        scroll.set_child(side)
        side.append(label("Описание модели", "section"))
        self.prompt = Gtk.Entry(placeholder_text="Например, табурет с четырьмя ножками")
        self.prompt.set_max_length(MAX_PROMPT)
        accessible(self.prompt, "Описание новой модели")
        self.prompt.connect("activate", lambda *_: self.request(False))
        side.append(self.prompt)
        self.mode = Gtk.DropDown.new_from_strings(["Game Ready", "Для рендера"])
        accessible(self.mode, "Профиль геометрии")
        self.mode.connect("notify::selected", self.change_mode)
        side.append(self.mode)
        self.generate = button("Создать модель", lambda: self.request(False), "primary")
        side.append(self.generate)
        side.append(Gtk.Separator())
        side.append(label("Детали", "section"))
        self.part_list = Gtk.DropDown.new_from_strings([])
        accessible(self.part_list, "Выбранная деталь")
        self.part_list.connect("notify::selected", self.change_selection)
        side.append(self.part_list)
        self.inspector = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        side.append(self.inspector)
        self.profile_section = Gtk.Expander(label="Контур детали")
        self.point_editor = ProfileEditor(
            self.apply_point, self.notify_status, self.request_point, self.update_guide
        )
        self.profile_section.set_child(self.point_editor)
        self.profile_section.connect("notify::expanded", lambda *_: self.update_guide())
        self.inspector.append(self.profile_section)
        self.fields = {}
        self.field_text = {}
        for attr, title, lo, hi, digits, step in (
            ("size", "Размеры, м", MIN_SIZE, MAX_LENGTH, 3, 0.01),
            ("center", "Центр, м", -MAX_LENGTH, MAX_LENGTH, 3, 0.01),
            ("rotation", "Поворот, °", -360, 360, 2, 1),
        ):
            self.inspector.append(label(title, "muted"))
            row = Gtk.Box(spacing=6, homogeneous=True)
            row.add_css_class("compact")
            self.fields[attr] = []
            for axis in "XYZ":
                group = Gtk.Box(spacing=3)
                group.append(label(axis, "muted"))
                spin = Gtk.SpinButton.new_with_range(lo, hi, step)
                spin.set_digits(digits)
                spin.set_width_chars(5)
                spin.set_max_width_chars(6)
                accessible(spin, f"{title}, {axis}")
                group.append(spin)
                row.append(group)
                self.fields[attr].append(spin)
            self.inspector.append(row)
        self.bore_row = Gtk.Box(spacing=6)
        self.bore_row.append(label("Внутренний диаметр, м", "muted"))
        self.bore = Gtk.SpinButton.new_with_range(0.000001, MAX_LENGTH, 0.001)
        self.bore.set_digits(6)
        accessible(self.bore, "Внутренний диаметр по X, м")
        self.bore_row.append(self.bore)
        self.inspector.append(self.bore_row)
        self.bore_row.set_visible(False)
        self.inspector.append(button("Применить значения", self.apply_numeric))
        self.inspector.append(Gtk.Separator())
        self.inspector.append(label("Правка выбранной детали", "section"))
        self.adjust = Gtk.Entry(placeholder_text="Например, сделай вдвое тоньше")
        self.adjust.set_max_length(MAX_PROMPT)
        accessible(self.adjust, "Инструкция для выбранной детали")
        self.adjust.connect("activate", lambda *_: self.request(True))
        self.inspector.append(self.adjust)
        self.inspector.append(button("Изменить деталь", lambda: self.request(True)))
        self.back = button("Отменить действие", self.undo)
        side.append(self.back)
        pane = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        layout.set_end_child(pane)
        toolbar = Gtk.Box(spacing=8)
        toolbar.add_css_class("toolbar")
        pane.append(toolbar)
        self.stats = label("", "muted")
        self.stats.set_hexpand(True)
        self.stats.set_ellipsize(Pango.EllipsizeMode.END)
        toolbar.append(self.stats)
        wire = Gtk.ToggleButton(label="Сетка")
        wire.connect("toggled", self.toggle_wire)
        toolbar.append(wire)
        toolbar.append(button("Вписать", lambda: self.view.reset_camera()))
        overlay = Gtk.Overlay(hexpand=True, vexpand=True)
        pane.append(overlay)
        self.view = Viewport(self.select, self.notify_status)
        overlay.set_child(self.view)
        self.empty = label("Введите описание и нажмите «Создать модель».", "empty")
        self.empty.set_halign(Gtk.Align.CENTER)
        self.empty.set_valign(Gtk.Align.CENTER)
        self.empty.set_can_target(False)
        self.empty.set_wrap(True)
        overlay.add_overlay(self.empty)
        bottom = Gtk.Box(spacing=8)
        bottom.add_css_class("status")
        self.spinner = Gtk.Spinner()
        bottom.append(self.spinner)
        self.status = label("", "muted")
        self.status.set_wrap(True)
        bottom.append(self.status)
        self.download = button("Скачать модель", self.download_weights)
        bottom.append(self.download)
        root.append(bottom)
        self.refresh()
        self.notify_status(f"Локальная модель · {self.planner.model}")

    def closing(self, *_):
        self.closed = True
        self.cancel_download.set()
        return False

    def notify_status(self, message):
        self.status.set_text(message)

    def controls(self):
        scene = self.doc.scene is not None
        self.download.set_visible(self.missing_model is not None)
        self.download.set_sensitive(not self.busy)
        for w in (self.generate, self.prompt, self.opener, self.mode):
            w.set_sensitive(not self.busy)
        for w in (self.saver, self.exporter, self.inspector, self.part_list):
            w.set_sensitive(scene and not self.busy)
        self.back.set_sensitive(bool(self.doc.history) and not self.busy)
        self.spinner.set_visible(self.busy)
        self.spinner.set_spinning(self.busy)

    def refresh(self, *, reset_point=False):
        self.syncing = True
        scene = self.doc.scene
        parts = scene.parts if scene else ()
        self.selected = min(self.selected, max(0, len(parts) - 1))
        self.mode.set_selected(MODES.index(self.doc.mode))
        self.part_list.set_model(
            Gtk.StringList.new([f"{i + 1}. {p.name}" for i, p in enumerate(parts)])
        )
        meshes = [build(p, self.doc.mode) for p in parts]
        self.view.set_meshes(meshes)
        if parts:
            self.select(self.selected, reset_point=reset_point)
        else:
            self.point_editor.set_part(None)
            self.profile_section.set_visible(False)
            self.view.set_guide()
        self.stats.set_text(
            f"Деталей: {len(parts)}   ·   Граней: {sum(len(m.faces) for m in meshes):,}".replace(
                ",", " "
            )
        )
        self.empty.set_visible(not parts)
        self.syncing = False
        self.controls()

    def select(self, index, *, reset_point=False):
        if self.busy or not self.doc.scene or not 0 <= index < len(self.doc.scene.parts):
            return
        reset_point = reset_point or self.selected != index
        self.selected = index
        self.part_list.set_selected(index)
        part = self.doc.scene.parts[index]
        self.profile_section.set_visible(part.shape in PROFILE_SHAPES)
        self.point_editor.set_part(part, reset_selection=reset_point)
        for attr, fields in self.fields.items():
            for spin, value in zip(fields, getattr(part, attr)):
                spin.set_value(value)
                self.field_text[spin] = spin.get_text()
        self.bore_row.set_visible(part.inner_diameter is not None)
        if part.inner_diameter is not None:
            self.bore.set_value(part.inner_diameter)
            self.field_text[self.bore] = self.bore.get_text()
        self.view.selected = index
        self.update_guide()

    def update_guide(self):
        if self.profile_section.get_expanded() and self.point_editor.part:
            self.view.set_guide(
                profile_guide(
                    self.point_editor.part, self.point_editor.selected_index, self.doc.mode
                )
            )
        else:
            self.view.set_guide()

    def change_selection(self, widget, _):
        if not self.syncing:
            self.select(widget.get_selected())

    def install(self, scene, mode):
        self.doc.apply(scene, mode)
        self.selected = 0
        self.refresh(reset_point=True)
        self.view.reset_camera()

    def change_mode(self, widget, _):
        if not self.syncing and not self.busy:
            self.doc.apply(self.doc.scene, MODES[widget.get_selected()])
            self.refresh()

    def toggle_wire(self, widget):
        self.view.wire = widget.get_active()
        self.view.queue_render()

    def apply_numeric(self):
        part = self.doc.scene.parts[self.selected]
        values = part.data()
        for key, fields in self.fields.items():
            updated = list(values[key])
            for i, spin in enumerate(fields):
                if spin.get_text() != self.field_text[spin]:
                    spin.update()
                    updated[i] = spin.get_value()
            values[key] = updated
        if (
            part.inner_diameter is not None
            and self.bore.get_text() != self.field_text[self.bore]
        ):
            self.bore.update()
            values["inner_diameter"] = self.bore.get_value()
        try:
            updated_part = Part.parse(values)
        except ValueError as exc:
            self.notify_status(str(exc))
            return
        self.doc.replace(self.selected, updated_part)
        self.refresh()
        self.notify_status(f"Размеры и положение применены · {part.name}")

    def undo(self):
        self.doc.undo()
        self.refresh()
        self.notify_status("Действие отменено")

    def apply_point(self, part):
        if self.busy or not self.doc.scene:
            return
        self.doc.replace(self.selected, part)
        self.refresh()
        self.notify_status(f"Точка {self.point_editor.selected_index + 1} · {part.name}")

    def request_point(self, point_index, prompt):
        if not self.doc.scene:
            return
        scene, index = self.doc.scene, self.selected
        self.run_planner(
            prompt,
            lambda text: self.planner.edit_point(text, scene, index, point_index),
            self.apply_point,
            f"Ollama рассчитывает положение точки {point_index + 1}…",
        )

    def request(self, edit):
        if edit and not self.doc.scene:
            return
        if edit:
            scene, index = self.doc.scene, self.selected

            def apply(part):
                self.doc.replace(index, part)
                self.refresh()
                self.notify_status(f"Деталь изменена · {part.name}")

            self.run_planner(
                self.adjust.get_text(),
                lambda text: self.planner.edit(text, scene, index),
                apply,
                "Ollama рассчитывает форму выбранной детали…",
            )
        else:

            def install(scene):
                self.install(scene, self.doc.mode)
                self.notify_status(
                    "Модель готова. Выберите деталь в окне просмотра или в списке."
                )

            self.run_planner(
                self.prompt.get_text(),
                self.planner.generate,
                install,
                "Ollama строит модель по описанию…",
            )

    def run_planner(self, prompt, compute, apply, progress):
        if self.busy:
            return
        prompt = prompt.strip()
        if not prompt:
            self.notify_status("Введите текст запроса")
            return
        self.missing_model = None
        self.run_task(lambda: compute(prompt), apply, progress)

    def download_weights(self):
        missing = self.missing_model
        if self.busy or missing is None:
            return

        def report(message):
            GLib.idle_add(show_progress, message)

        def show_progress(message):
            if self.busy and not self.closed and self.missing_model is missing:
                self.notify_status(message)
            return False

        def complete(_):
            self.missing_model = None
            self.notify_status(f"Модель {missing.model} загружена. Повторите запрос.")

        self.run_task(
            lambda: pull_model(
                missing.url, missing.model, report, self.cancel_download, opener=missing.opener
            ),
            complete,
            f"Загрузка {missing.model}…",
        )

    def run_task(self, compute, apply, progress):
        scene = self.doc.scene
        self.busy = True
        self.controls()
        self.notify_status(progress)

        def finish(value, error):
            if self.closed:
                return False
            self.busy = False
            if isinstance(error, ModelMissing):
                self.missing_model = error
                size = " (около 3,3 ГБ)" if error.model == "qwen3.5:4b" else ""
                self.notify_status(f"Для генерации нужна модель {error.model}{size}.")
            elif error:
                self.notify_status(str(error))
            elif self.doc.scene is not scene:
                self.notify_status("Во время запроса сцена изменилась. Повторите правку.")
            else:
                apply(value)
            self.controls()
            return False

        def work():
            try:
                value = compute()
            except (PlannerError, ValueError) as exc:
                GLib.idle_add(finish, None, exc)
            else:
                GLib.idle_add(finish, value, None)

        threading.Thread(target=work, daemon=True).start()

    def open_project(self, file):
        path = file.get_path()
        try:
            if not path:
                raise ValueError("Выберите локальный файл")
            self.install(*load_project(path))
        except (OSError, ValueError, RecursionError) as exc:
            self.notify_status(f"Ошибка файла · {exc}")
            return
        self.file_folder = str(Path(path).parent)
        self.notify_status(f"Файл открыт · {path}")

    def file_dialog(self, action):
        opening = action == "open"
        project = action != "obj"
        ext = "xtri" if project else "obj"
        dialog = Gtk.FileDialog(
            title={"open": "Открыть проект", "save": "Сохранить проект", "obj": "Экспорт OBJ"}[
                action
            ]
        )
        folder = os.environ.get("XTRIFID_FILES") or self.file_folder or str(Path.home())
        dialog.set_initial_folder(Gio.File.new_for_path(folder))
        file_filter = Gtk.FileFilter()
        file_filter.set_name("Xtrifid (*.xtri)" if project else "Wavefront OBJ (*.obj)")
        file_filter.add_pattern("*." + ext)
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(file_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(file_filter)
        if not opening:
            dialog.set_initial_name("model." + ext)

        def done(sender, result):
            try:
                file = (sender.open_finish if opening else sender.save_finish)(result)
                if opening:
                    self.open_project(file)
                    return
                path = file.get_path()
                if not path:
                    raise ValueError("Выберите локальный файл")
                if project:
                    save_project(path, self.doc.scene, self.doc.mode)
                else:
                    export_obj(path, self.doc.scene, self.doc.mode)
                self.file_folder = str(Path(path).parent)
                self.notify_status(f"Файл записан · {path}")
            except GLib.Error as exc:
                if not exc.matches(Gtk.dialog_error_quark(), Gtk.DialogError.DISMISSED):
                    self.notify_status(exc.message)
            except (OSError, ValueError, RecursionError) as exc:
                self.notify_status(f"Ошибка файла · {exc}")

        (dialog.open if opening else dialog.save)(self, None, done)
