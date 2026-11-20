import os
import threading
from dataclasses import replace
from pathlib import Path

from xtrifid.gtk import Gio, GLib, Gtk, Pango

from .geometry import MeshBudgetError, build, check_scene_budget
from .local_server import pull_model
from .loft_editor import LoftEditor
from .mask_editor import MaskEditor
from .model import (
    DETAILS,
    MAX_BODY_SCALE,
    MAX_LENGTH,
    MAX_PARTS,
    MIN_BODY_SCALE,
    MIN_SIZE,
    MODES,
    PHYSIQUES,
    POSES,
    Document,
    Part,
)
from .planner import MAX_PROMPT, ModelMissing, Planner, PlannerError
from .profile import PROFILE_SHAPES, profile_guide
from .profile_editor import ProfileEditor
from .reference import photo_scene, reference_scene
from .reference_ui import ReferenceDialog
from .sculpt_panel import SculptPanel
from .simulation_panel import SimulationPanel
from .storage import export_obj, load_project, save_project
from .surface import baked_part
from .texture_panel import TexturePanel
from .texture_plan import text_to_texture
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

DETAIL_LABELS = ("Низкая", "Средняя", "Высокая")
ZONE_LABELS = {
    "head": "Голова",
    "ears": "Уши",
    "hands": "Кисти",
    "feet": "Стопы",
    "shoulders": "Плечи",
    "waist": "Талия",
    "nose": "Нос",
    "lips": "Губы",
    "chin": "Подбородок",
}


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
        self.meshes = []
        self.reference_dialog = None
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
        self.prompt = Gtk.Entry(placeholder_text="Опишите форму, пропорции и детали")
        self.prompt.set_max_length(MAX_PROMPT)
        accessible(self.prompt, "Описание новой модели")
        self.prompt.connect("activate", lambda *_: self.request(False))
        side.append(self.prompt)
        self.mode = Gtk.DropDown.new_from_strings(["Game Ready", "Для рендера"])
        accessible(self.mode, "Профиль геометрии")
        self.detail = Gtk.DropDown.new_from_strings(["Выберите…", *DETAIL_LABELS])
        accessible(self.detail, "Детализация модели")
        settings = Gtk.Grid(column_spacing=8, row_spacing=3)
        for column, (title, selector) in enumerate(
            (("Профиль сетки", self.mode), ("Детализация", self.detail))
        ):
            settings.attach(label(title, "muted"), column, 0, 1, 1)
            selector.set_hexpand(True)
            settings.attach(selector, column, 1, 1, 1)
            selector.connect("notify::selected", self.change_settings)
        side.append(settings)
        self.generate = button("Создать модель", lambda: self.request(False), "primary")
        side.append(self.generate)
        sources = Gtk.Box(spacing=8)
        self.low_poly = Gtk.CheckButton(label="Low Poly")
        self.low_poly.set_tooltip_text("Упростить сетку и показать плоские грани")
        self.low_poly.connect("toggled", self.change_settings)
        self.reference_button = button("По референсам…", self.choose_references)
        self.reference_button.set_hexpand(True)
        sources.append(self.low_poly)
        sources.append(self.reference_button)
        side.append(sources)
        side.append(Gtk.Separator())
        side.append(label("Детали", "section"))
        self.part_list = Gtk.DropDown.new_from_strings([])
        accessible(self.part_list, "Выбранная деталь")
        self.part_list.connect("notify::selected", self.change_selection)
        side.append(self.part_list)
        self.inspector = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        side.append(self.inspector)
        self.anatomy = Gtk.Expander(label="Анатомия и поза")
        self.inspector.append(self.anatomy)
        anatomy = Gtk.Grid(column_spacing=6, row_spacing=5)
        self.anatomy.set_child(anatomy)
        self.pose = Gtk.DropDown.new_from_strings(["Стоя", "В шаге", "Руки в стороны"])
        self.physique = Gtk.DropDown.new_from_strings(["Нейтральное", "Мужское", "Женское"])
        self.zone = Gtk.DropDown.new_from_strings(list(ZONE_LABELS.values()))
        self.zone_scale = Gtk.SpinButton.new_with_range(MIN_BODY_SCALE, MAX_BODY_SCALE, 0.01)
        self.zone_scale.set_digits(2)
        for row, (title, control) in enumerate(
            (
                ("Телосложение", self.physique),
                ("Поза", self.pose),
                ("Область", self.zone),
                ("Масштаб области", self.zone_scale),
            )
        ):
            anatomy.attach(label(title, "muted"), 0, row, 1, 1)
            accessible(control, title)
            anatomy.attach(control, 1, row, 1, 1)
        self.zone.connect("notify::selected", lambda *_: self.read_zone())
        anatomy.attach(button("Применить к телу", self.apply_anatomy), 0, 4, 2, 1)
        self.profile_section = Gtk.Expander(label="Контур детали")
        self.point_editor = ProfileEditor(
            self.apply_point, self.notify_status, self.request_point, self.update_guide
        )
        self.profile_section.set_child(self.point_editor)
        self.profile_section.connect("notify::expanded", lambda *_: self.update_guide())
        self.inspector.append(self.profile_section)
        self.loft = LoftEditor(self.replace_part, self.notify_status, self.update_guide)
        self.inspector.append(self.loft)
        self.sculpt_editor = SculptPanel(self.replace_part, self.notify_status)
        self.inspector.append(self.sculpt_editor)
        self.mask_editor = MaskEditor(self.replace_part, self.notify_status)
        self.inspector.append(self.mask_editor)
        self.simulation = SimulationPanel(
            self.simulation_preview, self.bake_simulation, self.notify_status
        )
        self.material = TexturePanel(
            self.apply_texture, self.notify_status, self.request_texture
        )
        self.inspector.append(self.simulation)
        self.inspector.append(self.material)
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
        self.focus_part = button("К детали", lambda: self.view.focus_selected())
        self.focus_part.set_tooltip_text("Приблизить выбранную деталь (F)")
        toolbar.append(self.focus_part)
        toolbar.append(button("Вписать", lambda: self.view.reset_camera()))
        overlay = Gtk.Overlay(hexpand=True, vexpand=True)
        pane.append(overlay)
        self.view = Viewport(self.select, self.notify_status)
        overlay.set_child(self.view)
        self.empty = label("Введите описание и выберите детализацию.", "empty")
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
        self.present_scene(None, self.doc.mode)
        self.notify_status(f"Локальная модель · {self.planner.model}")

    def closing(self, *_):
        self.closed = True
        self.cancel_download.set()
        self.simulation.stop()
        if self.reference_dialog:
            self.reference_dialog.close()
        return False

    def notify_status(self, message):
        self.status.set_text(message)

    def controls(self):
        scene = self.doc.scene is not None
        self.download.set_visible(self.missing_model is not None)
        self.download.set_sensitive(not self.busy)
        for w in (
            self.prompt,
            self.opener,
            self.mode,
            self.detail,
            self.low_poly,
            self.reference_button,
        ):
            w.set_sensitive(not self.busy)
        self.generate.set_sensitive(not self.busy and self.detail.get_selected() > 0)
        for w in (self.saver, self.exporter, self.inspector, self.part_list, self.focus_part):
            w.set_sensitive(scene and not self.busy)
        self.back.set_sensitive(bool(self.doc.history) and not self.busy)
        self.spinner.set_visible(self.busy)
        self.spinner.set_spinning(self.busy)

    def present_scene(self, scene, mode, *, reset_point=False, undo=False):
        parts = scene.parts if scene else ()
        if scene:
            check_scene_budget(scene, mode)
        try:
            meshes = [build(p, mode, scene.detail, scene.style) for p in parts]
            self.view.set_meshes(meshes, parts)
        except MemoryError as exc:
            raise MeshBudgetError("Недостаточно памяти для предпросмотра.") from exc
        self.meshes = meshes
        if undo:
            self.doc.undo()
        else:
            self.doc.apply(scene, mode)
        self.syncing = True
        if reset_point:
            self.selected = 0
        self.selected = min(self.selected, max(0, len(parts) - 1))
        self.mode.set_selected(MODES.index(mode))
        if scene:
            self.detail.set_selected(DETAILS.index(scene.detail) + 1)
            self.low_poly.set_active(scene.style == "low_poly")
        self.part_list.set_model(
            Gtk.StringList.new([f"{i + 1}. {p.name}" for i, p in enumerate(parts)])
        )
        if parts:
            self.select(self.selected, reset_point=reset_point)
        else:
            self.anatomy.set_visible(False)
            self.material.set_part(None)
            self.sculpt_editor.set_part(None, True)
            self.mask_editor.set_part(None)
            self.simulation.set_target(None, None, "medium")
            self.loft.set_part(None, True)
            self.point_editor.set_part(None)
            self.profile_section.set_visible(False)
            self.view.set_guide()
        faces = sum(len(mesh.faces) for mesh in meshes)
        tris = sum(len(face) == 3 for mesh in meshes for face in mesh.faces)
        self.stats.set_text(
            f"Деталей {len(parts)}   ·   Треуг. {tris:_}   ·   Четырёхуг. {faces - tris:_}".replace(
                "_", " "
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
        self.anatomy.set_visible(part.shape == "human")
        if part.shape == "human":
            self.pose.set_selected(POSES.index(part.pose))
            self.physique.set_selected(PHYSIQUES.index(part.physique))
            self.read_zone()
        self.profile_section.set_visible(part.shape in PROFILE_SHAPES)
        self.point_editor.set_part(part, reset_selection=reset_point)
        self.loft.set_part(part, reset_point)
        for attr, fields in self.fields.items():
            for spin, value in zip(fields, getattr(part, attr)):
                spin.set_value(value)
                self.field_text[spin] = spin.get_text()
        self.bore_row.set_visible(part.inner_diameter is not None)
        if part.inner_diameter is not None:
            self.bore.set_value(part.inner_diameter)
            self.field_text[self.bore] = self.bore.get_text()
        self.material.set_part(part)
        self.mask_editor.set_part(part)
        self.sculpt_editor.set_part(part, reset_point)
        colliders = [
            (p, m)
            for i, (p, m) in enumerate(zip(self.doc.scene.parts, self.meshes))
            if i != index
        ]
        self.simulation.set_target(part, self.meshes[index], self.doc.scene.detail, colliders)
        self.view.selected = index
        self.focus_part.set_label("К лицу" if part.shape == "human" else "К детали")
        self.update_guide()

    def read_zone(self):
        if self.doc.scene:
            part = self.doc.scene.parts[self.selected]
            if part.shape == "human":
                zone = tuple(ZONE_LABELS)[self.zone.get_selected()]
                self.zone_scale.set_value(dict(part.body_scales).get(zone, 1.0))

    def apply_anatomy(self):
        from .character import with_character_pose

        part = self.doc.scene.parts[self.selected]
        self.zone_scale.update()
        scales = dict(part.body_scales)
        zone = tuple(ZONE_LABELS)[self.zone.get_selected()]
        scales[zone] = self.zone_scale.get_value()
        try:
            updated = Part.parse(
                part.data()
                | {"body_scales": scales, "physique": PHYSIQUES[self.physique.get_selected()]}
            )
            updated = with_character_pose(updated, POSES[self.pose.get_selected()])
            self.replace_part(updated)
        except ValueError as exc:
            self.notify_status(str(exc))
            return
        self.notify_status("Изменения фигуры применены")

    def update_guide(self):
        if self.loft.get_expanded() and self.loft.part and self.doc.scene:
            self.view.set_guide(
                self.loft.guide(self.doc.mode, self.doc.scene.detail, self.doc.scene.style)
            )
        elif self.profile_section.get_expanded() and self.point_editor.part:
            self.view.set_guide(
                profile_guide(
                    self.point_editor.part,
                    self.point_editor.selected_index,
                    self.doc.mode,
                    self.doc.scene.detail,
                    self.doc.scene.style,
                )
            )
        else:
            self.view.set_guide()

    def change_selection(self, widget, _):
        if not self.syncing:
            self.select(widget.get_selected())

    def install(self, scene, mode):
        self.present_scene(scene, mode, reset_point=True)
        self.view.reset_camera()

    def change_settings(self, widget, *_):
        if self.syncing or self.busy:
            return
        scene = self.doc.scene
        mode = MODES[self.mode.get_selected()]
        index = self.detail.get_selected()
        if scene is None:
            self.doc.apply(None, mode)
            self.controls()
            return
        style = self.chosen_style()
        if (
            index
            and mode == self.doc.mode
            and DETAILS[index - 1] == scene.detail
            and style == scene.style
        ):
            return
        try:
            if not index:
                raise ValueError("Выберите детализацию модели.")
            self.present_scene(replace(scene, detail=DETAILS[index - 1], style=style), mode)
        except ValueError as exc:
            self.syncing = True
            self.mode.set_selected(MODES.index(self.doc.mode))
            self.detail.set_selected(DETAILS.index(scene.detail) + 1)
            self.low_poly.set_active(scene.style == "low_poly")
            self.syncing = False
            self.notify_status(str(exc))
        else:
            self.notify_status(
                f"{self.mode.get_selected_item().get_string()} · {DETAIL_LABELS[index - 1]} детализация"
            )

    def chosen_style(self):
        return "low_poly" if self.low_poly.get_active() else "standard"

    def simulation_preview(self, mesh, replace_selected):
        if mesh is not None and self.chosen_style() == "low_poly":
            mesh = replace(mesh, smooth=(False,) * len(mesh.faces))
        self.view.set_simulation_preview(mesh, replace_selected)

    def bake_simulation(self, name, mesh, replace_selected):
        scene = self.doc.scene
        if scene is None or self.busy:
            return
        if not replace_selected and len(scene.parts) >= MAX_PARTS:
            raise ValueError("Достигнут предел числа деталей сцены")
        part = baked_part(name, mesh)
        if replace_selected:
            part = replace(part, texture=scene.parts[self.selected].texture)
            self.replace_part(part)
        else:
            self.present_scene(replace(scene, parts=(*scene.parts, part)), self.doc.mode)
            self.select(len(scene.parts))
        self.notify_status(f"Кадр сохранён в геометрии · {name}")

    def apply_texture(self, texture):
        if self.doc.scene:
            part = self.doc.scene.parts[self.selected]
            self.replace_part(replace(part, texture=texture))
            self.notify_status("Текстура применена" if texture else "Текстура снята")

    def request_texture(self, prompt, seed, resolution):
        self.run_planner(
            prompt,
            lambda text: text_to_texture(self.planner, text, seed, resolution),
            self.apply_texture,
            "ИИ подбирает узор и палитру…",
        )

    def choose_references(self):
        if not self.detail.get_selected():
            self.notify_status("Выберите детализацию модели.")
            self.detail.grab_focus()
            return
        dialog = ReferenceDialog(
            self,
            self.generate_reference,
            os.environ.get("XTRIFID_FILES") or self.file_folder,
            self.prompt.get_text(),
            os.environ.get("OLLAMA_URL", ""),
        )
        self.reference_dialog = dialog
        dialog.connect("close-request", lambda *_: setattr(self, "reference_dialog", None))
        dialog.present()

    def generate_reference(self, request):
        if self.busy or self.closed or not self.detail.get_selected():
            return
        mode, detail, style = (
            self.doc.mode,
            DETAILS[self.detail.get_selected() - 1],
            self.chosen_style(),
        )
        try:
            check_scene_budget(None, mode)
        except ValueError as exc:
            self.notify_status(str(exc))
            return

        def compute():
            if request.photo:
                return photo_scene(
                    self.planner,
                    request.front,
                    request.side,
                    height_m=request.height,
                    detail=detail,
                    prompt=request.prompt,
                    mode=mode,
                    style=style,
                )
            scene = reference_scene(
                request.front,
                request.side,
                height_m=request.height,
                depth_m=request.depth,
                detail=detail,
                tolerance=request.tolerance,
                rounded=request.rounded,
            )
            return replace(scene, style=style)

        def apply(scene):
            self.install(scene, mode)
            self.file_folder = str(request.front.parent)
            self.notify_status("Модель по референсу готова")

        self.missing_model = None
        self.run_task(compute, apply, "Построение модели по референсу…")

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
            self.replace_part(updated_part)
        except ValueError as exc:
            self.notify_status(str(exc))
            return
        self.notify_status(f"Размеры и положение применены · {part.name}")

    def undo(self):
        if not self.doc.history:
            return
        try:
            self.present_scene(*self.doc.history[-1], undo=True)
        except ValueError as exc:
            self.notify_status(str(exc))
            return
        self.notify_status("Действие отменено")

    def replace_part(self, part):
        self.present_scene(self.doc.scene.with_part(self.selected, part), self.doc.mode)

    def apply_point(self, part):
        if self.busy or not self.doc.scene:
            return
        try:
            self.replace_part(part)
        except ValueError as exc:
            self.notify_status(str(exc))
            return
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
        if not self.detail.get_selected():
            self.notify_status("Выберите детализацию модели.")
            return
        if edit:
            scene, index = self.doc.scene, self.selected

            def apply(part):
                self.present_scene(scene.with_part(index, part), self.doc.mode)
                self.notify_status(f"Деталь изменена · {part.name}")

            self.run_planner(
                self.adjust.get_text(),
                lambda text: self.planner.edit(text, scene, index),
                apply,
                "Ollama рассчитывает форму выбранной детали…",
            )
        else:

            def install(scene):
                self.install(replace(scene, style=self.chosen_style()), self.doc.mode)
                self.notify_status(
                    "Модель готова. Выберите деталь в окне просмотра или в списке."
                )

            detail = DETAILS[self.detail.get_selected() - 1]
            self.run_planner(
                self.prompt.get_text(),
                lambda text: self.planner.generate(text, detail=detail),
                install,
                "Построение модели по описанию…",
                new_scene=True,
            )

    def run_planner(self, prompt, compute, apply, progress, *, new_scene=False):
        if self.busy:
            return
        prompt = prompt.strip()
        if not prompt:
            self.notify_status("Введите текст запроса")
            return
        try:
            check_scene_budget(None if new_scene else self.doc.scene, self.doc.mode)
        except ValueError as exc:
            self.notify_status(str(exc))
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
                missing.url,
                missing.model,
                report,
                self.cancel_download,
                opener=missing.opener,
                memory_limit=missing.memory_limit,
            ),
            complete,
            f"Загрузка {missing.model}…",
        )

    def run_task(self, compute, apply, progress):
        scene = self.doc.scene
        self.simulation.stop()
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
                try:
                    apply(value)
                except ValueError as exc:
                    self.notify_status(str(exc))
            self.controls()
            return False

        def work():
            try:
                value = compute()
            except MemoryError:
                GLib.idle_add(finish, None, MeshBudgetError("Недостаточно памяти для запроса."))
            except (PlannerError, ValueError, OSError, GLib.Error, RecursionError) as exc:
                GLib.idle_add(finish, None, exc)
            else:
                GLib.idle_add(finish, value, None)

        threading.Thread(target=work, daemon=True).start()

    def open_project(self, file):
        path = file.get_path()
        try:
            if not path:
                raise ValueError("Выберите локальный файл")
            scene, mode = load_project(path)
            try:
                self.install(scene, mode)
            except MeshBudgetError as exc:
                for detail in reversed(DETAILS[: DETAILS.index(scene.detail)]):
                    candidate = replace(scene, detail=detail)
                    try:
                        check_scene_budget(candidate, mode)
                    except MeshBudgetError:
                        continue
                    self.offer_lower_detail(path, candidate, mode, str(exc))
                    return
                raise
        except (OSError, ValueError, RecursionError) as exc:
            self.notify_status(f"Ошибка файла · {exc}")
            return
        self.file_folder = str(Path(path).parent)
        self.notify_status(f"Файл открыт · {path}")

    def offer_lower_detail(self, path, candidate, mode, reason):
        level = DETAIL_LABELS[DETAILS.index(candidate.detail)]
        dialog = Gtk.AlertDialog(
            message=f"Открыть проект с детализацией «{level}»?",
            detail=reason,
            buttons=["Отмена", "Открыть"],
            cancel_button=0,
            default_button=1,
        )
        self.notify_status(reason)

        def chosen(sender, result):
            try:
                if sender.choose_finish(result) != 1 or self.closed:
                    return
                self.install(candidate, mode)
            except (GLib.Error, ValueError) as exc:
                self.notify_status(str(exc))
                return
            self.file_folder = str(Path(path).parent)
            self.notify_status(f"Файл открыт · {level} детализация · {path}")

        dialog.choose(self, None, chosen)

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
