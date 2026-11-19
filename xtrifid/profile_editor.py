"""Edit a part's planar or radial profile in its local metre coordinates."""

from math import hypot, isfinite, tau

from .gtk import Gtk
from .planner import MAX_PROMPT
from .profile import PROFILE_SHAPES, move_profile_point, profile_points


class ProfileEditor(Gtk.Box):
    def __init__(self, on_apply, on_error, on_ai, on_select=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.on_apply, self.on_error, self.on_ai = on_apply, on_error, on_ai
        self.on_select = on_select
        self.part = None
        self.points = ()
        self.drag_scale = None
        self.preview = None

        self.canvas = Gtk.DrawingArea(content_width=140, content_height=142, hexpand=True)
        self.canvas.set_tooltip_text("Выберите вершину контура и перетащите её")
        self.canvas.update_property([Gtk.AccessibleProperty.LABEL], ["Контур детали"])
        self.canvas.set_draw_func(self.draw)
        self.drag = Gtk.GestureDrag(button=1)
        self.drag.connect("drag-begin", self.begin_drag)
        self.drag.connect("drag-update", self.update_drag)
        self.drag.connect("drag-end", self.end_drag)
        self.drag.connect("cancel", self.cancel_drag)
        self.canvas.add_controller(self.drag)
        self.append(self.canvas)

        row = Gtk.Box(spacing=5)
        row.add_css_class("compact")
        row.append(Gtk.Label(label="Вершина"))
        self.selector = Gtk.SpinButton.new_with_range(1, 2, 1)
        self.selector.set_width_chars(2)
        self.selector.set_numeric(True)
        self.selector.update_property([Gtk.AccessibleProperty.LABEL], ["Номер вершины"])
        self.selection_handler = self.selector.connect("value-changed", self.select)
        row.append(self.selector)
        self.count = Gtk.Label()
        row.append(self.count)
        self.append(row)

        row = Gtk.Box(spacing=6)
        self.axes = []
        self.fields = []
        for _ in range(2):
            axis = Gtk.Label()
            axis.add_css_class("muted")
            entry = Gtk.Entry(width_chars=6, max_width_chars=8, hexpand=True)
            entry.set_input_purpose(Gtk.InputPurpose.NUMBER)
            entry.set_max_length(40)
            entry.connect("activate", self.apply_coordinates)
            self.axes.append(axis)
            self.fields.append(entry)
            row.append(axis)
            row.append(entry)
        self.append(row)
        self.apply_button = Gtk.Button(label="Применить координаты")
        self.apply_button.connect("clicked", self.apply_coordinates)
        self.append(self.apply_button)

        self.prompt = Gtk.Entry(placeholder_text="Например, сдвинь вершину на 2 см вверх")
        self.prompt.update_property([Gtk.AccessibleProperty.LABEL], ["Инструкция для вершины"])
        self.prompt.set_max_length(MAX_PROMPT)
        self.prompt.connect("activate", self.ask_ai)
        self.append(self.prompt)
        ask = Gtk.Button(label="Изменить вершину по описанию")
        ask.connect("clicked", self.ask_ai)
        self.append(ask)
        self.set_visible(False)

    @property
    def selected_index(self):
        return self.selector.get_value_as_int() - 1

    def set_part(self, part, *, reset_selection=False):
        previous = self.selected_index
        index = 0 if reset_selection else previous
        self.drag_scale = self.preview = None
        self.part = part if part is not None and part.shape in PROFILE_SHAPES else None
        self.points = profile_points(self.part) if self.part else ()
        self.set_visible(bool(self.points))
        if reset_selection or not self.points:
            self.prompt.set_text("")
        if not self.points:
            return
        names = ("X", "Y") if part.shape in ("box", "extrusion") else ("Радиус", "Z")
        for name, axis, field in zip(names, self.axes, self.fields):
            axis.set_text(f"{name}, м")
            field.update_property([Gtk.AccessibleProperty.LABEL], [f"Координата {name}, м"])
        self.count.set_text(f"из {len(self.points)}")
        with self.selector.handler_block(self.selection_handler):
            self.selector.set_range(1, len(self.points))
            self.selector.set_value(min(index + 1, len(self.points)))
        if self.selected_index != previous:
            self.select()
        else:
            self.sync_fields()

    def select(self, *_):
        self.drag_scale = self.preview = None
        self.prompt.set_text("")
        self.sync_fields()
        if self.on_select and 0 <= self.selected_index < len(self.points):
            self.on_select()

    def sync_fields(self):
        if 0 <= self.selected_index < len(self.points):
            point = self.preview or self.points[self.selected_index]
            for field, value in zip(self.fields, point):
                field.set_text(format(value, ".12g"))
        self.canvas.queue_draw()

    def apply_coordinates(self, *_):
        if not self.part or not self.is_sensitive():
            return
        try:
            point = []
            for field, original in zip(self.fields, self.points[self.selected_index]):
                text = field.get_text().strip().replace(",", ".")
                value = original if text == format(original, ".12g") else float(text)
                if not isfinite(value):
                    raise ValueError
                point.append(value)
        except ValueError:
            self.sync_fields()
            self.on_error("Введите две конечные координаты в метрах")
            return
        self.apply_point(tuple(point))

    def apply_point(self, point):
        try:
            updated = move_profile_point(self.part, self.selected_index, point)
        except ValueError as exc:
            self.sync_fields()
            self.on_error(str(exc))
            return
        if updated != self.part:
            self.on_apply(updated)
        else:
            self.sync_fields()

    def ask_ai(self, *_):
        if not self.part or not self.is_sensitive():
            return
        prompt = self.prompt.get_text().strip()
        if prompt:
            self.on_ai(self.selected_index, prompt)
        else:
            self.on_error("Введите инструкцию для выбранной вершины")

    def frame(self):
        # Fit both axes at one scale, including the rotation axis for hollow profiles.
        xs, ys = zip(*self.points)
        left, right = min(0, min(xs)), max(0, max(xs))
        low, high = min(ys), max(ys)
        width, height = self.canvas.get_width(), self.canvas.get_height()
        scale = min(max(width - 32, 1) / (right - left), max(height - 32, 1) / (high - low))
        return (
            width / 2 - (left + right) * scale / 2,
            height / 2 + (low + high) * scale / 2,
            scale,
        )

    def screen_point(self, point):
        ox, oy, scale = self.frame()
        return ox + point[0] * scale, oy - point[1] * scale

    def begin_drag(self, gesture, x, y):
        if not self.points or not self.is_sensitive():
            return
        distances = [hypot(sx - x, sy - y) for sx, sy in map(self.screen_point, self.points)]
        i = min(range(len(distances)), key=distances.__getitem__)
        if distances[i] <= 12:
            self.selector.set_value(i + 1)
            self.drag_scale = self.frame()[2]

    def update_drag(self, gesture, dx, dy):
        if self.drag_scale is None or not self.is_sensitive():
            return
        a, b = self.points[self.selected_index]
        self.preview = (
            (a + dx / self.drag_scale, b - dy / self.drag_scale) if hypot(dx, dy) >= 3 else None
        )
        self.sync_fields()

    def end_drag(self, gesture, dx, dy):
        if self.drag_scale is None:
            return
        self.update_drag(gesture, dx, dy)
        point = self.preview
        self.drag_scale = self.preview = None
        if point is not None and self.is_sensitive():
            self.apply_point(point)
        else:
            self.sync_fields()

    def cancel_drag(self, *_):
        self.drag_scale = self.preview = None
        self.sync_fields()

    def draw(self, area, cr, width, height):
        cr.set_source_rgb(0.07, 0.085, 0.11)
        cr.paint()
        if not self.points:
            return
        ox, oy, scale = self.frame()
        cr.set_source_rgb(0.23, 0.27, 0.32)
        cr.set_line_width(1)
        cr.move_to(ox, 0)
        cr.line_to(ox, height)
        cr.move_to(0, oy)
        cr.line_to(width, oy)
        cr.stroke()

        points = list(self.points)
        if self.preview is not None:
            points[self.selected_index] = self.preview
        pixels = [(ox + a * scale, oy - b * scale) for a, b in points]
        cr.move_to(*pixels[0])
        for xy in pixels[1:]:
            cr.line_to(*xy)
        cr.close_path()
        cr.set_source_rgba(0.34, 0.72, 0.65, 0.18)
        cr.fill_preserve()
        cr.set_source_rgb(0.58, 0.85, 0.80)
        cr.set_line_width(1.5)
        cr.stroke()
        for x, y in pixels:
            cr.arc(x, y, 3.5, 0, tau)
            cr.fill()
        x, y = pixels[self.selected_index]
        cr.set_source_rgb(1, 0.80, 0.42)
        cr.arc(x, y, 5, 0, tau)
        cr.fill()
        cr.arc(x, y, 8, 0, tau)
        cr.stroke()
