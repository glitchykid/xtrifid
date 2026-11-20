"""Retouch either input silhouette and rebuild the volume as one undoable edit."""

from dataclasses import replace
from math import floor

import cairo
import numpy as np

from .gtk import Gdk, Gtk
from .sculpt import parse_masks


class MaskEditor(Gtk.Expander):
    def __init__(self, on_apply, on_error):
        super().__init__(label="Силуэты модели")
        self.on_apply, self.on_error = on_apply, on_error
        self.part = None
        self.masks = []
        self.cursor = (0, 0)
        self.start = None
        self.last = None
        self.view = Gtk.DropDown.new_from_strings(["Вид спереди"])
        self.ink = Gtk.DropDown.new_from_strings(["Добавить", "Стереть"])
        self.brush = Gtk.SpinButton.new_with_range(1, 12, 1)
        self.brush.set_value(3)
        for control, name in (
            (self.view, "Редактируемый вид"),
            (self.ink, "Действие кисти"),
            (self.brush, "Радиус кисти, пиксели"),
        ):
            control.update_property([Gtk.AccessibleProperty.LABEL], [name])
        controls = Gtk.Box(spacing=4)
        for w in (self.ink, Gtk.Label(label="Радиус"), self.brush):
            controls.append(w)
        self.canvas = Gtk.DrawingArea(
            content_width=140, content_height=150, hexpand=True, focusable=True
        )
        self.canvas.set_tooltip_text(
            "Ведите кистью по силуэту. С клавиатуры — стрелки и пробел."
        )
        self.canvas.update_property([Gtk.AccessibleProperty.LABEL], ["Правка силуэта кистью"])
        self.canvas.set_draw_func(self.draw)
        drag = Gtk.GestureDrag(button=1)
        drag.connect("drag-begin", self.begin)
        drag.connect("drag-update", self.move)
        drag.connect("drag-end", self.end)
        self.canvas.add_controller(drag)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.key)
        self.canvas.add_controller(keys)
        self.view.connect("notify::selected", lambda *_: self.canvas.queue_draw())
        self.brush.connect("value-changed", lambda *_: self.canvas.queue_draw())
        self.rounded = Gtk.CheckButton(label="Округлить сечения")
        apply = Gtk.Button(label="Применить силуэты")
        apply.connect("clicked", lambda *_: self.apply_changes())
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        for w in (self.view, self.canvas, controls, self.rounded, apply):
            box.append(w)
        self.set_child(box)
        self.set_visible(False)

    def set_part(self, part):
        target = part if part and part.shape == "reference" else None
        self.set_visible(target is not None)
        if target is self.part:
            return
        old = self.view.get_selected()
        self.part = target
        self.masks = []
        self.start = self.last = None
        if target:
            self.rounded.set_active(target.rounded)
            self.masks = [
                np.array([[cell == "1" for cell in row] for row in view])
                for view in target.masks
            ]
            self.view.set_model(
                Gtk.StringList.new(["Вид спереди", "Вид справа"][: len(self.masks)])
            )
            self.view.set_selected(min(old, len(self.masks) - 1))
            self.cursor = (self.mask.shape[1] // 2, self.mask.shape[0] // 2)
        self.canvas.queue_draw()

    @property
    def mask(self):
        return self.masks[self.view.get_selected()]

    def box(self, w, h):
        rows, cols = self.mask.shape
        scale = min((w - 4) / cols, (h - 4) / rows)
        return scale, (w - scale * cols) / 2, (h - scale * rows) / 2

    def draw(self, _, cr, w, h):
        if not self.masks or w < 5 or h < 5:
            return
        mask = self.mask
        scale, x, y = self.box(w, h)
        rows, cols = mask.shape
        rgba = np.where(mask, np.uint32(0xFFCCE4DF), np.uint32(0xFF222C35))
        surf = cairo.ImageSurface.create_for_data(rgba, cairo.FORMAT_ARGB32, cols, rows)
        cr.translate(x, y)
        cr.scale(scale, scale)
        cr.set_source_surface(surf)
        cr.get_source().set_filter(cairo.FILTER_NEAREST)
        cr.paint()
        cr.set_source_rgb(0.96, 0.5, 0.20)
        cr.set_line_width(1.25 / scale)
        cr.arc(self.cursor[0] + 0.5, self.cursor[1] + 0.5, self.brush.get_value(), 0, 2 * np.pi)
        cr.stroke()

    def locate(self, x, y):
        scale, ox, oy = self.box(self.canvas.get_width(), self.canvas.get_height())
        if scale <= 0:
            return None
        col, row = floor((x - ox) / scale), floor((y - oy) / scale)
        return (
            (col, row)
            if 0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1]
            else None
        )

    def stamp(self, col, row):
        if not self.masks:
            return
        self.cursor = (col, row)
        y, x = np.ogrid[: self.mask.shape[0], : self.mask.shape[1]]
        self.mask[(x - col) ** 2 + (y - row) ** 2 <= self.brush.get_value_as_int() ** 2] = (
            self.ink.get_selected() == 0
        )
        self.canvas.queue_draw()

    def begin(self, _, x, y):
        if not self.masks:
            return
        self.canvas.grab_focus()
        self.start = (x, y)
        self.last = self.locate(x, y)
        if self.last:
            self.stamp(*self.last)

    def move(self, _, dx, dy):
        if not self.start:
            return
        pos = self.locate(self.start[0] + dx, self.start[1] + dy)
        if pos:
            first = self.last or pos
            steps = max(abs(a - b) for a, b in zip(first, pos))
            for x, y in np.linspace(first, pos, max(1, steps) + 1):
                self.stamp(round(x), round(y))
        self.last = pos

    def end(self, gesture, dx, dy):
        self.move(gesture, dx, dy)
        self.start = self.last = None

    def key(self, _, key, *__):
        if not self.masks:
            return False
        if key == Gdk.KEY_space:
            self.stamp(*self.cursor)
            return True
        delta = {
            Gdk.KEY_Left: (-1, 0),
            Gdk.KEY_Right: (1, 0),
            Gdk.KEY_Up: (0, -1),
            Gdk.KEY_Down: (0, 1),
        }.get(key)
        if delta is None:
            return False
        self.cursor = tuple(
            max(0, min(v + d, n - 1))
            for v, d, n in zip(self.cursor, delta, self.mask.shape[::-1])
        )
        self.canvas.queue_draw()
        return True

    def apply_changes(self):
        if not self.part:
            return
        try:
            masks = parse_masks(
                [["".join("1" if c else "0" for c in r) for r in view] for view in self.masks]
            )
            rounded = self.rounded.get_active()
            if masks != self.part.masks or rounded != self.part.rounded:
                self.on_apply(replace(self.part, masks=masks, rounded=rounded))
        except ValueError as exc:
            self.on_error(str(exc))
