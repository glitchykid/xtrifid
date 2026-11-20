"""Edit a single elliptical cross-section in the part's local metric coordinates."""

from math import tau

import numpy as np

from .geometry import radial_segments, rotation_matrix
from .gtk import Gtk
from .model import MAX_LENGTH, MAX_SECTIONS
from .sections import local_sections, move_section


class LoftEditor(Gtk.Expander):
    def __init__(self, apply, error, changed):
        super().__init__(label="Поперечные сечения")
        self.apply, self.error, self.changed = apply, error, changed
        self.part = None
        grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        self.set_child(grid)
        self.number = Gtk.SpinButton.new_with_range(1, MAX_SECTIONS, 1)
        self.fields = []
        titles = (
            "Сечение",
            "Высота Z, м",
            "Радиус X, м",
            "Радиус Y, м",
            "Центр X, м",
            "Центр Y, м",
        )
        for i, title in enumerate(titles):
            field = (
                self.number
                if i == 0
                else Gtk.SpinButton.new_with_range(
                    0 if i in (2, 3) else -MAX_LENGTH, MAX_LENGTH, 0.001
                )
            )
            if i:
                field.set_digits(6)
                self.fields.append(field)
            field.update_property([Gtk.AccessibleProperty.LABEL], [title])
            grid.attach(Gtk.Label(label=title, xalign=0), 0, i, 1, 1)
            grid.attach(field, 1, i, 1, 1)
        apply_button = Gtk.Button(label="Применить сечение")
        apply_button.connect("clicked", lambda *_: self.commit())
        grid.attach(apply_button, 0, len(titles), 2, 1)
        self.number.connect("value-changed", lambda *_: self.refresh())
        self.connect("notify::expanded", lambda *_: self.changed())

    def set_part(self, part, reset):
        self.part = part if part and part.shape == "loft" else None
        self.set_visible(self.part is not None)
        if self.part:
            self.number.set_range(1, len(part.sections))
            if reset:
                self.number.set_value(1)
            self.refresh()

    def refresh(self):
        if self.part:
            row = local_sections(self.part)[self.number.get_value_as_int() - 1]
            for field, value in zip(self.fields, row):
                field.set_value(value)
            self.text = [f.get_text() for f in self.fields]
            self.changed()

    def guide(self, mode, detail):
        z, rx, ry, x, y = local_sections(self.part)[self.number.get_value_as_int() - 1]
        angles = np.linspace(0, tau, radial_segments(mode, detail) + 1) if rx else [0]
        points = np.column_stack(
            (x + rx * np.sin(angles), y - ry * np.cos(angles), np.full(len(angles), z))
        )
        return points @ rotation_matrix(self.part.rotation).T + self.part.center

    def commit(self):
        index = self.number.get_value_as_int() - 1
        values = local_sections(self.part)[index]
        updated = False
        for i, field in enumerate(self.fields):
            if field.get_text() != self.text[i]:
                field.update()
                values[i] = field.get_value()
                updated = True
        if updated:
            try:
                self.apply(move_section(self.part, index, values))
            except ValueError as exc:
                self.error(str(exc))
