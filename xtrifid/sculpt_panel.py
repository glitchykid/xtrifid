"""Edit a named joint in the local coordinates of its construction recipe."""

from dataclasses import replace

from .gtk import Gtk
from .sculpt import MIN_RADIUS, parse_nodes


class SculptPanel(Gtk.Expander):
    def __init__(self, apply, error):
        super().__init__(label="Узлы формы")
        self.apply, self.error = apply, error
        self.part = None
        self.names = ()
        grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        self.set_child(grid)
        self.node = Gtk.DropDown.new_from_strings([])
        self.node.update_property([Gtk.AccessibleProperty.LABEL], ["Выберите узел формы"])
        grid.attach(self.node, 0, 0, 2, 1)
        self.fields = []
        for i, title in enumerate(("X · вправо", "Y · назад", "Z · вверх", "Радиус")):
            minimum = MIN_RADIUS if i == 3 else -1
            field = Gtk.SpinButton.new_with_range(minimum, 1, 0.01)
            field.set_digits(4)
            field.set_tooltip_text(f"Координаты рецепта: от {minimum:g} до 1")
            field.update_property([Gtk.AccessibleProperty.LABEL], [title])
            grid.attach(Gtk.Label(label=title, xalign=0), 0, i + 1, 1, 1)
            grid.attach(field, 1, i + 1, 1, 1)
            self.fields.append(field)
        button = Gtk.Button(label="Применить узел")
        button.connect("clicked", lambda *_: self.commit())
        grid.attach(button, 0, 5, 2, 1)
        self.node.connect("notify::selected", lambda *_: self.refresh())
        self.set_visible(False)

    def set_part(self, part, reset):
        self.part = part if part and part.shape == "sculpt" else None
        self.set_visible(self.part is not None)
        if self.part is None:
            return
        names = tuple(row[0] for row in self.part.nodes)
        if names != self.names:
            self.names = names
            self.node.set_model(Gtk.StringList.new(names))
        if reset:
            self.node.set_selected(0)
        self.refresh()

    def refresh(self):
        index = self.node.get_selected()
        if self.part is None or index >= len(self.part.nodes):
            return
        for field, value in zip(self.fields, self.part.nodes[index][2:]):
            field.set_value(value)
        self.old_text = tuple(field.get_text() for field in self.fields)

    def commit(self):
        index = self.node.get_selected()
        if self.part is None or index >= len(self.part.nodes):
            return
        nodes = list(self.part.nodes)
        row = list(nodes[index])
        for i, field in enumerate(self.fields):
            if field.get_text() != self.old_text[i]:
                field.update()
                row[i + 2] = field.get_value()
        if tuple(row) == nodes[index]:
            return
        nodes[index] = row
        try:
            self.apply(replace(self.part, nodes=parse_nodes(nodes)))
        except ValueError as exc:
            self.error(str(exc))
