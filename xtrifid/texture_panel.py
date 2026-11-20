"""GTK controls for a selected part's procedural diffuse colour map."""

from secrets import randbelow

from .gtk import Gdk, Gtk
from .textures import MOTIFS, PALETTES, RESOLUTIONS, TextureSpec, texture_from_prompt


class TexturePanel(Gtk.Expander):
    def __init__(self, on_apply_texture_spec_or_none, on_status, on_ai=None):
        super().__init__(label="Текстура детали")
        self.apply, self.status, self.on_ai = on_apply_texture_spec_or_none, on_status, on_ai
        grid = Gtk.Grid(row_spacing=4, column_spacing=5)
        self.set_child(grid)
        self.material = Gtk.DropDown.new_from_strings(
            ["Дерево", "Камень", "Ткань", "Кожа", "Металл", "Гладкое покрытие"]
        )
        self.resolution = Gtk.DropDown.new_from_strings([f"{n} × {n}" for n in RESOLUTIONS])
        self.resolution.set_selected(1)
        self.seed = Gtk.SpinButton.new_with_range(0, 2**31 - 1, 1)
        self.scale = Gtk.SpinButton.new_with_range(0.25, 8, 0.05)
        self.scale.set_digits(2)
        self.motif = Gtk.DropDown.new_from_strings(["Без узора", "Пятна", "Полосы", "Чешуя"])
        self.colors = []
        palette = Gtk.Box(spacing=4)
        for name in ("Первый цвет", "Второй цвет", "Цвет узора"):
            dialog = Gtk.ColorDialog(title=name, with_alpha=False)
            color = Gtk.ColorDialogButton.new(dialog)
            color.set_tooltip_text(name)
            color.update_property([Gtk.AccessibleProperty.LABEL], [name])
            palette.append(color)
            self.colors.append(color)
        for row, (label, widget) in enumerate(
            (
                ("Материал", self.material),
                ("Узор", self.motif),
                ("Цвета", palette),
                ("Разрешение", self.resolution),
                ("Зерно", self.seed),
                ("Частота узора", self.scale),
            )
        ):
            grid.attach(Gtk.Label(label=label, xalign=0), 0, row, 1, 1)
            if widget != palette:
                widget.update_property([Gtk.AccessibleProperty.LABEL], [label])
            grid.attach(widget, 1, row, 1, 1)
        self.prompt = Gtk.Entry(
            placeholder_text="Например, синяя ткань", max_length=500, hexpand=True
        )
        self.prompt.update_property([Gtk.AccessibleProperty.LABEL], ["Описание текстуры"])
        grid.attach(self.prompt, 0, 6, 2, 1)
        buttons = Gtk.Box(spacing=4)
        grid.attach(buttons, 0, 7, 2, 1)
        self.apply_button = Gtk.Button(label="Назначить")
        self.apply_button.connect("clicked", lambda *_: self.commit())
        buttons.append(self.apply_button)
        self.remove_button = Gtk.Button(label="Убрать")
        self.remove_button.connect("clicked", lambda *_: self.apply(None))
        buttons.append(self.remove_button)
        self.reroll = Gtk.Button(label="Новый вариант")
        self.reroll.connect("clicked", lambda *_: self.new_seed())
        buttons.append(self.reroll)
        self.ai_button = Gtk.Button(label="Создать по описанию (ИИ)")
        self.ai_button.set_visible(on_ai is not None)
        self.ai_button.connect("clicked", lambda *_: self.request_ai())
        grid.attach(self.ai_button, 0, 8, 2, 1)
        self.prompt.connect(
            "activate", lambda *_: self.request_ai() if self.on_ai else self.commit()
        )
        self.material.connect(
            "notify::selected",
            lambda *_: self._colors(
                (*PALETTES[self._material()], PALETTES[self._material()][0])
            ),
        )
        self.set_part(None)

    def _material(self):
        return tuple(PALETTES)[self.material.get_selected()]

    def _colors(self, palette):
        for button, value in zip(self.colors, palette):
            color = Gdk.RGBA()
            color.parse(value)
            button.set_rgba(color)

    def set_part(self, part):
        self.set_sensitive(part is not None)
        self.prompt.set_text("")
        applied = part.texture if part else None
        self.remove_button.set_sensitive(applied is not None)
        spec = applied or TextureSpec.parse({"material": "wood"})
        self.material.set_selected(tuple(PALETTES).index(spec.material))
        self._colors((spec.color_a, spec.color_b, spec.accent or spec.color_a))
        self.motif.set_selected(MOTIFS.index(spec.motif))
        self.seed.set_value(spec.seed)
        self.scale.set_value(spec.scale)
        self.resolution.set_selected(RESOLUTIONS.index(spec.resolution))

    def commit(self):
        try:
            self.seed.update()
            self.scale.update()
            seed = self.seed.get_value_as_int()
            resolution = RESOLUTIONS[self.resolution.get_selected()]
            prompt = self.prompt.get_text().strip()
            if prompt:
                spec = texture_from_prompt(prompt, seed=seed, resolution=resolution)
            else:
                palette = []
                for button in self.colors:
                    color = button.get_rgba()
                    palette.append(
                        "#"
                        + "".join(
                            f"{round(c * 255):02x}"
                            for c in (color.red, color.green, color.blue)
                        )
                    )
                spec = TextureSpec(
                    self._material(),
                    seed,
                    resolution,
                    palette[0],
                    palette[1],
                    self.scale.get_value(),
                    MOTIFS[self.motif.get_selected()],
                    palette[2],
                )
            self.apply(spec)
        except ValueError as exc:
            self.status(str(exc))

    def new_seed(self):
        self.seed.set_value(randbelow(2**31))
        self.commit()

    def request_ai(self):
        text = self.prompt.get_text().strip()
        if not text:
            self.status("Опишите цвет, материал или желаемый рисунок")
            self.prompt.grab_focus()
            return
        self.seed.update()
        if self.on_ai:
            self.on_ai(
                text, self.seed.get_value_as_int(), RESOLUTIONS[self.resolution.get_selected()]
            )
