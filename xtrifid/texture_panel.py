"""GTK controls for a selected part's procedural surface."""

from dataclasses import replace
from secrets import randbelow

from .gtk import Gdk, Gtk
from .textures import (
    MOTIFS,
    PALETTES,
    RESOLUTIONS,
    PBRMaterial,
    TextureSpec,
    pbr_preset,
    texture_from_prompt,
)


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
        grid.attach(self.prompt, 0, 8, 2, 1)
        buttons = Gtk.Box(spacing=4)
        grid.attach(buttons, 0, 9, 2, 1)
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
        grid.attach(self.ai_button, 0, 10, 2, 1)
        self.physical = Gtk.CheckButton(label="PBR-материал")
        self.physical.set_tooltip_text("Добавить карты нормалей, шероховатости и металличности")
        grid.attach(self.physical, 0, 6, 2, 1)
        self.pbr_grid = Gtk.Grid(row_spacing=3, column_spacing=5)
        grid.attach(self.pbr_grid, 0, 7, 2, 1)
        self.roughness, self.metallic, self.normal_strength = (
            Gtk.SpinButton.new_with_range(0, upper, 0.01) for upper in (1, 1, 2)
        )
        for row, (name, spin, tip) in enumerate(
            (
                ("Шероховатость", self.roughness, "0 — полированная поверхность, 1 — матовая"),
                ("Металличность", self.metallic, "0 — диэлектрик, 1 — голый металл"),
                (
                    "Сила рельефа",
                    self.normal_strength,
                    "0 — плоская карта нормалей, 1 — исходный рельеф",
                ),
            )
        ):
            spin.set_digits(2)
            spin.set_numeric(True)
            spin.set_tooltip_text(tip)
            name_label = Gtk.Label(label=name, xalign=0, mnemonic_widget=spin)
            spin.update_property([Gtk.AccessibleProperty.LABEL], [name])
            self.pbr_grid.attach(name_label, 0, row, 1, 1)
            self.pbr_grid.attach(spin, 1, row, 1, 1)
        self.physical.connect("toggled", self._show_physical)
        self.prompt.connect(
            "activate", lambda *_: self.request_ai() if self.on_ai else self.commit()
        )
        self.material.connect("notify::selected", self._preset)
        self.set_part(None)

    def _material(self):
        return tuple(PALETTES)[self.material.get_selected()]

    def _colors(self, palette):
        for button, value in zip(self.colors, palette):
            color = Gdk.RGBA()
            color.parse(value)
            button.set_rgba(color)

    def _preset(self, *_):
        material = self._material()
        self._colors((*PALETTES[material], PALETTES[material][0]))
        self._set_physical_values(pbr_preset(material))

    def _set_physical_values(self, material):
        self.roughness.set_value(material.roughness)
        self.metallic.set_value(material.metallic)
        self.normal_strength.set_value(material.normal_strength)

    def _show_physical(self, *_):
        active = self.physical.get_active()
        self.pbr_grid.set_visible(active)
        self.pbr_grid.set_sensitive(active)

    def set_part(self, part, mode="render"):
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
        self._set_physical_values(spec.pbr or pbr_preset(spec.material))
        self.physical.set_active(spec.pbr is not None if applied else mode == "game")
        self._show_physical()

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
            if self.physical.get_active():
                selected_defaults = pbr_preset(self._material())
                recipe_defaults = pbr_preset(spec.material)
                values = []
                for name in ("roughness", "metallic", "normal_strength"):
                    spin = getattr(self, name)
                    spin.update()
                    value = spin.get_value()
                    if prompt and abs(value - getattr(selected_defaults, name)) < 1e-6:
                        value = getattr(recipe_defaults, name)
                    values.append(value)
                spec = replace(spec, pbr=PBRMaterial(*values))
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
                text,
                self.seed.get_value_as_int(),
                RESOLUTIONS[self.resolution.get_selected()],
                physical=self.physical.get_active(),
            )
