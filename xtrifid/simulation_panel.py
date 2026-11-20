"""GTK controls for a temporary fixed-step simulation and an explicit geometry bake."""

from time import monotonic

from .gtk import GLib, Gtk
from .simulation import FPS, KINDS, MAX_FRAME, Simulation


def _labelled_row(box, text, widget):
    row = Gtk.Box(spacing=6)
    title = Gtk.Label(label=text, xalign=0, hexpand=True)
    title.set_mnemonic_widget(widget)
    widget.update_property([Gtk.AccessibleProperty.LABEL], [text])
    row.append(title)
    row.append(widget)
    box.append(row)


class SimulationPanel(Gtk.Expander):
    """The owner receives transient world-space meshes, then a single baked frame."""

    def __init__(self, on_preview, on_bake, on_status):
        super().__init__(label="Физика")
        self.on_preview = on_preview
        self.on_bake = on_bake
        self.on_status = on_status
        self.sim = None
        self.part = self.source = None
        self.detail = "medium"
        self.colliders = ()
        self._signature = None
        self._timer = 0
        self.playing = False
        self._wanted = 0
        self._sync = False
        self._showing = False

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.set_child(box)
        self.kind = Gtk.DropDown.new_from_strings(
            ["Ткань", "Волосы", "Водная поверхность", "Объёмная жидкость"]
        )
        _labelled_row(box, "Среда", self.kind)
        self.wind = Gtk.SpinButton.new_with_range(-8, 8, 0.25)
        self.wind.set_digits(2)
        self.wind.set_value(1)
        _labelled_row(box, "Ветер, м/с²", self.wind)
        self.gravity = Gtk.SpinButton.new_with_range(0.1, 2, 0.1)
        self.gravity.set_digits(1)
        self.gravity.set_value(1)
        _labelled_row(box, "Гравитация, ×g", self.gravity)
        self.readout = Gtk.Label(xalign=0)
        box.append(self.readout)
        self.timeline = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, MAX_FRAME, 1)
        self.timeline.set_draw_value(False)
        self.timeline.update_property([Gtk.AccessibleProperty.LABEL], ["Кадр симуляции"])
        box.append(self.timeline)
        row = Gtk.Box(spacing=4)
        self.play = Gtk.Button(label="Воспроизвести", hexpand=True)
        self.first = Gtk.Button(label="В начало")
        row.append(self.play)
        row.append(self.first)
        box.append(row)
        self.bake = Gtk.Button(label="Запечь кадр в модель")
        box.append(self.bake)
        self.play.connect("clicked", self._toggle)
        self.first.connect("clicked", self._reset)
        self.bake.connect("clicked", self._bake)
        self.timeline.connect("value-changed", self._scrub)
        self.kind.connect("notify::selected", self._settings)
        self.wind.connect("value-changed", self._settings)
        self.gravity.connect("value-changed", self._settings)
        self.connect("notify::expanded", self._expanded)
        self.connect("unmap", lambda *_: self.stop())
        self._position(0)
        self.set_sensitive(False)

    def set_target(self, part, source_mesh, detail, colliders=()):
        colliders = tuple(colliders)
        signature = (
            id(part),
            id(source_mesh),
            detail,
            tuple((id(p), id(m)) for p, m in colliders),
        )
        if self._signature == signature:
            return
        self.stop()
        self._signature = signature
        self.part, self.source, self.detail, self.colliders = (
            part,
            source_mesh,
            detail,
            colliders,
        )
        self.set_sensitive(part is not None and source_mesh is not None)

    def _pause(self):
        self.playing = False
        if self._timer:
            GLib.source_remove(self._timer)
            self._timer = 0
        self.play.set_label("Воспроизвести")

    def stop(self):
        self._pause()
        self.sim = None
        self._wanted = 0
        self._position(0)
        if self._showing:
            self._showing = False
            self.on_preview(None, False)

    def _settings(self, *_):
        active = self.sim is not None
        self.stop()
        if active and self.get_expanded():
            self._reset()

    def _expanded(self, *_):
        if not self.get_expanded():
            self.stop()

    def _prepare(self):
        if self.sim is not None:
            return True
        if self.part is None or self.source is None:
            return False
        try:
            self.sim = Simulation(
                KINDS[self.kind.get_selected()],
                self.part,
                self.source,
                self.detail,
                self.colliders,
                wind=self.wind.get_value(),
                strength=self.gravity.get_value(),
            )
        except (ValueError, MemoryError) as exc:
            self.stop()
            self.on_status(str(exc) or "Недостаточно памяти для симуляции")
            return False
        return True

    def _position(self, frame):
        self._sync = True
        try:
            self.timeline.set_value(frame)
        finally:
            self._sync = False
        self.readout.set_label(f"Кадр {frame} / {MAX_FRAME}    {frame / FPS:.2f} с")

    def _show(self):
        mesh = self.sim.mesh()
        self._showing = True
        self.on_preview(mesh, self.sim.replace_selected)

    def _schedule(self):
        if not self._timer:
            self._timer = GLib.timeout_add(round(1000 / FPS), self._tick)

    def _toggle(self, *_):
        if self.playing:
            self._pause()
            return
        if self._prepare():
            if self.sim.frame == MAX_FRAME:
                self.sim.reset()
            self.playing = True
            self.play.set_label("Пауза")
            self._schedule()

    def _tick(self):
        session = self.sim
        try:
            if self.playing:
                self.sim.advance()
                self._wanted = self.sim.frame
                self._position(self.sim.frame)
            else:
                deadline = monotonic() + 0.008
                self.sim.seek(self._wanted, limit=1)
                while self.sim.frame != self._wanted and monotonic() < deadline:
                    self.sim.advance()
                self.readout.set_label(
                    f"Кадр {self.sim.frame} / {MAX_FRAME}    {self.sim.frame / FPS:.2f} с"
                )
            self._show()
        except (ValueError, MemoryError) as exc:
            self._timer = 0
            self.stop()
            self.on_status(str(exc) or "Недостаточно памяти для симуляции")
            return False
        if self.sim is not session:
            return False
        if self.sim.frame != MAX_FRAME and self.playing:
            return True
        if not self.playing and self.sim.frame != self._wanted:
            return True
        self.playing = False
        self._timer = 0
        self.play.set_label("Воспроизвести")
        return False

    def _scrub(self, widget):
        if self._sync:
            return
        self._pause()
        if self._prepare():
            self._wanted = round(widget.get_value())
            self._schedule()

    def _reset(self, *_):
        self._pause()
        if self._prepare():
            self.sim.reset()
            self._wanted = 0
            self._position(0)
            try:
                self._show()
            except (ValueError, MemoryError) as exc:
                self.stop()
                self.on_status(str(exc) or "Недостаточно памяти для симуляции")

    def _bake(self, *_):
        if not self._prepare():
            return
        # A scrub can still be advancing; baking always captures the visible frame.
        self._pause()
        try:
            self.on_bake(self.sim.name, self.sim.mesh(), self.sim.replace_selected)
        except (ValueError, MemoryError) as exc:
            self.on_status(str(exc) or "Недостаточно памяти для запекания")
            return
        self.stop()
