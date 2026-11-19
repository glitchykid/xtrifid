import signal
import sys
from pathlib import Path

from . import __version__
from .gtk import Gio, GLib, Gtk
from .local_server import LocalServer
from .ui import Studio


class Application(Gtk.Application):
    def __init__(self):
        super().__init__(
            application_id="io.github.glitchykid.Xtrifid",
            flags=Gio.ApplicationFlags.HANDLES_OPEN,
        )
        self.add_main_option(
            "version",
            0,
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            "Show the application version",
            None,
        )
        bundle = getattr(sys, "_MEIPASS", None)
        self.server = LocalServer(Path(bundle) / "ollama/bin/ollama") if bundle else None

    def do_handle_local_options(self, options):
        if options.contains("version"):
            print(f"Xtrifid {__version__}")
            return 0
        return -1

    def do_activate(self):
        window = self.get_active_window() or Studio(self, self.server)
        window.present()

    def do_open(self, files, n_files, hint):
        for file in files:
            window = Studio(self, self.server)
            window.open_project(file)
            window.present()

    def do_shutdown(self):
        if self.server:
            self.server.close()
        Gtk.Application.do_shutdown(self)


def main():
    app = Application()

    def quit_app():
        app.quit()
        return GLib.SOURCE_CONTINUE

    handlers = [
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, quit_app)
        for sig in (signal.SIGINT, signal.SIGTERM)
    ]
    try:
        return app.run(sys.argv)
    finally:
        for handler in handlers:
            GLib.source_remove(handler)


if __name__ == "__main__":
    sys.exit(main())
