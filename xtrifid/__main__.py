import sys

from .ui import Gtk, Studio


def main():
    app = Gtk.Application(application_id="io.github.glitchykid.Xtrifid")

    def activate(app):
        window = app.get_active_window() or Studio(app)
        window.present()

    app.connect("activate", activate)
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
