import os
import selectors
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_model import spec

from xtrifid.gtk import Gio
from xtrifid.model import Scene
from xtrifid.storage import save_project


class DesktopTests(unittest.TestCase):
    def test_version_works_without_a_graphical_session(self):
        result = subprocess.run(
            [sys.executable, "-m", "xtrifid", "--version"],
            capture_output=True,
            text=True,
            env=dict(os.environ, DISPLAY="", WAYLAND_DISPLAY=""),
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r"^Xtrifid \d+\.\d+\.\d+$")

    @unittest.skipUnless(os.getenv("DISPLAY"), "requires X11")
    def test_sigterm_shuts_down_the_owned_backend(self):
        runner = """
import xtrifid.__main__ as desktop
class Backend:
    def close(self):
        print("BACKEND CLOSED", flush=True)
class App(desktop.Application):
    def do_activate(self):
        self.server = Backend()
        self.hold()
        print("READY", flush=True)
desktop.Application = App
raise SystemExit(desktop.main())
"""
        with subprocess.Popen([sys.executable, "-c", runner], stdout=subprocess.PIPE) as proc:
            try:
                with selectors.DefaultSelector() as sel:
                    sel.register(proc.stdout, selectors.EVENT_READ)
                    self.assertTrue(sel.select(10), "Application did not start")
                self.assertEqual(proc.stdout.readline().strip(), b"READY")
                proc.terminate()
                stdout, _ = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 0)
                self.assertIn(b"BACKEND CLOSED", stdout)
            finally:
                if proc.poll() is None:
                    proc.kill()

    @unittest.skipUnless(os.getenv("DISPLAY"), "requires X11")
    def test_open_files_through_gio_preserves_existing_windows(self):
        from xtrifid.__main__ import Application

        app = Application()
        app.set_application_id("io.github.glitchykid.Xtrifid." + self._testMethodName)
        app.set_flags(app.get_flags() | Gio.ApplicationFlags.NON_UNIQUE)
        app.register(None)
        self.addCleanup(lambda: [w.close() for w in app.get_windows()])
        app.activate()
        first = app.get_windows()[0]
        existing = Scene.parse({"parts": [spec()]})
        first.install(existing, "game")
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "Моя втулка.xtri"
            opened = Scene.parse({"parts": [spec("tube", inner_diameter=1)]})
            save_project(good, opened, "render")
            app.open(
                [
                    Gio.File.new_for_path(str(good)),
                    Gio.File.new_for_uri("https://invalid.invalid/a.xtri"),
                ],
                "",
            )
        self.assertEqual(first.doc.scene, existing)
        self.assertEqual(len(app.get_windows()), 3)
        restored = next(w for w in app.get_windows() if w.doc.scene == opened)
        self.assertEqual(restored.doc.mode, "render")
        self.assertIn(good.name, restored.status.get_text())
        self.assertTrue(any("локальный" in w.status.get_text() for w in app.get_windows()))

    @unittest.skipUnless(os.getenv("DISPLAY"), "requires X11")
    def test_offer_model_download_only_after_missing_model_and_explicit_click(self):
        from unittest.mock import patch

        from test_studio import until

        from xtrifid.__main__ import Application
        from xtrifid.planner import ModelMissing, PlannerError

        app = Application()
        app.set_application_id("io.github.glitchykid.Xtrifid." + self._testMethodName)
        app.set_flags(app.get_flags() | Gio.ApplicationFlags.NON_UNIQUE)
        app.register(None)
        app.activate()
        window = app.get_windows()[0]
        self.addCleanup(window.close)
        original = Scene.parse({"parts": [spec()]})
        window.install(original, "game")
        window.prompt.set_text("Втулка")
        missing = ModelMissing("http://127.0.0.1:11434", "qwen3.5:4b", None)
        with (
            patch.object(window.planner, "generate", side_effect=missing),
            patch("xtrifid.ui.pull_model") as pull,
        ):
            window.generate.emit("clicked")
            until(lambda: not window.busy)
            self.assertTrue(window.download.get_visible())
            self.assertIn("3,3", window.status.get_text())
            pull.assert_not_called()
            pull.side_effect = PlannerError("Загрузка прервалась")
            window.download.emit("clicked")
            until(lambda: not window.busy)
            self.assertTrue(window.download.get_sensitive())
            self.assertIn("прервалась", window.status.get_text())
            pull.side_effect = None
            window.download.emit("clicked")
            until(lambda: not window.busy)
            self.assertFalse(window.download.get_visible())
            self.assertIn("загружена", window.status.get_text())
        self.assertEqual(window.doc.scene, original)


if __name__ == "__main__":
    unittest.main()
