import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from test_model import spec

from xtrifid import resources
from xtrifid.model import Part, Scene
from xtrifid.planner import Planner, PlannerError


class PlannerHTTPCase(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.answers = []
        self.capacity = self.enterContext(
            patch.object(resources, "capacity", return_value=resources.Capacity(10 << 30, 8))
        )
        self.models = [
            {"name": name, "size": 128 << 20, "digest": name}
            for name in ("test:latest", "native-model:latest", "example:3b", "qwen3.5:4b")
        ]
        self.running_models = []
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                inventory = {"/api/tags": test.models, "/api/ps": test.running_models}
                test.assertIn(self.path, inventory)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"models": inventory[self.path]}).encode())

            def do_POST(self):
                test.assertEqual(self.path, "/api/chat")
                test.calls.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                answer = test.answers.pop(0)
                if callable(answer):
                    answer = answer()
                if answer is None:
                    self.close_connection = True
                    return
                status, answer = answer if isinstance(answer, tuple) else (200, answer)
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(answer).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = Planner(f"http://127.0.0.1:{self.server.server_port}", "test")

    def answer(self, data):
        self.answers.append({"done": True, "message": {"content": json.dumps(data)}})


class PlannerTests(PlannerHTTPCase):
    def test_schema_generation_and_targeted_edit(self):
        self.answer(
            {"parts": [spec() | {"support": 0}, spec("sphere") | dict(name="Шар", support=1)]}
        )
        scene = self.client.generate("Опора с шаром")
        self.assertEqual(len(scene.parts), 2)
        self.assertIn("parts", self.calls[0]["format"]["properties"])
        self.assertFalse(self.calls[0]["stream"])
        self.answer({"size": [1, 1, 1]})
        new = self.client.edit("Диаметр 1 метр", scene, 1)
        self.assertIsInstance(new, Part)
        self.assertEqual(new.size, (1, 1, 1))
        self.assertIn("Шар", self.calls[1]["messages"][-1]["content"])
        self.assertEqual(scene.parts[0], Part.parse(spec()))

    def test_decimal_echo_during_edit_keeps_existing_coordinate(self):
        scene = Scene.parse({"parts": [spec() | {"center": [0, 0, 0.4 + 0.02]}]})
        self.answer({"size": [2, 3, 8], "center": [0, 0, 0.42]})
        updated = self.client.edit("Измени только высоту", scene, 0)
        self.assertEqual(updated.center, scene.parts[0].center)
        self.assertEqual(updated.size, (2, 3, 8))

    def test_planned_profiles_require_outline_in_decode_schema(self):
        self.answer(
            {
                "parts": [
                    spec("lathe", outline=[[0.5, 0], [1, 0], [1, 1], [0.5, 1]]) | {"support": 0}
                ]
            }
        )
        self.client.generate("Полая втулка")
        shapes = self.calls[0]["format"]["properties"]["parts"]["items"]["oneOf"]
        curved = next(s for s in shapes if "lathe" in s["properties"]["shape"]["enum"])
        self.assertIn("outline", curved["required"])

    def test_invalid_model_data_repaired_once(self):
        self.answer({"parts": []})
        self.answer({"parts": [spec() | {"support": 0}]})
        self.assertEqual(len(self.client.generate("куб").parts), 1)
        self.assertEqual(len(self.calls), 2)
        self.answer({"parts": []})
        self.answer({"parts": []})
        with self.assertRaises(PlannerError):
            self.client.generate("куб")

    def test_contact_constraints_allow_forward_references_and_rotation(self):
        self.answer(
            {
                "parts": [
                    spec() | {"name": "Top", "support": 2},
                    spec() | {"rotation": [90, 0, 0], "support": 0},
                ]
            }
        )
        scene = self.client.generate("Столешница на повёрнутой опоре")
        self.assertAlmostEqual(scene.parts[1].center[2], 1.5)
        self.assertAlmostEqual(scene.parts[0].center[2], 5.0)

    def test_support_cycle_unknown_target_and_missing_support_are_rejected(self):
        for parts in ([spec() | {"support": 1}], [spec() | {"support": 2}], [spec()]):
            self.answer({"parts": parts})
            self.answer({"parts": parts})
            with self.assertRaises(PlannerError):
                self.client.generate("Стол")

    def test_empty_input_and_truncated_response(self):
        with self.assertRaises(ValueError):
            self.client.generate("   ")
        self.assertFalse(self.calls)
        self.answers.append(
            {"done": True, "done_reason": "length", "message": {"content": "{}"}}
        )
        with self.assertRaises(PlannerError):
            self.client.generate("стул")
        self.assertEqual(len(self.calls), 1)

    def test_failure_preserves_scene(self):
        scene = Scene.parse({"parts": [spec()]})
        self.answers.append({"error": "model missing"})
        with self.assertRaises(PlannerError):
            self.client.edit("выше", scene, 0)
        self.assertEqual(scene.parts, (Part.parse(spec()),))
