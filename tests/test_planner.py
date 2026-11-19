import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from test_model import spec

from xtrifid.model import Part, Scene
from xtrifid.planner import Planner, PlannerError


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.answers = []
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                test.assertEqual(self.path, "/api/chat")
                test.calls.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                answer = test.answers.pop(0)
                self.send_response(200)
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

    def test_schema_generation_and_targeted_edit(self):
        self.answer({"parts": [spec(), spec("sphere") | dict(name="Шар")]})
        scene = self.client.generate("Опора с шаром")
        self.assertEqual(len(scene.parts), 2)
        self.assertIn("parts", self.calls[0]["format"]["properties"])
        self.assertFalse(self.calls[0]["stream"])
        self.answer(spec("sphere") | dict(size=[1, 1, 1], name="Шар"))
        new = self.client.edit("Диаметр 1 метр", scene, 1)
        self.assertIsInstance(new, Part)
        self.assertEqual(new.size, (1, 1, 1))
        self.assertIn("Шар", self.calls[1]["messages"][-1]["content"])
        self.assertEqual(scene.parts[0], Part.parse(spec()))

    def test_invalid_model_data_repaired_once(self):
        self.answer({"parts": []})
        self.answer({"parts": [spec()]})
        self.assertEqual(len(self.client.generate("куб").parts), 1)
        self.assertEqual(len(self.calls), 2)
        self.answer({"parts": []})
        self.answer({"parts": []})
        with self.assertRaises(PlannerError):
            self.client.generate("куб")

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
