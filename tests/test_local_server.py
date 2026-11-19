import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread

from xtrifid.local_server import LocalServer, pull_model
from xtrifid.planner import PlannerError


class LocalServerTests(unittest.TestCase):
    def test_simultaneous_requests_share_a_private_server_and_shutdown_reaps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / "ollama"
            pidfile = Path(tmp) / "pid"
            program.write_text(
                "#!/usr/bin/python3\n"
                "import os, sys\n"
                "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:] == ['serve']\n"
                "host, port = os.environ['OLLAMA_HOST'].split(':')\n"
                "assert host == '127.0.0.1'\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                " def do_GET(self):\n"
                "  assert self.path == '/api/version'\n"
                "  self.send_response(200); self.end_headers()\n"
                '  self.wfile.write(b\'{"version": "test"}\')\n'
                "server = HTTPServer((host, int(port)), Handler)\n"
                "with Path(__file__).with_name('pid').open('a') as f: print(os.getpid(), file=f)\n"
                "server.serve_forever()\n"
            )
            program.chmod(0o700)
            server = LocalServer(program)
            self.addCleanup(server.close)
            with ThreadPoolExecutor(max_workers=2) as executor:
                urls = list(executor.map(lambda _: server.start(), range(2)))
            self.assertEqual(urls[0], urls[1])
            self.assertTrue(urls[0].startswith("http://127.0.0.1:"))
            pid = int(pidfile.read_text())
            server.close()
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            with self.assertRaises(PlannerError):
                server.start()

    def test_missing_executable_is_reported(self):
        server = LocalServer(Path("/missing/xtrifid/ollama"))
        self.addCleanup(server.close)
        with self.assertRaisesRegex(PlannerError, "Ollama"):
            server.start()


class PullTests(unittest.TestCase):
    def setUp(self):
        self.frames = []
        self.requests = []
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                test.requests.append((self.path, json.loads(body)))
                self.send_response(200)
                self.end_headers()
                for frame in test.frames:
                    self.wfile.write(frame + b"\n")

            def log_message(self, *args):
                pass

        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = Thread(target=http.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        self.url = f"http://127.0.0.1:{http.server_port}"

    def test_progress_and_explicit_completion(self):
        self.frames = [
            b'{"status":"pulling manifest"}',
            b'{"status":"pulling abc","total":100,"completed":50}',
            b'{"status":"success"}',
        ]
        updates = []
        pull_model(self.url, "qwen3.5:4b", updates.append, Event())
        self.assertEqual(
            self.requests, [("/api/pull", {"model": "qwen3.5:4b", "stream": True})]
        )
        self.assertTrue(any("50%" in text for text in updates))

    def test_partial_malformed_stream_and_server_error_fail(self):
        for stream in (
            [b'{"status":"pulling manifest"}'],
            [b"null"],
            [b'{"error":"disk full"}'],
            [b"x" * 66560],
        ):
            with self.subTest(stream=stream[0][:28]):
                self.frames = stream
                with self.assertRaises(PlannerError):
                    pull_model(self.url, "test", lambda text: None, Event())

    def test_cancelled_download_does_not_issue_a_request(self):
        cancelled = Event()
        cancelled.set()
        with self.assertRaises(PlannerError):
            pull_model(self.url, "test", lambda text: None, cancelled)
        self.assertFalse(self.requests)
