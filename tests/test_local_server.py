import json
import os
import select
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from xtrifid import resources
from xtrifid.local_server import LocalServer, pull_model
from xtrifid.planner import PlannerError


class LocalServerTests(unittest.TestCase):
    def test_simultaneous_requests_share_a_private_server_and_shutdown_reaps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / "ollama"
            pidfile = Path(tmp) / "pid"
            limitsfile = Path(tmp) / "limits"
            program.write_text(
                "#!/usr/bin/python3\n"
                "import json, os, subprocess, sys\n"
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
                "keys = ['OLLAMA_NUM_PARALLEL','OLLAMA_MAX_LOADED_MODELS','OLLAMA_MAX_QUEUE',"
                "'OLLAMA_KEEP_ALIVE','GOMAXPROCS']\n"
                "Path(__file__).with_name('limits').write_text(json.dumps("
                "[os.environ.get(k) for k in keys] + [len(os.sched_getaffinity(0))]))\n"
                "worker = subprocess.Popen([sys.executable, '-c', "
                "'import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print(1, flush=True); signal.pause()'], stdout=subprocess.PIPE)\n"
                "worker.stdout.readline()\n"
                "Path(__file__).with_name('worker').write_text(str(worker.pid))\n"
                "server.serve_forever()\n"
            )
            program.chmod(0o700)
            server = LocalServer(program)
            self.addCleanup(server.close)
            usable = min(4, len(os.sched_getaffinity(0)))
            self.enterContext(
                patch.object(
                    resources, "capacity", return_value=resources.Capacity(2 << 30, usable)
                )
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                urls = list(executor.map(lambda _: server.start(), range(2)))
            self.assertEqual(urls[0], urls[1])
            self.assertTrue(urls[0].startswith("http://127.0.0.1:"))
            pid = int(pidfile.read_text())
            threads = max(1, usable // 2)
            self.assertEqual(
                json.loads(limitsfile.read_text()), ["1", "1", "1", "0", str(threads), threads]
            )
            self.assertFalse(server.over_memory_limit(urls[0]))
            server.reclaim("http://127.0.0.1:1")
            os.kill(pid, 0)
            worker = int((Path(tmp) / "worker").read_text())
            worker_fd = os.pidfd_open(worker)
            self.addCleanup(os.close, worker_fd)
            exits = select.poll()
            exits.register(worker_fd, select.POLLIN)
            server.reclaim(urls[0])
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            self.assertTrue(exits.poll(2000))
            self.assertTrue(server.start().startswith("http://127.0.0.1:"))
            second = int(pidfile.read_text().splitlines()[-1])
            self.assertNotEqual(second, pid)
            server.close()
            with self.assertRaises(ProcessLookupError):
                os.kill(second, 0)
            with self.assertRaises(PlannerError):
                server.start()

    def test_low_headroom_prevents_server_start_and_pull(self):
        server = LocalServer(Path("/missing/xtrifid/ollama"))
        self.addCleanup(server.close)
        with patch.object(resources, "capacity", return_value=resources.Capacity(0, 1)):
            with self.assertRaisesRegex(PlannerError, "памят"):
                server.start()
            with self.assertRaisesRegex(PlannerError, "памят"):
                pull_model("http://127.0.0.1:1", "test", lambda text: None, Event())

    def test_missing_executable_is_reported(self):
        server = LocalServer(Path("/missing/xtrifid/ollama"))
        self.addCleanup(server.close)
        with self.assertRaisesRegex(PlannerError, "Ollama"):
            server.start()


class PullTests(unittest.TestCase):
    def setUp(self):
        self.capacity = self.enterContext(
            patch.object(resources, "capacity", return_value=resources.Capacity(8 << 30, 8))
        )
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

    def test_announced_layer_sizes_stop_an_unusable_custom_download(self):
        self.capacity.return_value = resources.Capacity(3 << 30, 2)
        self.frames = [
            b'{"status":"pulling a","digest":"a","total":1073741824,"completed":1}',
            b'{"status":"pulling a","digest":"a","total":1073741824,"completed":100}',
            b'{"status":"pulling b","digest":"b","total":1073741824,"completed":1}',
            b'{"status":"success"}',
        ]
        updates = []
        with self.assertRaisesRegex(PlannerError, "памят"):
            pull_model(self.url, "custom:large", updates.append, Event())
        self.assertGreater(len(updates), 0)
