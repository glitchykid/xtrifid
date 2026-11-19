import os
import socket
import subprocess
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request

from test_model import spec
from test_planner import PlannerHTTPCase

from xtrifid import ollama
from xtrifid.planner import ModelMissing, Planner, PlannerError


class ConnectionTests(PlannerHTTPCase):
    def setUp(self):
        super().setUp()
        self.service = self.client.url
        # A bound, non-listening port reliably refuses connections.
        unused = socket.socket()
        unused.bind(("127.0.0.1", 0))
        self.addCleanup(unused.close)
        self.unavailable = f"http://127.0.0.1:{unused.getsockname()[1]}"
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch.object(ollama, "DEFAULT_URL", self.unavailable))
        self.bridge = self.enterContext(patch.object(ollama, "docker_url", return_value=None))
        self.enterContext(
            patch.object(
                ollama, "_inspect_service", return_value="mock 1 6442450944 4000000000"
            )
        )
        self.managed = Mock()
        self.managed.start.return_value = self.service

    def test_native_server_has_priority_and_ignores_proxy(self):
        with patch.object(ollama, "DEFAULT_URL", self.service):
            os.environ["http_proxy"] = self.unavailable
            os.environ["XTRIFID_MODEL"] = "native-model"
            self.answer({"parts": [spec() | {"support": 0}]})
            Planner(local_server=self.managed).generate("Куб")
        self.assertEqual(self.calls[0]["model"], "native-model")
        self.bridge.assert_not_called()
        self.managed.start.assert_not_called()

    def test_explicit_urls_remain_authoritative_even_if_refused(self):
        os.environ["OLLAMA_URL"] = self.unavailable
        for planner in (Planner(local_server=self.managed), Planner(self.unavailable)):
            with self.assertRaisesRegex(PlannerError, "OLLAMA_URL"):
                planner.generate("Куб")
        self.answer({"parts": [spec() | {"support": 0}]})
        Planner(self.service).generate("Куб")
        self.bridge.assert_not_called()
        self.managed.start.assert_not_called()

    def test_bridge_handles_request_after_native_refusal(self):
        self.bridge.return_value = self.service
        os.environ["http_proxy"] = self.unavailable
        self.answer({"parts": [spec() | {"support": 0}]})
        scene = Planner(local_server=self.managed).generate("Куб")
        self.assertEqual(scene.parts[0].shape, "box")
        self.assertIn("parts", self.calls[0]["format"]["properties"])
        self.managed.start.assert_not_called()

    def test_absent_or_refused_bridge_allows_managed_service(self):
        client = Planner(local_server=self.managed)
        for bridge in (None, self.unavailable):
            with self.subTest(bridge=bridge):
                self.bridge.return_value = bridge
                self.answer({"parts": [spec() | {"support": 0}]})
                client.generate("Куб")
        self.assertEqual(self.managed.start.call_count, 2)
        self.assertEqual(len(self.calls), 2)

    def test_unroutable_bridge_allows_managed_service_before_any_request(self):
        self.bridge.return_value = "http://172.18.0.2:11434"
        connect = ollama.http.client.HTTPConnection.connect

        def route(connection):
            if connection.host == "172.18.0.2":
                raise TimeoutError("no route from this host")
            return connect(connection)

        self.answer({"parts": [spec() | {"support": 0}]})
        with patch.object(ollama.http.client.HTTPConnection, "connect", route):
            Planner(local_server=self.managed).generate("Куб")
        self.managed.start.assert_called_once()
        self.assertEqual(len(self.calls), 1)

    def test_ambiguous_disconnection_does_not_replay(self):
        self.answers.append(None)
        with patch.object(ollama, "DEFAULT_URL", self.service), self.assertRaises(PlannerError):
            Planner(local_server=self.managed).generate("Куб")
        self.assertEqual(len(self.calls), 1)
        self.bridge.assert_not_called()
        self.managed.start.assert_not_called()

    def test_only_model_not_found_reports_download_target(self):
        self.bridge.return_value = self.service
        planner = Planner(model="example:3b", local_server=self.managed)
        self.answers.append((404, {"error": "model 'example:3b' not found"}))
        with self.assertRaises(ModelMissing) as missing:
            planner.generate("Куб")
        self.assertEqual(
            (missing.exception.url, missing.exception.model), (self.service, "example:3b")
        )
        self.answers.append({"done": True})
        os.environ["http_proxy"] = self.unavailable
        with missing.exception.opener(
            Request(self.service + "/api/chat", data=b"{}"), timeout=3
        ) as response:
            self.assertEqual(response.status, 200)
        for status, reason in (
            (404, "route not found"),
            (404, "model 'other' not found"),
            (503, "model 'example:3b' not found"),
        ):
            self.answers.append((status, {"error": reason}))
            with self.assertRaisesRegex(PlannerError, str(status)) as failure:
                planner.generate("Куб")
            self.assertNotIsInstance(failure.exception, ModelMissing)
        self.managed.start.assert_not_called()

    def test_no_automatic_backend_has_actionable_error(self):
        with self.assertRaisesRegex(PlannerError, "ollama serve"):
            Planner().generate("Куб")


class DockerAddressTests(unittest.TestCase):
    def test_uses_named_service_on_local_docker_and_private_bridge_address(self):
        with patch("subprocess.run") as inspect:
            inspect.return_value.stdout = "172.18.0.2\n"
            self.assertEqual(ollama.docker_url(), "http://172.18.0.2:11434")
            args = inspect.call_args.args[0]
            self.assertIn("unix:///var/run/docker.sock", args)
            self.assertIn("xtrifid-ollama-1", args)
            self.assertTrue(any("xtrifid_default" in arg for arg in args))
            for value in ("", "8.8.8.8", "127.0.0.1", "172.18.0.2/path"):
                inspect.return_value.stdout = value
                self.assertIsNone(ollama.docker_url())
            for error in (
                FileNotFoundError(),
                subprocess.CalledProcessError(1, ["docker"]),
                subprocess.TimeoutExpired(["docker"], 3),
            ):
                inspect.side_effect = error
                self.assertIsNone(ollama.docker_url())
