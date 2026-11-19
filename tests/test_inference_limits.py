import json
import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, patch

from test_model import spec
from test_planner import PlannerHTTPCase

from xtrifid import ollama, resources
from xtrifid.planner import ModelMissing, Planner, PlannerError


class InferenceBudgetTests(PlannerHTTPCase):
    def request(self):
        self.answer({"parts": [spec() | {"support": 0}]})
        return self.client.generate("Куб")

    def test_all_routes_check_size_before_loading_and_bound_request_options(self):
        self.capacity.return_value = resources.Capacity(3 << 30, 4)
        self.models[0]["size"] = 30 << 30
        with self.assertRaisesRegex(PlannerError, "памят"):
            self.request()
        self.assertFalse(self.calls)
        self.models[0]["size"] = 128 << 20
        self.request()
        payload = self.calls[0]
        self.assertEqual(payload["options"]["num_thread"], 2)
        self.assertLessEqual(payload["options"]["num_ctx"], 4096)
        self.assertLessEqual(payload["options"]["num_batch"], 128)
        self.assertLess(payload["options"]["num_predict"], payload["options"]["num_ctx"])
        self.assertEqual(payload["keep_alive"], 0)

    def test_readable_matching_tag_and_valid_positive_size_are_required(self):
        for invalid in (0, -1, True, None, "unknown"):
            self.models[0]["size"] = invalid
            with self.subTest(size=invalid), self.assertRaisesRegex(PlannerError, "размер"):
                self.request()
        self.models.clear()
        with self.assertRaises(ModelMissing) as exc:
            self.request()
        self.assertEqual(exc.exception.url, self.client.url)
        self.assertFalse(self.calls)

    def test_loaded_weights_with_matching_context_are_not_charged_twice(self):
        self.models[0]["size"] = 3 << 30
        self.capacity.return_value = resources.Capacity(2 << 30, 4)
        model = self.models[0] | {"size_vram": 0, "context_length": 8192}
        self.running_models = [model | {"digest": "some other revision"}]
        with self.assertRaises(PlannerError):
            self.request()
        self.running_models = [model]
        self.request()
        self.assertEqual(len(self.calls), 1)

    def test_model_sent_is_the_one_measured_even_for_qualified_custom_tags(self):
        self.models = [{"name": "registry.ollama.ai/library/custom:small", "size": 256 << 20}]
        self.answers.append({"done": True})
        ollama.post_chat(
            self.client.url, "custom:small", json.dumps({"model": "very-large"}).encode()
        )
        self.assertEqual(self.calls[0]["model"], "custom:small")

    def test_unknown_inventory_and_partial_running_records_do_not_start_a_large_job(self):
        self.models = None
        with self.assertRaisesRegex(PlannerError, "неполные"):
            self.request()
        self.models = [{"name": "test", "size": 7 << 30, "digest": "abc"}]
        self.running_models = [{"name": "test", "digest": "abc", "size": 7 << 30}]
        self.capacity.return_value = resources.Capacity(2 << 30, 6)
        with self.assertRaisesRegex(PlannerError, "памят"):
            self.request()
        self.assertFalse(self.calls)

    def test_discovered_docker_honours_container_memory_even_on_a_larger_workstation(self):
        unused = socket.socket()
        self.addCleanup(unused.close)
        unused.bind(("127.0.0.1", 0))
        self.models[0]["size"] = 1 << 30
        with (
            patch.object(ollama, "DEFAULT_URL", f"http://127.0.0.1:{unused.getsockname()[1]}"),
            patch.object(ollama, "docker_url", return_value=self.client.url),
            patch.object(
                ollama, "_inspect_service", return_value="other 1234 536870912 1000000000"
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            with self.assertRaisesRegex(PlannerError, "контейнер"):
                Planner(model="test").generate("Куб")
        self.assertFalse(self.calls)

    def test_private_rss_guard_reclaims_only_the_owner_of_the_current_request(self):
        started, release = Event(), Event()
        owner = Mock()
        owner.start.return_value = self.client.url
        owner.over_memory_limit.side_effect = lambda url: started.is_set()
        unused = socket.socket()
        self.addCleanup(unused.close)
        unused.bind(("127.0.0.1", 0))

        def wait_for_stop():
            started.set()
            release.wait(5)
            return None

        self.answers.append(wait_for_stop)
        with (
            patch.object(ollama, "WATCH_INTERVAL", 0.02),
            patch.object(ollama, "DEFAULT_URL", f"http://127.0.0.1:{unused.getsockname()[1]}"),
            patch.object(ollama, "docker_url", return_value=None),
            patch.dict("os.environ", {}, clear=True),
        ):
            try:
                with self.assertRaisesRegex(PlannerError, "памят"):
                    Planner(local_server=owner, model="test").generate("Куб")
                owner.reclaim.assert_called_once_with(self.client.url)
            finally:
                release.set()

    def test_lost_docker_inspect_prevents_a_load_with_unknown_container_limit(self):
        unused = socket.socket()
        self.addCleanup(unused.close)
        unused.bind(("127.0.0.1", 0))
        self.answers.append({"done": True})
        with (
            patch.object(ollama, "DEFAULT_URL", f"http://127.0.0.1:{unused.getsockname()[1]}"),
            patch.object(ollama, "docker_url", return_value=self.client.url),
            patch.object(ollama, "_inspect_service", side_effect=OSError),
        ):
            with self.assertRaisesRegex(PlannerError, "контейнер"):
                ollama.post_chat(None, "test", b"{}")
        self.assertFalse(self.calls)

    def test_wall_clock_limit_releases_the_active_slot_on_a_stalled_response(self):
        release = Event()

        def stalled():
            release.wait(3)
            return None

        self.answers.append(stalled)
        try:
            with (
                patch.object(ollama, "CHAT_TIMEOUT", 0.03),
                patch.object(ollama, "WATCH_INTERVAL", 0.02),
                self.assertRaisesRegex(PlannerError, "время"),
            ):
                ollama.post_chat(self.client.url, "test", b"{}")
        finally:
            release.set()
        self.answers.append({"done": True})
        ollama.post_chat(self.client.url, "test", b"{}")
        self.assertEqual(len(self.calls), 2)

    def test_low_memory_aborts_active_request_without_starting_a_second_generation(self):
        started, release = Event(), Event()

        def wait_for_cancel():
            started.set()
            release.wait(5)
            return None

        self.answers.append(wait_for_cancel)
        with patch.object(ollama, "WATCH_INTERVAL", 0.01), ThreadPoolExecutor() as worker:
            first = worker.submit(self.client.generate, "Куб")
            try:
                self.assertTrue(started.wait(3))
                with self.assertRaisesRegex(PlannerError, "уже"):
                    self.request()
                self.capacity.return_value = resources.Capacity(0, 1)
                with self.assertRaisesRegex(PlannerError, "памят"):
                    first.result(timeout=3)
            finally:
                release.set()
        self.assertEqual(len(self.calls), 1)
