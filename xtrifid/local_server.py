"""Run the packaged CPU backend on demand and stream explicit model downloads."""

import http.client
import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from . import resources
from .ollama import PlannerError

_direct_open = build_opener(ProxyHandler({})).open


class LocalServer:
    def __init__(self, executable: Path):
        self.executable = executable
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._process = None
        self._url = None
        self._memory_limit = 0

    def start(self):
        with self._lock:
            if self._closed.is_set():
                raise PlannerError("Студия закрывается")
            if self._process is None or self._process.poll() is not None:
                capacity = resources.capacity()
                if capacity.headroom < 256 * resources.MIB:
                    raise PlannerError("Недостаточно памяти для запуска встроенной Ollama.")
                try:
                    with socket.socket() as probe:
                        probe.bind(("127.0.0.1", 0))
                        address = f"127.0.0.1:{probe.getsockname()[1]}"
                    env = dict(
                        os.environ,
                        OLLAMA_HOST=address,
                        OLLAMA_NO_CLOUD="1",
                        OLLAMA_NOPRUNE="1",
                        OLLAMA_NUM_PARALLEL="1",
                        OLLAMA_MAX_LOADED_MODELS="1",
                        OLLAMA_MAX_QUEUE="1",
                        OLLAMA_KEEP_ALIVE="0",
                        OLLAMA_CONTEXT_LENGTH="4096",
                        OLLAMA_LOAD_TIMEOUT="2m",
                        OLLAMA_MAX_TRANSFER_STREAMS="1",
                        GOMAXPROCS=str(capacity.threads),
                        OMP_NUM_THREADS=str(capacity.threads),
                    )
                    # Linux affinity applies to this calling thread. The new Go service
                    # and all its workers inherit the mask before they can create threads.
                    affinity = os.sched_getaffinity(0)
                    os.sched_setaffinity(0, sorted(affinity)[: capacity.threads])
                    try:
                        self._process = subprocess.Popen(
                            [self.executable, "serve"],
                            env=env,
                            start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    finally:
                        os.sched_setaffinity(0, affinity)
                    self._memory_limit = capacity.headroom
                except OSError as exc:
                    self._stop()
                    raise PlannerError(
                        f"Не удалось запустить встроенную Ollama: {exc.strerror}"
                    ) from exc
                self._url = f"http://{address}"
            process, url = self._process, self._url

        deadline = time.monotonic() + 20
        while not self._closed.is_set() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise PlannerError(
                    f"Встроенная Ollama завершилась с кодом {process.returncode}"
                )
            try:
                with _direct_open(url + "/api/version", timeout=1) as reply:
                    info = json.loads(reply.read(4097))
                if isinstance(info, dict) and info.get("version") and process.poll() is None:
                    return url
            except (OSError, ValueError, http.client.HTTPException):
                pass
            self._closed.wait(0.1)
        with self._lock:
            if self._process is process:
                self._stop()
        raise PlannerError("Встроенная Ollama не успела запуститься")

    def _stop(self):
        proc = self._process
        self._process = None
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                # Retain ownership of the process-group ID until surviving workers
                # have also been signalled, even if the server exits first.
                if os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                    break
                time.sleep(0.03)
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def over_memory_limit(self, url):
        with self._lock:
            if self._url != url or self._process is None:
                return False
            try:
                return resources.process_tree_rss(self._process.pid) > self._memory_limit
            except (OSError, ValueError, IndexError):
                return True

    def reclaim(self, url):
        """Stop only our private process group and allow a smaller job to restart it."""
        with self._lock:
            if self._url == url:
                self._stop()

    def close(self):
        self._closed.set()
        with self._lock:
            self._stop()


def pull_model(url, model, on_progress, cancelled, *, opener=_direct_open, memory_limit=None):
    if cancelled.is_set():
        raise PlannerError("Загрузка отменена")
    if resources.capacity().headroom < 256 * resources.MIB:
        raise PlannerError("Недостаточно памяти для загрузки весов.")
    request = Request(
        url.rstrip("/") + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_message = None
    layers = {}
    try:
        with opener(request, timeout=300) as response:
            while not cancelled.is_set():
                raw = response.readline(65537)
                if not raw:
                    raise PlannerError("Ollama прервала загрузку. Попробуйте ещё раз.")
                if len(raw) > 65536:
                    raise PlannerError("Слишком большая строка в ответе Ollama")
                frame = json.loads(raw)
                if not isinstance(frame, dict):
                    raise ValueError
                if frame.get("error"):
                    raise PlannerError(f"Ошибка загрузки · {str(frame['error'])[:240]}")
                status = frame.get("status")
                if status == "success":
                    return
                if not isinstance(status, str):
                    raise ValueError
                done, total = frame.get("completed"), frame.get("total")
                digest = frame.get("digest")
                if type(total) is int and total > 0 and isinstance(digest, str):
                    if total > layers.get(digest, 0):
                        layers[digest] = total
                        headroom = resources.capacity().headroom
                        if memory_limit is not None:
                            headroom = min(headroom, memory_limit)
                        if (
                            len(layers) > 4096
                            or resources.model_working_bytes(sum(layers.values()), 2048)
                            > headroom
                        ):
                            raise PlannerError(
                                "Загрузка остановлена: для объявленного размера модели недостаточно памяти."
                            )
                message = f"Загрузка {model}…"
                if (
                    type(done) is int
                    and type(total) is int
                    and 0 <= done <= total
                    and total > 0
                ):
                    message = f"Загрузка {model} · {done * 100 // total}% файла"
                elif status == "verifying sha256 digest":
                    message = "Проверка загруженных весов…"
                if message != last_message:
                    on_progress(message)
                    last_message = message
        raise PlannerError("Загрузка отменена")
    except HTTPError as exc:
        raise PlannerError(f"Ошибка загрузки: Ollama вернула HTTP {exc.code}") from exc
    except (URLError, OSError, http.client.HTTPException) as exc:
        raise PlannerError("Соединение с Ollama прервалось. Повторите загрузку.") from exc
    except (ValueError, RecursionError) as exc:
        raise PlannerError("Некорректный ответ на запрос загрузки модели") from exc
