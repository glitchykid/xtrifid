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

from .ollama import PlannerError

_direct_open = build_opener(ProxyHandler({})).open


class LocalServer:
    def __init__(self, executable: Path):
        self.executable = executable
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._process = None
        self._url = None

    def start(self):
        with self._lock:
            if self._closed.is_set():
                raise PlannerError("Студия закрывается")
            if self._process is None or self._process.poll() is not None:
                try:
                    with socket.socket() as probe:
                        probe.bind(("127.0.0.1", 0))
                        address = f"127.0.0.1:{probe.getsockname()[1]}"
                    env = dict(
                        os.environ, OLLAMA_HOST=address, OLLAMA_NO_CLOUD="1", OLLAMA_NOPRUNE="1"
                    )
                    self._process = subprocess.Popen(
                        [self.executable, "serve"],
                        env=env,
                        start_new_session=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError as exc:
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
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        except ProcessLookupError:
            proc.wait()

    def close(self):
        self._closed.set()
        with self._lock:
            self._stop()


def pull_model(url, model, on_progress, cancelled, *, opener=_direct_open):
    if cancelled.is_set():
        raise PlannerError("Загрузка отменена")
    request = Request(
        url.rstrip("/") + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_message = None
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
