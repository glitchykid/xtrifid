"""Select the local Ollama server and send one bounded chat request."""

import http.client
import ipaddress
import json
import os
import socket
import subprocess
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
    urlopen,
)

from . import resources
from .resources import GIB, MIB

DEFAULT_URL = "http://127.0.0.1:11434"
CHAT_TIMEOUT = 300
WATCH_INTERVAL = 0.4
RESPONSE_LIMIT = 524288
_active = Lock()
PRIVATE_NETWORKS = tuple(
    ipaddress.IPv4Network(cidr) for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


class PlannerError(Exception):
    pass


class ModelMissing(PlannerError):
    def __init__(self, url, model, opener=urlopen, memory_limit=None):
        self.url = url
        self.model = model
        self.opener = opener
        self.memory_limit = memory_limit
        super().__init__(f"На сервере Ollama отсутствует модель {model}. Загрузите модель.")


def _inspect_service(template):
    result = subprocess.run(
        [
            "docker",
            "--host",
            "unix:///var/run/docker.sock",
            "inspect",
            "--type",
            "container",
            "--format",
            template,
            "xtrifid-ollama-1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        text=True,
        check=True,
        timeout=3,
    )
    return result.stdout.strip()


def docker_url():
    """Read just the service address from the project's local Linux Docker bridge."""
    try:
        data = _inspect_service(
            '{{with index .NetworkSettings.Networks "xtrifid_default"}}{{.IPAddress}}{{end}}'
        )
        address = ipaddress.IPv4Address(data)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if any(address in network for network in PRIVATE_NETWORKS):
        return f"http://{address}:11434"
    return None


@dataclass(frozen=True)
class _DockerLimits:
    memory: int = 6 * GIB
    cpus: int = 4
    pid: int = 0

    @property
    def budget(self):
        return self.memory - min(512 * MIB, self.memory // 8)


def _docker_limits(discovered):
    if discovered:
        try:
            ident, pid, memory, nano = _inspect_service(
                "{{.Id}} {{.State.Pid}} {{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}"
            ).split()
            pid, memory, nano = int(pid), int(memory), int(nano)
            # A Docker Desktop VM can return a PID from another namespace.
            # Read a sibling cgroup only when it belongs to this container.
            try:
                group = (Path("/proc") / str(pid) / "cgroup").read_text()
            except OSError:
                group = ""
            known_pid = pid if len(ident) == 64 and ident in group and pid > 1 else 0
            return _DockerLimits(
                min(6 * GIB, memory or 6 * GIB),
                min(4, max(1, nano // 1_000_000_000)) if nano else 4,
                known_pid,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return _DockerLimits(0, 1)
    try:
        return _DockerLimits(
            min(6 * GIB, int(os.environ.get("XTRIFID_RUNNER_MEMORY", 6 * GIB))),
            min(4, max(1, int(os.environ.get("XTRIFID_RUNNER_CPUS", 4)))),
        )
    except ValueError:
        return _DockerLimits(0, 1)


def _capacity(server):
    client = resources.capacity()
    if server is not None and server.pid:
        other = resources.capacity(pid=server.pid)
        return resources.Capacity(
            min(client.headroom, other.headroom), min(client.cpus, other.cpus)
        )
    return client


def _local_urls(local_server):
    yield DEFAULT_URL, ConnectionRefusedError, None
    bridge = docker_url()
    if bridge:
        # A Docker Desktop VM's bridge may be unreachable from the host.
        yield bridge, OSError, None
    if local_server is not None:
        yield local_server.start(), ConnectionRefusedError, local_server


def _read_response(response, url, model, opener, memory_limit):
    if response.status == 404:
        try:
            error = json.loads(response.read(4096)).get("error")
        except (ValueError, AttributeError, RecursionError):
            error = None
        if error == f"model '{model}' not found":
            raise ModelMissing(url, model, opener, memory_limit)
    if response.status != 200:
        raise PlannerError(f"Ollama вернула HTTP {response.status}.")
    raw = response.read(RESPONSE_LIMIT + 1)
    if len(raw) > RESPONSE_LIMIT:
        raise PlannerError("Ответ Ollama слишком большой")
    return raw


def _inventory(opener, url, route):
    with opener(url + route, timeout=5) as response:
        raw = response.read(RESPONSE_LIMIT + 1)
    try:
        if len(raw) <= RESPONSE_LIMIT:
            models = json.loads(raw)["models"]
            if isinstance(models, list) and all(isinstance(m, dict) for m in models):
                return models
    except (KeyError, ValueError, TypeError, RecursionError):
        pass
    raise PlannerError("Ollama вернула неполные сведения о моделях.")


def _tag(name):
    if not isinstance(name, str) or not name.strip():
        raise PlannerError("Ollama не сообщила имя установленной модели.")
    name = name.lower().removeprefix("registry.ollama.ai/").removeprefix("library/")
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def _loaded_bytes(running, model, context):
    # /api/ps reports total memory and the part in VRAM. Only subtract weights
    # that are already accounted for in available system memory.
    for item in running:
        if (
            model.get("digest")
            and item.get("digest") == model["digest"]
            and type(item.get("context_length")) is int
            and item["context_length"] >= context
            and type(item.get("size")) is int
            and type(item.get("size_vram")) is int
            and 0 <= item["size_vram"] <= item["size"]
        ):
            return min(model["size"], item["size"] - item["size_vram"])
    return 0


def _bound_options(settings, model, running, server):
    size = model.get("size")
    if type(size) is not int or not 0 < size < 1 << 63:
        raise PlannerError("Ollama не сообщила размер весов выбранной модели.")
    if server is not None and server.memory <= 0:
        raise PlannerError("Не удалось определить лимит памяти контейнера Ollama.")
    cap = _capacity(server)
    limit = 4096 if cap.headroom < 4 * GIB else 8192
    context, predict = settings.get("num_ctx", limit), settings.get("num_predict", 6000)
    if type(context) is not int or type(predict) is not int or min(context, predict) < 1:
        raise PlannerError("Недопустимый размер контекста Ollama.")
    context = min(context, limit)
    for ctx in dict.fromkeys((context, min(context, 4096), min(context, 2048))):
        estimate = resources.model_working_bytes(size, ctx)
        if server is not None and estimate > server.budget:
            continue
        if estimate - _loaded_bytes(running, model, ctx) <= cap.headroom:
            return settings | {
                "num_ctx": ctx,
                "num_thread": min(cap.threads, server.cpus)
                if server is not None
                else cap.threads,
                "num_batch": min(ctx, 128),
                "num_predict": max(1, min(6000, predict, ctx * 3 // 4)),
            }
    limit_info = "" if server is None else f" Лимит контейнера {server.memory / GIB:.1f} ГиБ."
    raise PlannerError(
        f"Недостаточно памяти для модели ({size / GIB:.1f} ГиБ весов). "
        f"После резерва свободно {cap.headroom / GIB:.1f} ГиБ.{limit_info} "
        "Закройте другие приложения или выберите меньшую модель."
    )


class _Watch(HTTPHandler, HTTPSHandler):
    """Interrupt an in-flight HTTP read on pressure, even before the first response byte."""

    def __init__(self, url, owner, server):
        HTTPSHandler.__init__(self)
        self.url, self.owner = url, owner
        self.server = server
        self.done = Event()
        self.lock = Lock()
        self.socket = None
        self.reason = None
        self.thread = Thread(target=self._run, daemon=True)

    def _track(self, kind):
        watch = self

        class Connection(kind):
            def connect(self):
                super().connect()
                with watch.lock:
                    if watch.reason:
                        self.close()
                        raise PlannerError(watch.reason)
                    if watch.socket is not None:
                        watch.socket.close()
                    # A separate handle survives urllib handing the socket to the response.
                    watch.socket = socket.socket(fileno=os.dup(self.sock.fileno()))

        return Connection

    def http_open(self, request):
        return self.do_open(self._track(http.client.HTTPConnection), request)

    def https_open(self, request):
        return self.do_open(
            self._track(http.client.HTTPSConnection), request, context=self._context
        )

    def _run(self):
        deadline = time.monotonic() + CHAT_TIMEOUT
        while not self.done.wait(WATCH_INTERVAL):
            if _capacity(self.server).headroom < 128 * MIB or (
                self.owner is not None and self.owner.over_memory_limit(self.url)
            ):
                reason = "Генерация остановлена, чтобы сохранить резерв памяти."
            elif time.monotonic() >= deadline:
                reason = "Истекло время ожидания генерации Ollama."
            else:
                continue
            with self.lock:
                if self.done.is_set():
                    return
                self.reason = reason
                if self.socket is not None:
                    try:
                        self.socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass  # The connection may have just closed on its own.
            if self.owner is not None:
                self.owner.reclaim(self.url)
            return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()
        if self.socket is not None:
            self.socket.close()
        if self.reason:
            raise PlannerError(self.reason)


def _send(url, model, body, *, direct=False, owner=None, server=None):
    proxy = (ProxyHandler({}),) if direct else ()
    opener = build_opener(*proxy).open
    installed = _inventory(opener, url, "/api/tags")
    selected = next(
        (m for m in installed if _tag(m.get("name") or m.get("model")) == _tag(model)), None
    )
    memory_limit = server.budget if server is not None else None
    if selected is None:
        raise ModelMissing(url, model, opener, memory_limit)
    try:
        running = _inventory(opener, url, "/api/ps")
    except (PlannerError, OSError, http.client.HTTPException):
        running = []  # Without positive evidence, reserve the cost of a cold load.
    payload = json.loads(body)
    payload["model"] = model
    payload["options"] = _bound_options(payload.get("options", {}), selected, running, server)
    payload["keep_alive"] = 0
    request = Request(
        url + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _Watch(url, owner, server) as guard:
        try:
            # Let the wall-clock watchdog report expiry and reclaim our own runner
            # before the socket's independent inactivity timeout can fire.
            response = build_opener(*proxy, guard).open(
                request, timeout=CHAT_TIMEOUT + 2 * WATCH_INTERVAL
            )
        except HTTPError as exc:
            response = exc
        with response:
            return _read_response(response, url, model, opener, memory_limit)


def post_chat(configured_url, model, body, local_server=None):
    if not _active.acquire(blocking=False):
        raise PlannerError("Другой запрос Ollama уже выполняется.")
    try:
        if configured_url:
            server = _docker_limits(False) if configured_url == "http://ollama:11434" else None
            return _send(configured_url, model, body, server=server)
        for url, connection_unavailable, owner in _local_urls(local_server):
            endpoint = urlsplit(url)
            with closing(
                http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
            ) as connection:
                # Only fail over before sending any part of the request.
                try:
                    connection.connect()
                except connection_unavailable:
                    continue
            server = _docker_limits(True) if owner is None and url != DEFAULT_URL else None
            return _send(url, model, body, direct=True, owner=owner, server=server)
    except (OSError, http.client.HTTPException) as exc:
        hint = " Проверьте OLLAMA_URL и доступность сервера." if configured_url else ""
        raise PlannerError(f"Не удалось получить ответ Ollama.{hint}") from exc
    finally:
        _active.release()
    raise PlannerError(
        "Локальный сервер Ollama недоступен. Запустите ollama serve "
        "или контейнер проекта командой docker compose up -d ollama."
    )
