"""Select the local Ollama server and send one bounded chat request."""

import http.client
import ipaddress
import json
import subprocess
from contextlib import closing
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen

DEFAULT_URL = "http://127.0.0.1:11434"
CHAT_TIMEOUT = 300
RESPONSE_LIMIT = 524288
PRIVATE_NETWORKS = tuple(
    ipaddress.IPv4Network(cidr) for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


class PlannerError(Exception):
    pass


class ModelMissing(PlannerError):
    def __init__(self, url, model, opener=urlopen):
        self.url = url
        self.model = model
        self.opener = opener
        super().__init__(f"На сервере Ollama отсутствует модель {model}. Загрузите модель.")


def docker_url():
    """Read just the service address from the project's local Linux Docker bridge."""
    try:
        result = subprocess.run(
            [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "inspect",
                "--type",
                "container",
                "--format",
                '{{with index .NetworkSettings.Networks "xtrifid_default"}}{{.IPAddress}}{{end}}',
                "xtrifid-ollama-1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            text=True,
            check=True,
            timeout=3,
        )
        address = ipaddress.IPv4Address(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if any(address in network for network in PRIVATE_NETWORKS):
        return f"http://{address}:11434"
    return None


def _local_urls(local_server):
    yield DEFAULT_URL, ConnectionRefusedError
    bridge = docker_url()
    if bridge:
        # A Docker Desktop VM's bridge may be unreachable from the host.
        yield bridge, OSError
    if local_server is not None:
        yield local_server.start(), ConnectionRefusedError


def _read_response(response, url, model, opener):
    if response.status == 404:
        try:
            error = json.loads(response.read(4096)).get("error")
        except (ValueError, AttributeError, RecursionError):
            error = None
        if error == f"model '{model}' not found":
            raise ModelMissing(url, model, opener)
    if response.status != 200:
        raise PlannerError(f"Ollama вернула HTTP {response.status}.")
    raw = response.read(RESPONSE_LIMIT + 1)
    if len(raw) > RESPONSE_LIMIT:
        raise PlannerError("Ответ Ollama слишком большой")
    return raw


def post_chat(configured_url, model, body, local_server=None):
    headers = {"Content-Type": "application/json"}
    try:
        if configured_url:
            request = Request(configured_url + "/api/chat", data=body, headers=headers)
            try:
                response = urlopen(request, timeout=CHAT_TIMEOUT)
            except HTTPError as exc:
                response = exc
            with response:
                return _read_response(response, configured_url, model, urlopen)
        direct_open = build_opener(ProxyHandler({})).open
        for url, connection_unavailable in _local_urls(local_server):
            endpoint = urlsplit(url)
            with closing(
                http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
            ) as connection:
                # Only fail over before sending any part of the request.
                try:
                    connection.connect()
                except connection_unavailable:
                    continue
                connection.sock.settimeout(CHAT_TIMEOUT)
                connection.request("POST", "/api/chat", body, headers)
                with connection.getresponse() as response:
                    return _read_response(response, url, model, direct_open)
    except (OSError, http.client.HTTPException) as exc:
        hint = " Проверьте OLLAMA_URL и доступность сервера." if configured_url else ""
        raise PlannerError(f"Не удалось получить ответ Ollama.{hint}") from exc
    raise PlannerError(
        "Локальный сервер Ollama недоступен. Запустите ollama serve "
        "или контейнер проекта командой docker compose up -d ollama."
    )
