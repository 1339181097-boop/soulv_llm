from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib import error, parse, request

from src.data_pipeline.data_utils import configure_console_output, log_info
from src.tool_use import AmapClient, OpenAICompatibleChatClient, ToolCallingOrchestrator, build_amap_tool_schemas

DEFAULT_FRONTEND_DIR = Path(__file__).resolve().with_name("web")
DEFAULT_TOOL_SCHEMAS = build_amap_tool_schemas()
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class ServerConfig:
    frontend_dir: Path
    upstream_base_url: str
    upstream_api_key: str
    default_model: str
    request_timeout_seconds: int


def _normalize_upstream_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("upstream_base_url must be a non-empty string")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized.rstrip("/")


def _build_upstream_request_headers(
    request_headers: Mapping[str, str],
    *,
    server_side_api_key: str,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in request_headers.items():
        lowered = name.lower()
        if lowered in HOP_BY_HOP_HEADERS or lowered in {"host", "content-length"}:
            continue
        headers[name] = value
    if server_side_api_key:
        headers["Authorization"] = f"Bearer {server_side_api_key}"
    return headers


class FrontendHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class, *, app_config: ServerConfig) -> None:  # noqa: ANN001
        super().__init__(server_address, handler_class)
        self.app_config = app_config


class FrontendRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    @property
    def app_config(self) -> ServerConfig:
        return self.server.app_config  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Connection", "close")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        log_info(f"{self.address_string()} - {format % args}")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = parse.urlsplit(self.path)
        if parsed.path == "/healthz":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "default_model": self.app_config.default_model,
                    "frontend_dir": str(self.app_config.frontend_dir),
                },
            )
            return
        if parsed.path == "/api/config":
            self._write_json(
                HTTPStatus.OK,
                {
                    "default_model": self.app_config.default_model,
                    "default_api_base_url": "/v1",
                    "tool_orchestrate_path": "/api/tool-orchestrate",
                    "tool_schemas": DEFAULT_TOOL_SCHEMAS,
                },
            )
            return
        if parsed.path == "/v1" or parsed.path.startswith("/v1/"):
            self._proxy_request()
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = parse.urlsplit(self.path)
        if parsed.path == "/api/tool-orchestrate":
            self._handle_tool_orchestrate()
            return
        if parsed.path == "/v1" or parsed.path.startswith("/v1/"):
            self._proxy_request()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {parsed.path}"})

    def _handle_tool_orchestrate(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "messages must be a non-empty list"})
            return

        model = str(payload.get("model") or self.app_config.default_model).strip()
        if not model:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "model must be a non-empty string"})
            return

        disable_thinking = bool(payload.get("disable_thinking", True))
        max_tokens = int(payload.get("max_tokens", 512))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        max_tool_rounds = int(payload.get("max_tool_rounds", 2))
        tool_test_mode = payload.get("tool_test_mode")
        if tool_test_mode is not None and not isinstance(tool_test_mode, dict):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "tool_test_mode must be an object if provided"})
            return

        chat_client = OpenAICompatibleChatClient(
            base_url=self.app_config.upstream_base_url,
            api_key=self.app_config.upstream_api_key,
            timeout_seconds=self.app_config.request_timeout_seconds,
            disable_thinking=disable_thinking,
        )
        orchestrator = ToolCallingOrchestrator(
            chat_client=chat_client,
            model=model,
            amap_client=AmapClient(),
            max_tool_rounds=max_tool_rounds,
        )

        try:
            result = orchestrator.run(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tool_test_mode=tool_test_mode,
            )
        except Exception as exc:  # noqa: BLE001
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return

        self._write_json(
            HTTPStatus.OK,
            {
                "model": model,
                "base_url": f"{self.app_config.upstream_base_url}/v1",
                "result": result,
            },
        )

    def _proxy_request(self) -> None:
        parsed = parse.urlsplit(self.path)
        upstream_url = f"{self.app_config.upstream_base_url}{parsed.path}"
        if parsed.query:
            upstream_url = f"{upstream_url}?{parsed.query}"

        body = self._read_raw_body()
        headers = _build_upstream_request_headers(
            self.headers,
            server_side_api_key=self.app_config.upstream_api_key,
        )
        headers.setdefault("Accept", "*/*")
        headers["X-Forwarded-Host"] = self.headers.get("Host", "")
        headers["X-Forwarded-Proto"] = "http"

        req = request.Request(
            upstream_url,
            data=body if body else None,
            headers=headers,
            method=self.command,
        )

        try:
            with request.urlopen(req, timeout=self.app_config.request_timeout_seconds) as upstream:
                self._relay_upstream_response(
                    status=upstream.status,
                    response_headers=dict(upstream.headers.items()),
                    body_stream=upstream,
                )
        except error.HTTPError as exc:
            self._relay_upstream_response(
                status=exc.code,
                response_headers=dict(exc.headers.items()),
                body_bytes=exc.read(),
            )
        except error.URLError as exc:
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": f"Upstream request failed: {exc}"})

    def _relay_upstream_response(
        self,
        *,
        status: int,
        response_headers: Mapping[str, str],
        body_bytes: bytes | None = None,
        body_stream=None,  # noqa: ANN001
    ) -> None:
        self.send_response(status)
        content_length = response_headers.get("Content-Length")
        for name, value in response_headers.items():
            lowered = name.lower()
            if lowered in HOP_BY_HOP_HEADERS or lowered in {"content-length", "server", "date"}:
                continue
            self.send_header(name, value)
        if body_bytes is not None:
            self.send_header("Content-Length", str(len(body_bytes)))
        elif content_length and content_length.isdigit():
            self.send_header("Content-Length", content_length)
        self.end_headers()

        if body_bytes is not None:
            self.wfile.write(body_bytes)
            self.wfile.flush()
            return

        if body_stream is None:
            return

        while True:
            chunk = body_stream.read(64 * 1024)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()

    def _read_raw_body(self) -> bytes:
        content_length = self.headers.get("Content-Length")
        if not content_length:
            return b""
        try:
            byte_count = int(content_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        return self.rfile.read(byte_count)

    def _read_json_body(self) -> dict[str, Any]:
        body = self._read_raw_body()
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Request body must be valid UTF-8 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the static frontend plus a same-origin vLLM gateway.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the public gateway.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind the public gateway.")
    parser.add_argument(
        "--frontend-dir",
        default=str(DEFAULT_FRONTEND_DIR),
        help="Directory containing the static frontend files.",
    )
    parser.add_argument(
        "--upstream-base-url",
        default="http://127.0.0.1:8000",
        help="Internal vLLM base URL. Both http://host:port and http://host:port/v1 are accepted.",
    )
    parser.add_argument(
        "--upstream-api-key",
        default="",
        help="Optional API key injected server-side when calling the upstream vLLM server.",
    )
    parser.add_argument(
        "--default-model",
        default="qwen3_8b_stage2_amap_tool_use",
        help="Default model name shown in the frontend.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=300,
        help="HTTP timeout in seconds for upstream requests.",
    )
    return parser


def main() -> None:
    configure_console_output()
    args = build_arg_parser().parse_args()
    frontend_dir = Path(args.frontend_dir).resolve()
    if not frontend_dir.exists():
        raise FileNotFoundError(f"Frontend directory does not exist: {frontend_dir}")

    app_config = ServerConfig(
        frontend_dir=frontend_dir,
        upstream_base_url=_normalize_upstream_base_url(args.upstream_base_url),
        upstream_api_key=args.upstream_api_key.strip(),
        default_model=args.default_model.strip(),
        request_timeout_seconds=args.request_timeout_seconds,
    )

    handler_class = partial(FrontendRequestHandler, directory=str(frontend_dir))
    server = FrontendHTTPServer((args.host, args.port), handler_class, app_config=app_config)
    log_info(f"Frontend dir: {frontend_dir}")
    log_info(f"Proxying /v1 to: {app_config.upstream_base_url}/v1")
    log_info(f"Tool orchestration endpoint: http://{args.host}:{args.port}/api/tool-orchestrate")
    log_info(f"Open the UI at: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_info("Shutting down frontend gateway")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
