from __future__ import annotations

from src.deploy.frontend_server import _build_upstream_request_headers, _normalize_upstream_base_url


def test_normalize_upstream_base_url_strips_v1_suffix() -> None:
    assert _normalize_upstream_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert _normalize_upstream_base_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000"
    assert _normalize_upstream_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"


def test_build_upstream_request_headers_prefers_server_side_api_key() -> None:
    headers = _build_upstream_request_headers(
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer client-token",
            "Connection": "keep-alive",
        },
        server_side_api_key="server-token",
    )

    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer server-token"
    assert "Connection" not in headers
