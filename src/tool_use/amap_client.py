from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, parse, request

from .protocol import build_tool_empty, build_tool_error, build_tool_success

AMAP_BASE_URL = "https://restapi.amap.com"
DEFAULT_TIMEOUT_SECONDS = 20


def _normalize_location_token(value: str) -> str:
    return value.strip().replace(" ", "")


def _is_coordinate_pair(value: str) -> bool:
    token = _normalize_location_token(value)
    if "," not in token:
        return False
    parts = token.split(",", 1)
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


class AmapClient:
    def __init__(self, *, api_key: str | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.api_key = api_key or os.getenv("AMAP_API_KEY", "").strip()
        self.timeout_seconds = timeout_seconds

    def _request(self, path: str, query: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            return build_tool_error("missing_amap_api_key")

        cleaned_query = {key: value for key, value in query.items() if value not in (None, "", [])}
        cleaned_query["key"] = self.api_key
        url = f"{AMAP_BASE_URL}{path}?{parse.urlencode(cleaned_query)}"

        try:
            with request.urlopen(url, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except error.HTTPError as exc:
            return build_tool_error(f"amap_http_{exc.code}", retryable=exc.code >= 500)
        except error.URLError:
            return build_tool_error("amap_network_error", retryable=True)
        except json.JSONDecodeError:
            return build_tool_error("amap_invalid_json")

        if payload.get("status") != "1":
            return build_tool_error(str(payload.get("info") or payload.get("infocode") or "amap_request_failed"))

        return build_tool_success(payload)

    def geocode(self, address: str, city: str | None = None) -> dict[str, Any]:
        raw_result = self._request("/v3/geocode/geo", {"address": address, "city": city, "output": "JSON"})
        if raw_result["status"] != "success":
            return raw_result

        payload = raw_result["data"]
        geocodes = payload.get("geocodes") or []
        if not geocodes:
            return build_tool_empty()

        first = geocodes[0]
        return build_tool_success(
            {
                "query": address,
                "city": city,
                "formatted_address": first.get("formatted_address"),
                "province": first.get("province"),
                "city_name": first.get("city"),
                "district": first.get("district"),
                "location": first.get("location"),
                "level": first.get("level"),
            }
        )

    def search_poi(
        self,
        keyword: str,
        *,
        city: str | None = None,
        around_location: str | None = None,
        radius_m: int | None = 3000,
    ) -> dict[str, Any]:
        location = around_location
        if location and not _is_coordinate_pair(location):
            geocode_result = self.geocode(location, city=city)
            if geocode_result["status"] == "success":
                location = geocode_result["data"].get("location")
            elif geocode_result["status"] == "error":
                return geocode_result
            else:
                location = None

        if location:
            raw_result = self._request(
                "/v3/place/around",
                {
                    "keywords": keyword,
                    "location": _normalize_location_token(location),
                    "radius": radius_m or 3000,
                    "output": "JSON",
                },
            )
        else:
            raw_result = self._request(
                "/v3/place/text",
                {
                    "keywords": keyword,
                    "city": city,
                    "citylimit": "true" if city else None,
                    "output": "JSON",
                },
            )

        if raw_result["status"] != "success":
            return raw_result

        payload = raw_result["data"]
        pois = payload.get("pois") or []
        if not pois:
            return build_tool_empty()

        normalized = []
        for poi in pois[:5]:
            normalized.append(
                {
                    "name": poi.get("name"),
                    "address": poi.get("address"),
                    "cityname": poi.get("cityname"),
                    "adname": poi.get("adname"),
                    "type": poi.get("type"),
                    "location": poi.get("location"),
                    "distance": poi.get("distance"),
                }
            )

        return build_tool_success(
            {
                "keyword": keyword,
                "city": city,
                "around_location": location,
                "count": len(pois),
                "pois": normalized,
            }
        )

    def plan_route(
        self,
        origin: str,
        destination: str,
        *,
        mode: str = "transit",
        city: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = (mode or "transit").strip().lower()
        origin_token = self._coerce_route_location(origin, city=city)
        if origin_token["status"] != "success":
            return origin_token
        destination_token = self._coerce_route_location(destination, city=city)
        if destination_token["status"] != "success":
            return destination_token

        origin_location = origin_token["data"]["location"]
        destination_location = destination_token["data"]["location"]
        path = "/v3/direction/transit/integrated"
        query = {
            "origin": origin_location,
            "destination": destination_location,
            "city": city,
            "output": "JSON",
        }

        if normalized_mode == "walking":
            path = "/v3/direction/walking"
            query.pop("city", None)
        elif normalized_mode == "driving":
            path = "/v3/direction/driving"
            query.pop("city", None)
        elif normalized_mode == "bicycling":
            path = "/v4/direction/bicycling"
            query.pop("city", None)
        elif normalized_mode != "transit":
            return build_tool_error(f"unsupported_route_mode:{normalized_mode}")

        raw_result = self._request(path, query)
        if raw_result["status"] != "success":
            return raw_result

        payload = raw_result["data"]
        route = payload.get("route") or payload.get("data") or {}
        paths = route.get("paths") or route.get("transits") or []
        if not paths:
            return build_tool_empty()

        first_path = paths[0]
        return build_tool_success(
            {
                "mode": normalized_mode,
                "origin": origin,
                "destination": destination,
                "origin_location": origin_location,
                "destination_location": destination_location,
                "distance": first_path.get("distance"),
                "duration": first_path.get("duration") or first_path.get("cost", {}).get("duration"),
                "taxi_cost": first_path.get("taxi_cost"),
                "walking_distance": first_path.get("walking_distance"),
                "segments": first_path.get("segments"),
            }
        )

    def _coerce_route_location(self, value: str, *, city: str | None = None) -> dict[str, Any]:
        if _is_coordinate_pair(value):
            return build_tool_success({"location": _normalize_location_token(value), "resolved_from": value})

        geocode_result = self.geocode(value, city=city)
        if geocode_result["status"] != "success":
            return geocode_result

        location = geocode_result["data"].get("location")
        if not isinstance(location, str) or not location.strip():
            return build_tool_empty("missing_location")

        return build_tool_success({"location": _normalize_location_token(location), "resolved_from": value})
