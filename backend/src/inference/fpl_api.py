"""Minimal official FPL API client for live players and fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"


@dataclass(frozen=True, slots=True)
class LiveFplData:
    bootstrap: dict[str, Any]
    fixtures: list[dict[str, Any]]


def fetch_json(url: str, *, timeout: float = 30.0) -> object:
    request = Request(url, headers={"User-Agent": "fpl-xpts-inference/1.0"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310 - configured URL
        return json.loads(response.read().decode("utf-8"))


def fetch_live_fpl(
    *, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0
) -> LiveFplData:
    root = base_url.rstrip("/")
    bootstrap = fetch_json(f"{root}/bootstrap-static/", timeout=timeout)
    fixtures = fetch_json(f"{root}/fixtures/", timeout=timeout)
    if not isinstance(bootstrap, dict):
        raise ValueError("FPL bootstrap-static response must be an object")
    if not isinstance(fixtures, list):
        raise ValueError("FPL fixtures response must be a list")
    for key in ("elements", "teams", "element_types", "events"):
        if not isinstance(bootstrap.get(key), list):
            raise ValueError(f"FPL bootstrap-static response is missing {key}")
    return LiveFplData(bootstrap=bootstrap, fixtures=fixtures)
