"""Fetch and normalise current-season history from the official FPL API."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .vaastav import SeasonData, normalise_season


DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"
DEFAULT_REQUEST_ATTEMPTS = 3
DEFAULT_REQUEST_WORKERS = 8


@dataclass(frozen=True, slots=True)
class LiveFplData:
    bootstrap: dict[str, Any]
    fixtures: list[dict[str, Any]]


def fetch_json(
    url: str,
    *,
    timeout: float = 30.0,
    attempts: int = DEFAULT_REQUEST_ATTEMPTS,
) -> object:
    """Fetch JSON with bounded retries for unattended batch runs."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "fpl-xpts-inference/1.0"})
            with urlopen(request, timeout=timeout) as response:  # nosec B310 - configured URL
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("JSON request exhausted without returning or raising")


def fetch_live_fpl(
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    attempts: int = DEFAULT_REQUEST_ATTEMPTS,
) -> LiveFplData:
    root = base_url.rstrip("/")
    bootstrap = fetch_json(
        f"{root}/bootstrap-static/", timeout=timeout, attempts=attempts
    )
    fixtures = fetch_json(f"{root}/fixtures/", timeout=timeout, attempts=attempts)
    if not isinstance(bootstrap, dict):
        raise ValueError("FPL bootstrap-static response must be an object")
    if not isinstance(fixtures, list):
        raise ValueError("FPL fixtures response must be a list")
    for key in ("elements", "teams", "element_types", "events"):
        if not isinstance(bootstrap.get(key), list):
            raise ValueError(f"FPL bootstrap-static response is missing {key}")
    return LiveFplData(bootstrap=bootstrap, fixtures=fixtures)


def fetch_player_histories(
    players: Sequence[Mapping[str, object]],
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    attempts: int = DEFAULT_REQUEST_ATTEMPTS,
    workers: int = DEFAULT_REQUEST_WORKERS,
    progress: Callable[[str], None] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Fetch current-season fixture history for every current FPL player."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = base_url.rstrip("/")
    player_ids = [int(player["id"]) for player in players]

    def fetch_history(player_id: int) -> tuple[int, list[dict[str, Any]]]:
        payload = fetch_json(
            f"{root}/element-summary/{player_id}/",
            timeout=timeout,
            attempts=attempts,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("history"), list):
            raise ValueError(f"FPL element-summary response is invalid for player {player_id}")
        return player_id, payload["history"]

    histories: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (player_id, history) in enumerate(
            executor.map(fetch_history, player_ids), start=1
        ):
            histories[player_id] = history
            if progress and (index % 50 == 0 or index == len(player_ids)):
                progress(f"[current FPL] downloaded {index:,}/{len(player_ids):,} player histories")
    return histories


def normalise_current_season(
    live: LiveFplData,
    player_histories: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    season: str,
) -> pd.DataFrame:
    """Return official FPL history in the shared canonical archive format."""

    positions = {
        int(position["id"]): str(position["singular_name_short"])
        for position in live.bootstrap["element_types"]
    }
    finished_fixture_ids = {
        int(fixture["id"])
        for fixture in live.fixtures
        if fixture.get("finished")
    }
    rows: list[dict[str, object]] = []
    for player in live.bootstrap["elements"]:
        player_id = int(player["id"])
        position = positions[int(player["element_type"])]
        for history in player_histories.get(player_id, []):
            if int(history["fixture"]) not in finished_fixture_ids:
                continue
            row = dict(history)
            row.update(
                {
                    "element": player_id,
                    "GW": history.get("round"),
                    "name": str(player["web_name"]),
                    "position": position,
                }
            )
            rows.append(row)

    required_history_columns = {
        "element",
        "fixture",
        "GW",
        "kickoff_time",
        "minutes",
        "opponent_team",
        "position",
        "was_home",
        "name",
    }
    player_fixtures = pd.DataFrame(rows)
    for column in required_history_columns.difference(player_fixtures.columns):
        player_fixtures[column] = pd.NA

    season_data = SeasonData(
        season=season,
        player_fixtures=player_fixtures,
        players=pd.DataFrame(live.bootstrap["elements"]),
        teams=pd.DataFrame(live.bootstrap["teams"]),
        fixtures=pd.DataFrame(live.fixtures),
    )
    return normalise_season(season_data)


def load_current_season(
    live: LiveFplData,
    *,
    season: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    attempts: int = DEFAULT_REQUEST_ATTEMPTS,
    workers: int = DEFAULT_REQUEST_WORKERS,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    histories = fetch_player_histories(
        live.bootstrap["elements"],
        base_url=base_url,
        timeout=timeout,
        attempts=attempts,
        workers=workers,
        progress=progress,
    )
    return normalise_current_season(live, histories, season=season)
