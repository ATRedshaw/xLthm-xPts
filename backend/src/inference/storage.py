"""Atomic SQLite storage for live xPts projection batches."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .simulation import gameweek_projection


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    code INTEGER NOT NULL,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL
);

CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    code INTEGER NOT NULL,
    name TEXT NOT NULL,
    position TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    price REAL NOT NULL,
    selected_by REAL NOT NULL,
    status TEXT NOT NULL,
    availability_probability REAL NOT NULL,
    news TEXT NOT NULL
);

CREATE TABLE fixtures (
    id INTEGER PRIMARY KEY,
    gameweek INTEGER,
    kickoff_time TEXT NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams(id),
    away_team_id INTEGER NOT NULL REFERENCES teams(id),
    home_expected_goals REAL NOT NULL,
    away_expected_goals REAL NOT NULL,
    home_win_probability REAL NOT NULL,
    draw_probability REAL NOT NULL,
    away_win_probability REAL NOT NULL,
    home_clean_sheet_probability REAL NOT NULL,
    away_clean_sheet_probability REAL NOT NULL
);

CREATE TABLE player_fixture_projections (
    player_id INTEGER NOT NULL REFERENCES players(id),
    fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
    gameweek INTEGER,
    opponent_team_id INTEGER NOT NULL REFERENCES teams(id),
    is_home INTEGER NOT NULL,
    xpts REAL NOT NULL,
    xmins REAL NOT NULL,
    action_probabilities TEXT NOT NULL,
    expected_actions TEXT NOT NULL,
    xpts_breakdown TEXT NOT NULL,
    outcome_probabilities TEXT NOT NULL,
    model_context TEXT NOT NULL,
    PRIMARY KEY (player_id, fixture_id)
);

CREATE TABLE player_gameweek_projections (
    player_id INTEGER NOT NULL REFERENCES players(id),
    gameweek INTEGER NOT NULL,
    xpts REAL NOT NULL,
    xmins REAL NOT NULL,
    fixture_ids TEXT NOT NULL,
    outcome_probabilities TEXT NOT NULL,
    PRIMARY KEY (player_id, gameweek)
);

CREATE INDEX idx_players_position ON players(position);
CREATE INDEX idx_players_team ON players(team_id);
CREATE INDEX idx_fixtures_gameweek ON fixtures(gameweek);
CREATE INDEX idx_player_fixture_gameweek ON player_fixture_projections(gameweek);
CREATE INDEX idx_player_gameweek_gameweek ON player_gameweek_projections(gameweek);
"""


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _metadata_rows(metadata: Mapping[str, object]) -> list[tuple[str, str]]:
    return [(str(key), _json(value)) for key, value in metadata.items()]


def _blank_gameweek() -> dict[str, object]:
    return {
        "fixture_ids": [],
        "xpts": 0.0,
        "xmins": 0.0,
        "outcome_probabilities": {
            "negative_points": 0.0,
            "zero_points": 1.0,
            "two_plus_points": 0.0,
            "five_plus_points": 0.0,
            "ten_plus_points": 0.0,
            "percentiles": {
                name: 0.0 for name in ("p10", "p25", "p50", "p75", "p90")
            },
            "points_distribution": {"0": 1.0},
        },
    }


def write_projection_database(
    path: str | Path,
    *,
    context: Mapping[str, object],
    metadata: Mapping[str, object],
    fixture_forecasts: pd.DataFrame,
    fixture_projections: Mapping[str, Sequence[Mapping[str, object]]],
    gameweek_samples: Mapping[tuple[str, int], Mapping[str, object]],
    gameweeks: Sequence[int],
) -> Path:
    """Write a complete batch to a temporary database, then replace the live file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection = sqlite3.connect(temporary_path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            _metadata_rows(metadata),
        )
        teams = context["teams"]
        connection.executemany(
            "INSERT INTO teams (id, code, name, short_name) VALUES (?, ?, ?, ?)",
            [
                (int(team["id"]), int(team["code"]), str(team["name"]), str(team["short_name"]))
                for team in teams.values()
            ],
        )
        players = context["players"]
        connection.executemany(
            """
            INSERT INTO players (
                id, code, name, position, team_id, price, selected_by,
                status, availability_probability, news
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(player["id"]), int(player["code"]), str(player["web_name"]),
                    str(player["position"]), int(player["team_id"]),
                    float(player["price_millions"]), float(player["ownership_percent"]),
                    str(player["status"]), float(player["availability_probability"]),
                    str(player["news"]),
                )
                for player in players.values()
            ],
        )
        fixture_context = {
            int(fixture["id"]): fixture for fixture in context["fixtures"]
        }
        connection.executemany(
            """
            INSERT INTO fixtures (
                id, gameweek, kickoff_time, home_team_id, away_team_id,
                home_expected_goals, away_expected_goals,
                home_win_probability, draw_probability, away_win_probability,
                home_clean_sheet_probability, away_clean_sheet_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(forecast["fixture_id"]),
                    None if pd.isna(forecast["gameweek"]) else int(forecast["gameweek"]),
                    str(forecast["kickoff_time"]),
                    int(fixture_context[int(forecast["fixture_id"])]["home_team_id"]),
                    int(fixture_context[int(forecast["fixture_id"])]["away_team_id"]),
                    float(forecast["home_expected_goals"]),
                    float(forecast["away_expected_goals"]),
                    float(forecast["home_win_probability"]),
                    float(forecast["draw_probability"]),
                    float(forecast["away_win_probability"]),
                    float(forecast["home_clean_sheet_probability"]),
                    float(forecast["away_clean_sheet_probability"]),
                )
                for _, forecast in fixture_forecasts.iterrows()
            ],
        )
        player_fixture_rows = []
        for player_id, projections in fixture_projections.items():
            for projection in projections:
                fixture = fixture_context[int(projection["fixture_id"])]
                team_id = int(players[player_id]["team_id"])
                opponent_id = (
                    int(fixture["away_team_id"])
                    if team_id == int(fixture["home_team_id"])
                    else int(fixture["home_team_id"])
                )
                player_fixture_rows.append(
                    (
                        int(player_id), int(projection["fixture_id"]),
                        None if fixture["gameweek"] is None else int(fixture["gameweek"]),
                        opponent_id, int(bool(projection["is_home"])),
                        float(projection["xpts"]), float(projection["xmins"]),
                        _json(projection["action_probabilities"]),
                        _json(projection["expected_actions"]),
                        _json(projection["xpts_breakdown"]),
                        _json(projection["outcome_probabilities"]),
                        _json(projection["model_context"]),
                    )
                )
        connection.executemany(
            """
            INSERT INTO player_fixture_projections (
                player_id, fixture_id, gameweek, opponent_team_id, is_home,
                xpts, xmins, action_probabilities, expected_actions,
                xpts_breakdown, outcome_probabilities, model_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            player_fixture_rows,
        )
        gameweek_rows = []
        for player_id in players:
            for gameweek in gameweeks:
                sample = gameweek_samples.get((player_id, int(gameweek)))
                projection = _blank_gameweek() if sample is None else gameweek_projection(sample)
                gameweek_rows.append(
                    (
                        int(player_id), int(gameweek), float(projection["xpts"]),
                        float(projection["xmins"]), _json(projection["fixture_ids"]),
                        _json(projection["outcome_probabilities"]),
                    )
                )
        connection.executemany(
            """
            INSERT INTO player_gameweek_projections (
                player_id, gameweek, xpts, xmins, fixture_ids, outcome_probabilities
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            gameweek_rows,
        )
        connection.commit()
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    temporary_path.replace(output_path)
    return output_path
