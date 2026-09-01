"""Build future player-fixture features from Vaastav history and live FPL data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.features import build_training_dataset
from preprocessing.vaastav import OUTCOME_COLUMNS


LIVE_PLAYER_COLUMNS = [
    "live_status",
    "live_availability_probability",
    "live_chance_of_playing_next_round",
    "live_news",
    "live_news_added",
    "live_price_millions",
    "live_ownership_percent",
]


def _as_utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _availability_probability(player: dict[str, Any]) -> float:
    chance = player.get("chance_of_playing_next_round")
    if chance is not None:
        return float(np.clip(float(chance) / 100.0, 0, 1))
    return 1.0 if player.get("status") == "a" else 0.0


def normalise_live_future_rows(
    bootstrap: dict[str, Any],
    fixtures: list[dict[str, Any]],
    *,
    season: str,
    retrieved_at: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return one row for every current player and unfinished team fixture."""
    retrieved = retrieved_at or datetime.now(timezone.utc)
    teams = {int(team["id"]): team for team in bootstrap["teams"]}
    positions = {
        int(position["id"]): str(position["singular_name_short"]).upper()
        for position in bootstrap["element_types"]
    }
    remaining = []
    skipped = []
    for fixture in fixtures:
        if fixture.get("finished") or fixture.get("started"):
            continue
        kickoff = _as_utc(fixture.get("kickoff_time"))
        if pd.isna(kickoff):
            skipped.append({"fixture_id": fixture.get("id"), "reason": "kickoff_time is not assigned"})
            continue
        remaining.append(fixture)
    if not remaining:
        raise ValueError("The FPL API returned no remaining fixtures with kickoff times")

    fixtures_by_team: dict[int, list[dict[str, Any]]] = {team_id: [] for team_id in teams}
    for fixture in remaining:
        fixtures_by_team[int(fixture["team_h"])].append(fixture)
        fixtures_by_team[int(fixture["team_a"])].append(fixture)

    total_players = float(bootstrap.get("total_players") or 0)
    rows: list[dict[str, object]] = []
    live_players: dict[str, dict[str, object]] = {}
    for player in bootstrap["elements"]:
        team_id = int(player["team"])
        team = teams[team_id]
        player_id = int(player["id"])
        player_code = int(player["code"])
        player_key = f"code:{player_code}"
        position = positions[int(player["element_type"])]
        ownership = float(player.get("selected_by_percent") or 0)
        selected = ownership * total_players / 100 if total_players else np.nan
        availability = _availability_probability(player)
        live_players[str(player_id)] = {
            "id": player_id,
            "code": player_code,
            "player_key": player_key,
            "web_name": str(player["web_name"]),
            "first_name": str(player.get("first_name") or ""),
            "second_name": str(player.get("second_name") or ""),
            "position": position,
            "team_id": team_id,
            "team_code": int(team["code"]),
            "team_name": str(team["name"]),
            "status": str(player.get("status") or ""),
            "availability_probability": availability,
            "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
            "news": str(player.get("news") or ""),
            "news_added": player.get("news_added"),
            "price_millions": float(player["now_cost"]) / 10,
            "ownership_percent": ownership,
        }
        for fixture in fixtures_by_team.get(team_id, []):
            is_home = team_id == int(fixture["team_h"])
            opponent_id = int(fixture["team_a"] if is_home else fixture["team_h"])
            opponent = teams[opponent_id]
            row: dict[str, object] = {
                "season": season,
                "season_start": int(season.split("-", 1)[0]),
                "fixture_id": int(fixture["id"]),
                "gameweek": fixture.get("event"),
                "kickoff_time": _as_utc(fixture["kickoff_time"]),
                "player_id": player_id,
                "player_code": player_code,
                "player_key": player_key,
                "player_name": str(player["web_name"]),
                "position": position,
                "team_id": team_id,
                "team_code": int(team["code"]),
                "team_key": f"code:{int(team['code'])}",
                "team_name": str(team["name"]),
                "opponent_team_id": opponent_id,
                "opponent_team_code": int(opponent["code"]),
                "opponent_team_key": f"code:{int(opponent['code'])}",
                "opponent_team_name": str(opponent["name"]),
                "is_home": int(is_home),
                "home_team_id": int(fixture["team_h"]),
                "away_team_id": int(fixture["team_a"]),
                "home_goals": np.nan,
                "away_goals": np.nan,
                "fpl_expected_points": pd.to_numeric(player.get("ep_next"), errors="coerce"),
                "price_tenths": float(player["now_cost"]),
                "selected": selected,
                "transfers_balance": float(player.get("transfers_in_event") or 0) - float(player.get("transfers_out_event") or 0),
                "transfers_in": float(player.get("transfers_in_event") or 0),
                "transfers_out": float(player.get("transfers_out_event") or 0),
                "live_status": str(player.get("status") or ""),
                "live_availability_probability": availability,
                "live_chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
                "live_news": str(player.get("news") or ""),
                "live_news_added": player.get("news_added"),
                "live_price_millions": float(player["now_cost"]) / 10,
                "live_ownership_percent": ownership,
            }
            row.update({column: np.nan for column in OUTCOME_COLUMNS})
            rows.append(row)
    future = pd.DataFrame(rows).sort_values(
        ["kickoff_time", "fixture_id", "team_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    if future.duplicated(["season", "fixture_id", "player_key"]).any():
        raise ValueError("Live FPL data produced duplicate player-fixture rows")
    fixture_context = [
        {
            "id": int(fixture["id"]),
            "gameweek": fixture.get("event"),
            "kickoff_time": fixture.get("kickoff_time"),
            "home_team_id": int(fixture["team_h"]),
            "away_team_id": int(fixture["team_a"]),
        }
        for fixture in remaining
    ]
    context = {
        "retrieved_at": retrieved.isoformat(),
        "season": season,
        "players": live_players,
        "fixtures": fixture_context,
        "skipped_fixtures": skipped,
    }
    return future, context


def _anchor_rows(
    player_snapshots: pd.DataFrame,
    team_ids: list[int],
    team_details: dict[int, dict[str, object]],
    *,
    reverse: bool,
    kickoff: pd.Timestamp,
    gameweek: float,
) -> pd.DataFrame:
    anchors = []
    for pair_number in range(0, len(team_ids), 2):
        first, second = team_ids[pair_number : pair_number + 2]
        home_id, away_id = (second, first) if reverse else (first, second)
        fixture_id = -(1000 + pair_number + int(reverse))
        fixture_kickoff = kickoff + pd.Timedelta(seconds=pair_number)
        for team_id, opponent_id, is_home in ((home_id, away_id, 1), (away_id, home_id, 0)):
            team = team_details[team_id]
            opponent = team_details[opponent_id]
            players = player_snapshots.loc[player_snapshots["team_id"].eq(team_id)].copy()
            players["fixture_id"] = fixture_id
            players["gameweek"] = gameweek
            players["kickoff_time"] = fixture_kickoff
            players["team_id"] = team_id
            players["team_code"] = team["team_code"]
            players["team_key"] = team["team_key"]
            players["team_name"] = team["team_name"]
            players["opponent_team_id"] = opponent_id
            players["opponent_team_code"] = opponent["team_code"]
            players["opponent_team_key"] = opponent["team_key"]
            players["opponent_team_name"] = opponent["team_name"]
            players["is_home"] = is_home
            players["home_team_id"] = home_id
            players["away_team_id"] = away_id
            anchors.append(players)
    return pd.concat(anchors, ignore_index=True)


def _scheduled_rest_days(
    future: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    entity_key: str,
) -> pd.Series:
    last_kickoff = historical.groupby(entity_key)["kickoff_time"].max().to_dict()
    result = pd.Series(index=future.index, dtype=float)
    for entity, group in future.sort_values(["kickoff_time", "fixture_id"]).groupby(entity_key, sort=False):
        previous = last_kickoff.get(entity)
        for index, row in group.iterrows():
            kickoff = row["kickoff_time"]
            result.loc[index] = (
                (kickoff - previous).total_seconds() / 86400 if previous is not None and not pd.isna(previous) else np.nan
            )
            previous = kickoff
    return result


def build_future_feature_rows(
    historical_rows: pd.DataFrame,
    future_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze form at the live cut-off and add schedule-aware future context."""
    historical = historical_rows.copy()
    historical["kickoff_time"] = pd.to_datetime(historical["kickoff_time"], utc=True)
    future = future_rows.copy()
    future["kickoff_time"] = pd.to_datetime(future["kickoff_time"], utc=True)
    if future.empty:
        raise ValueError("At least one future player-fixture row is required")
    team_ids = sorted(int(value) for value in future["team_id"].unique())
    if len(team_ids) % 2:
        raise ValueError("The live competition must contain an even number of teams")
    team_details = {
        int(row["team_id"]): {
            "team_code": int(row["team_code"]),
            "team_key": row["team_key"],
            "team_name": row["team_name"],
        }
        for _, row in future.drop_duplicates("team_id").iterrows()
    }
    snapshot_columns = [
        "season", "season_start", "player_id", "player_code", "player_key",
        "player_name", "position", "team_id", "team_code", "team_key", "team_name",
        "fpl_expected_points", "price_tenths", "selected", "transfers_balance",
        "transfers_in", "transfers_out", "home_goals", "away_goals", *OUTCOME_COLUMNS,
    ]
    snapshots = future.drop_duplicates("player_key", keep="first")[snapshot_columns]
    anchor_kickoff = future["kickoff_time"].min()
    assigned_gameweeks = pd.to_numeric(future["gameweek"], errors="coerce").dropna()
    anchor_gameweek = float(pd.to_numeric(historical["gameweek"], errors="coerce").max() + 100)
    first_pass = _anchor_rows(
        snapshots, team_ids, team_details,
        reverse=False, kickoff=anchor_kickoff, gameweek=anchor_gameweek,
    )
    second_pass = _anchor_rows(
        snapshots, team_ids, team_details,
        reverse=True, kickoff=anchor_kickoff, gameweek=anchor_gameweek,
    )
    anchor_source = pd.concat([first_pass, second_pass], ignore_index=True)
    built = build_training_dataset(pd.concat([historical, anchor_source], ignore_index=True))
    anchors = built.loc[built["fixture_id"].lt(0)].copy()

    player_feature_columns = [column for column in anchors if column.startswith("feature_player_") or column.startswith("feature_price") or column.startswith("feature_ownership") or column.startswith("feature_fpl_") or column.startswith("feature_selected") or column.startswith("feature_transfers") or column.startswith("feature_manager") or column.startswith("feature_team_position_")]
    player_templates = anchors.drop_duplicates("player_key", keep="first")[["player_key"] + player_feature_columns]
    team_feature_columns = [
        column for column in anchors
        if column.startswith("feature_team_") and not column.startswith("feature_team_position_")
    ]
    team_anchor_rows = anchors.sort_values(["kickoff_time", "fixture_id"]).drop_duplicates(
        ["team_key", "is_home"], keep="first"
    )
    venue_columns = [column for column in team_feature_columns if "_venue_" in column]
    general_columns = [column for column in team_feature_columns if column not in venue_columns]
    team_general = team_anchor_rows.drop_duplicates("team_key", keep="first")[["team_key"] + general_columns]
    team_venue = team_anchor_rows[["team_key", "is_home"] + venue_columns]
    team_templates = team_venue.merge(team_general, on="team_key", how="left", validate="many_to_one")

    identifier_columns = [
        "season", "season_start", "fixture_id", "gameweek", "kickoff_time",
        "player_id", "player_code", "player_key", "player_name", "position",
        "team_id", "team_code", "team_key", "team_name", "opponent_team_id",
        "opponent_team_code", "opponent_team_key", "opponent_team_name", "is_home",
    ]
    output = future[identifier_columns + LIVE_PLAYER_COLUMNS].copy()
    output = output.merge(player_templates, on="player_key", how="left", validate="many_to_one")
    output = output.merge(team_templates, on=["team_key", "is_home"], how="left", validate="many_to_one")
    opponent_templates = team_templates.rename(columns={
        "team_key": "opponent_team_key",
        "is_home": "opponent_is_home",
        **{column: column.replace("feature_team_", "feature_opponent_", 1) for column in team_feature_columns},
    })
    output["opponent_is_home"] = 1 - output["is_home"]
    output = output.merge(opponent_templates, on=["opponent_team_key", "opponent_is_home"], how="left", validate="many_to_one").drop(columns="opponent_is_home")

    player_schedule = future[["season", "fixture_id", "player_key", "kickoff_time"]].copy()
    player_schedule["feature_player_rest_days"] = _scheduled_rest_days(
        player_schedule, historical, entity_key="player_key"
    )
    output = output.drop(columns="feature_player_rest_days", errors="ignore").merge(
        player_schedule[["season", "fixture_id", "player_key", "feature_player_rest_days"]],
        on=["season", "fixture_id", "player_key"], how="left", validate="one_to_one",
    )
    historical_teams = historical[["season", "fixture_id", "kickoff_time", "team_key"]].drop_duplicates()
    team_future = future[["season", "fixture_id", "team_key", "kickoff_time"]].drop_duplicates()
    team_future["feature_team_rest_days"] = _scheduled_rest_days(
        team_future, historical_teams, entity_key="team_key"
    )
    output = output.drop(columns="feature_team_rest_days", errors="ignore").merge(
        team_future[["season", "fixture_id", "team_key", "feature_team_rest_days"]],
        on=["season", "fixture_id", "team_key"], how="left", validate="many_to_one",
    )
    opponent_rest = output[["season", "fixture_id", "opponent_team_key"]].merge(
        output[["season", "fixture_id", "team_key", "feature_team_rest_days"]].drop_duplicates(),
        left_on=["season", "fixture_id", "opponent_team_key"],
        right_on=["season", "fixture_id", "team_key"],
        how="left",
        validate="many_to_one",
    )["feature_team_rest_days"]
    output["feature_opponent_rest_days"] = opponent_rest.to_numpy()
    schedule = output[["season", "fixture_id", "gameweek", "kickoff_time", "team_key"]].drop_duplicates()
    schedule = schedule.sort_values(["season", "gameweek", "team_key", "kickoff_time", "fixture_id"])
    schedule["fixture_count"] = schedule.groupby(["season", "gameweek", "team_key"], dropna=False)["fixture_id"].transform("size")
    schedule["fixture_number"] = schedule.groupby(["season", "gameweek", "team_key"], dropna=False).cumcount() + 1
    output = output.drop(columns=["feature_team_gameweek_fixture_count", "feature_team_gameweek_fixture_number"], errors="ignore").merge(
        schedule[["season", "fixture_id", "team_key", "fixture_count", "fixture_number"]],
        on=["season", "fixture_id", "team_key"], how="left", validate="many_to_one",
    ).rename(columns={"fixture_count": "feature_team_gameweek_fixture_count", "fixture_number": "feature_team_gameweek_fixture_number"})
    opponent_schedule = schedule[["season", "fixture_id", "team_key", "fixture_count", "fixture_number"]].rename(columns={
        "team_key": "opponent_team_key", "fixture_count": "feature_opponent_gameweek_fixture_count", "fixture_number": "feature_opponent_gameweek_fixture_number",
    })
    output = output.drop(columns=["feature_opponent_gameweek_fixture_count", "feature_opponent_gameweek_fixture_number"], errors="ignore").merge(
        opponent_schedule, on=["season", "fixture_id", "opponent_team_key"], how="left", validate="many_to_one"
    )
    if output.duplicated(["season", "fixture_id", "player_key"]).any():
        raise ValueError("Future feature builder produced duplicate player-fixture rows")
    feature_columns = [column for column in output if column.startswith("feature_")]
    if output[feature_columns].isna().all(axis=1).any():
        raise ValueError("Future feature builder produced a row without historical features")
    return output.sort_values(["kickoff_time", "fixture_id", "player_id"]).reset_index(drop=True)
