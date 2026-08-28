"""Download and normalise Vaastav FPL archive data without storing raw files."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/vaastav/"
    "Fantasy-Premier-League/master/data"
)
POSITION_NAMES = {
    "1": "GK",
    "2": "DEF",
    "3": "MID",
    "4": "FWD",
    "GKP": "GK",
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

SOURCE_FILES = {
    "player_fixtures": "gws/merged_gw.csv",
    "players": "players_raw.csv",
    "teams": "teams.csv",
    "fixtures": "fixtures.csv",
}

REQUIRED_COLUMNS = {
    "player_fixtures": {
        "element",
        "fixture",
        "GW",
        "kickoff_time",
        "minutes",
        "opponent_team",
        "position",
        "was_home",
    },
    "players": {"id", "code"},
    "teams": {"id", "code", "name"},
    "fixtures": {
        "id",
        "event",
        "kickoff_time",
        "team_h",
        "team_a",
        "team_h_score",
        "team_a_score",
    },
}

SNAPSHOT_COLUMNS = {
    "xP": "fpl_expected_points",
    "value": "price_tenths",
    "selected": "selected",
    "transfers_balance": "transfers_balance",
    "transfers_in": "transfers_in",
    "transfers_out": "transfers_out",
}

OUTCOME_COLUMNS = (
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "clean_sheets",
    "goals_conceded",
    "expected_goals_conceded",
    "saves",
    "penalties_saved",
    "bonus",
    "bps",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "creativity",
    "influence",
    "threat",
    "ict_index",
    "total_points",
)


@dataclass(slots=True)
class SeasonData:
    season: str
    player_fixtures: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    fixtures: pd.DataFrame


def fetch_csv(url: str, *, timeout: float = 30.0) -> pd.DataFrame:
    """Read one source CSV into memory."""

    request = Request(url, headers={"User-Agent": "fpl-xpts-preprocessing/1.0"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310 - configured URL
        content = response.read()
    return pd.read_csv(io.BytesIO(content), low_memory=False)


def fetch_season(
    season: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    progress: Callable[[str], None] | None = None,
) -> SeasonData:
    """Fetch one season directly from Vaastav without a local raw-data store."""

    frames: dict[str, pd.DataFrame] = {}
    for name, filename in SOURCE_FILES.items():
        if progress:
            progress(f"[{season}] downloading {filename}")
        url = f"{base_url.rstrip('/')}/{season}/{filename}"
        frames[name] = fetch_csv(url, timeout=timeout)
    return SeasonData(season=season, **frames)


def _require_columns(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(REQUIRED_COLUMNS[name].difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _normalise_position(value: object) -> str:
    text = str(value).strip().upper()
    return POSITION_NAMES.get(text, "UNK")


def _remove_exact_duplicates(rows: pd.DataFrame, season: str) -> pd.DataFrame:
    rows = rows.drop_duplicates().copy()
    key = ["element", "fixture"]
    if rows.duplicated(key, keep=False).any():
        duplicate = rows.loc[rows.duplicated(key, keep=False), key].iloc[0]
        raise ValueError(
            f"{season} contains conflicting rows for player "
            f"{duplicate['element']} in fixture {duplicate['fixture']}"
        )
    return rows


def normalise_season(data: SeasonData) -> pd.DataFrame:
    """Return one canonical row per historical player fixture."""

    for name in REQUIRED_COLUMNS:
        _require_columns(getattr(data, name), name)

    rows = _remove_exact_duplicates(data.player_fixtures, data.season)
    players = data.players[["id", "code"]].rename(
        columns={"id": "element", "code": "player_code"}
    )
    if "opta_code" in data.players:
        players = players.join(data.players["opta_code"])
    players = players.drop_duplicates("element", keep="last")

    teams = data.teams[["id", "code", "name"]].rename(
        columns={"id": "team_id", "code": "team_code", "name": "team_name"}
    )
    teams = teams.drop_duplicates("team_id", keep="last")

    fixture_columns = [
        "id",
        "event",
        "kickoff_time",
        "team_h",
        "team_a",
        "team_h_score",
        "team_a_score",
    ]
    fixtures = data.fixtures[fixture_columns].rename(
        columns={
            "id": "fixture",
            "event": "fixture_gameweek",
            "kickoff_time": "fixture_kickoff_time",
            "team_h_score": "fixture_home_goals",
            "team_a_score": "fixture_away_goals",
        }
    )
    fixtures = fixtures.drop_duplicates("fixture", keep="last")

    rows = rows.merge(players, on="element", how="left", validate="many_to_one")
    rows = rows.merge(fixtures, on="fixture", how="left", validate="many_to_one")
    if rows["team_h"].isna().any() or rows["team_a"].isna().any():
        raise ValueError(f"{data.season} contains player rows without a fixture")

    was_home = rows["was_home"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if was_home.isna().any():
        raise ValueError(f"{data.season} contains invalid was_home values")
    rows["is_home"] = was_home.astype(int)
    rows["team_id"] = rows["team_h"].where(was_home, rows["team_a"])
    rows["opponent_team_id"] = rows["team_a"].where(was_home, rows["team_h"])

    source_opponent = pd.to_numeric(rows["opponent_team"], errors="coerce")
    opponent_mismatch = source_opponent.notna() & source_opponent.ne(
        rows["opponent_team_id"]
    )
    if opponent_mismatch.any():
        raise ValueError(f"{data.season} contains inconsistent opponent team IDs")

    rows = rows.merge(teams, on="team_id", how="left", validate="many_to_one")
    opponent_teams = teams.rename(
        columns={
            "team_id": "opponent_team_id",
            "team_code": "opponent_team_code",
            "team_name": "opponent_team_name",
        }
    )
    rows = rows.merge(
        opponent_teams, on="opponent_team_id", how="left", validate="many_to_one"
    )

    output = pd.DataFrame(index=rows.index)
    output["season"] = data.season
    output["season_start"] = int(data.season.split("-", 1)[0])
    output["fixture_id"] = pd.to_numeric(rows["fixture"], errors="raise").astype(int)
    output["gameweek"] = pd.to_numeric(
        rows["fixture_gameweek"].fillna(rows["GW"]), errors="raise"
    ).astype(int)
    output["kickoff_time"] = pd.to_datetime(
        rows["fixture_kickoff_time"].fillna(rows["kickoff_time"]),
        utc=True,
        errors="coerce",
    )
    if output["kickoff_time"].isna().any():
        raise ValueError(f"{data.season} contains invalid kickoff times")

    output["player_id"] = pd.to_numeric(rows["element"], errors="raise").astype(int)
    output["player_code"] = pd.to_numeric(rows["player_code"], errors="coerce").astype(
        "Int64"
    )
    player_fallback = (
        "fpl:" + output["season"] + ":" + output["player_id"].astype(str)
    )
    output["player_key"] = player_fallback
    has_code = output["player_code"].notna()
    output.loc[has_code, "player_key"] = (
        "code:" + output.loc[has_code, "player_code"].astype(str)
    )
    output["player_name"] = rows["name"].astype(str)
    output["position"] = rows["position"].map(_normalise_position)

    for column in (
        "team_id",
        "team_code",
        "opponent_team_id",
        "opponent_team_code",
    ):
        output[column] = pd.to_numeric(rows[column], errors="coerce").astype("Int64")
    output["team_key"] = "code:" + output["team_code"].astype(str)
    output["opponent_team_key"] = "code:" + output["opponent_team_code"].astype(str)
    output["team_name"] = rows["team_name"]
    output["opponent_team_name"] = rows["opponent_team_name"]
    output["is_home"] = rows["is_home"].astype(int)
    output["home_team_id"] = pd.to_numeric(rows["team_h"], errors="coerce").astype(
        "Int64"
    )
    output["away_team_id"] = pd.to_numeric(rows["team_a"], errors="coerce").astype(
        "Int64"
    )
    output["home_goals"] = pd.to_numeric(rows["fixture_home_goals"], errors="coerce")
    output["away_goals"] = pd.to_numeric(rows["fixture_away_goals"], errors="coerce")

    for source, destination in SNAPSHOT_COLUMNS.items():
        values = rows[source] if source in rows else pd.Series(pd.NA, index=rows.index)
        output[destination] = pd.to_numeric(values, errors="coerce")
    output["transfers_balance"] = output["transfers_balance"].fillna(
        output["transfers_in"] - output["transfers_out"]
    )

    for column in OUTCOME_COLUMNS:
        values = rows[column] if column in rows else pd.Series(pd.NA, index=rows.index)
        output[column] = pd.to_numeric(values, errors="coerce")

    output = output[output["position"].isin({"GK", "DEF", "MID", "FWD"})]
    return output.sort_values(
        ["kickoff_time", "fixture_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)


def load_seasons(
    seasons: Sequence[str],
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fetch and combine canonical player-fixture rows for several seasons."""

    frames = [
        normalise_season(
            fetch_season(
                season,
                base_url=base_url,
                timeout=timeout,
                progress=progress,
            )
        )
        for season in seasons
    ]
    if not frames:
        raise ValueError("At least one season is required")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["kickoff_time", "season", "fixture_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
