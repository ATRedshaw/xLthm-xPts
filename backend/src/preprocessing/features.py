"""Create leakage-safe features and targets for the component model pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .vaastav import OUTCOME_COLUMNS


ROLLING_WINDOWS = (3, 5, 10)

IDENTIFIER_COLUMNS = (
    "season",
    "season_start",
    "fixture_id",
    "gameweek",
    "kickoff_time",
    "player_id",
    "player_code",
    "player_key",
    "player_name",
    "position",
    "team_id",
    "team_code",
    "team_key",
    "team_name",
    "opponent_team_id",
    "opponent_team_code",
    "opponent_team_key",
    "opponent_team_name",
    "is_home",
)

PLAYER_BINARY_STATS = (
    "appeared",
    "started",
    "sixty_plus",
    "clean_sheets",
)

PLAYER_RATE_STATS = (
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
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

TEAM_STATS = (
    "goals_for",
    "goals_against",
    "clean_sheet",
    "expected_goals_for",
    "expected_goals_against",
    "expected_assists",
    "saves",
    "defensive_contribution",
    "bps",
    "yellow_cards",
    "red_cards",
    "total_points",
    "players_used",
    "starters",
    "players_sixty_plus",
)


def _group_values(rows: pd.DataFrame, columns: Sequence[str]) -> list[pd.Series]:
    return [rows[column] for column in columns]


def _shifted_rolling(
    rows: pd.DataFrame,
    source: pd.Series,
    groups: Sequence[str],
    window: int,
    aggregation: str,
) -> pd.Series:
    grouped = source.groupby(_group_values(rows, groups), sort=False, dropna=False)
    return grouped.transform(
        lambda values: getattr(
            values.shift(1).rolling(window, min_periods=1), aggregation
        )()
    )


def _shifted_expanding_sum(
    rows: pd.DataFrame, source: pd.Series, groups: Sequence[str]
) -> pd.Series:
    grouped = source.groupby(_group_values(rows, groups), sort=False, dropna=False)
    return grouped.transform(
        lambda values: values.shift(1).expanding(min_periods=1).sum()
    )


def _snapshot_features(rows: pd.DataFrame) -> pd.DataFrame:
    snapshot_values = [
        "price_tenths",
        "selected",
        "transfers_balance",
        "transfers_in",
        "transfers_out",
    ]
    snapshot_key = ["season", "gameweek", "player_key"]
    conflicts = rows.groupby(snapshot_key, dropna=False)[snapshot_values].nunique(
        dropna=False
    )
    if conflicts.gt(1).any(axis=None):
        raise ValueError("A player has conflicting snapshot values within a gameweek")

    snapshots = rows[
        snapshot_key + ["season_start", "team_key", "position"] + snapshot_values
    ].drop_duplicates(snapshot_key, keep="last")
    snapshots = snapshots.sort_values(
        ["season_start", "gameweek", "player_key"], kind="mergesort"
    ).reset_index(drop=True)

    manager_counts = (
        snapshots.groupby(["season", "gameweek"], as_index=False)["selected"]
        .sum(min_count=1)
        .rename(columns={"selected": "selected_total"})
    )
    manager_counts["manager_count"] = manager_counts["selected_total"] / 15.0
    manager_counts["manager_count"] = manager_counts.groupby("season")[
        "manager_count"
    ].cummax()
    snapshots = snapshots.merge(
        manager_counts[["season", "gameweek", "manager_count"]],
        on=["season", "gameweek"],
        how="left",
        validate="many_to_one",
    )

    snapshots["ownership_percent"] = (
        100.0 * snapshots["selected"] / snapshots["manager_count"]
    ).clip(0, 100)
    snapshots["transfers_in_percent"] = (
        100.0 * snapshots["transfers_in"] / snapshots["manager_count"]
    )
    snapshots["transfers_out_percent"] = (
        100.0 * snapshots["transfers_out"] / snapshots["manager_count"]
    )
    snapshots["transfers_balance_percent"] = (
        100.0 * snapshots["transfers_balance"] / snapshots["manager_count"]
    )
    snapshots["price_millions"] = snapshots["price_tenths"] / 10.0

    season_player = snapshots.groupby(["season", "player_key"], sort=False)
    snapshots["price_change_previous_snapshot"] = season_player[
        "price_tenths"
    ].diff()
    snapshots["price_change_season"] = (
        snapshots["price_tenths"]
        - season_player["price_tenths"].transform("first")
    )

    team_position = snapshots.groupby(
        ["season", "gameweek", "team_key", "position"], dropna=False
    )
    snapshots["team_position_price_rank"] = team_position["price_tenths"].rank(
        method="average", ascending=False, pct=True
    )
    snapshots["team_position_ownership_rank"] = team_position[
        "ownership_percent"
    ].rank(method="average", ascending=False, pct=True)

    feature_names = snapshot_values + [
        "manager_count",
        "ownership_percent",
        "transfers_in_percent",
        "transfers_out_percent",
        "transfers_balance_percent",
        "price_millions",
        "price_change_previous_snapshot",
        "price_change_season",
        "team_position_price_rank",
        "team_position_ownership_rank",
    ]
    snapshots = snapshots[snapshot_key + feature_names].rename(
        columns={name: f"feature_{name}" for name in feature_names}
    )
    return rows[snapshot_key].merge(
        snapshots, on=snapshot_key, how="left", validate="many_to_one"
    ).drop(columns=snapshot_key)


def _previous_season_player_features(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["player_key", "season_start"]
    grouped = rows.groupby(group_columns, dropna=False)
    summary = grouped.size().rename("feature_player_previous_season_fixtures").to_frame()
    summary["feature_player_previous_season_minutes"] = grouped["minutes"].sum(
        min_count=1
    )
    summary["feature_player_previous_season_minutes_per_fixture"] = grouped[
        "minutes"
    ].mean()
    for stat in PLAYER_BINARY_STATS:
        summary[f"feature_player_previous_season_{stat}_rate"] = grouped[stat].mean()
    for stat in PLAYER_RATE_STATS:
        total = grouped[stat].sum(min_count=1)
        summary[f"feature_player_previous_season_{stat}_total"] = total
        available_minutes = rows["minutes"].where(rows[stat].notna())
        minutes = available_minutes.groupby(
            _group_values(rows, group_columns), dropna=False
        ).sum(min_count=1)
        summary[f"feature_player_previous_season_{stat}_per90"] = (
            90.0 * total / minutes.replace(0, np.nan)
        )
    summary = summary.reset_index()
    summary["season_start"] += 1
    keys = rows[["player_key", "season_start"]]
    return keys.merge(
        summary,
        on=["player_key", "season_start"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["player_key", "season_start"])


def _player_features(rows: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    player_group = ["player_key"]
    season_player_group = ["season", "player_key"]

    features["feature_player_history_fixtures"] = rows.groupby(
        "player_key", sort=False
    ).cumcount()
    features["feature_player_season_history_fixtures"] = rows.groupby(
        season_player_group, sort=False
    ).cumcount()
    features["feature_player_history_minutes"] = _shifted_expanding_sum(
        rows, rows["minutes"], player_group
    )
    features["feature_player_season_history_minutes"] = _shifted_expanding_sum(
        rows, rows["minutes"], season_player_group
    )

    player_groups = _group_values(rows, player_group)
    for source, name in (
        ("minutes", "minutes"),
        ("appeared", "appeared"),
        ("started", "started"),
        ("sixty_plus", "sixty_plus"),
    ):
        features[f"feature_player_previous_{name}"] = rows[source].groupby(
            player_groups, sort=False, dropna=False
        ).shift()

    previous_kickoff = rows["kickoff_time"].groupby(
        player_groups, sort=False, dropna=False
    ).shift()
    features["feature_player_rest_days"] = (
        rows["kickoff_time"] - previous_kickoff
    ).dt.total_seconds().div(86400)
    previous_team = rows["team_key"].groupby(
        player_groups, sort=False, dropna=False
    ).shift()
    features["feature_player_changed_team"] = (
        previous_team.notna() & previous_team.ne(rows["team_key"])
    ).astype(int)
    features["feature_player_is_new"] = (
        features["feature_player_history_fixtures"] == 0
    ).astype(int)
    features["feature_player_is_new_season"] = (
        features["feature_player_season_history_fixtures"] == 0
    ).astype(int)
    features["feature_player_is_returning"] = (
        (features["feature_player_is_new_season"] == 1)
        & (features["feature_player_history_fixtures"] > 0)
    ).astype(int)

    for window in ROLLING_WINDOWS:
        minute_sum = _shifted_rolling(
            rows, rows["minutes"], player_group, window, "sum"
        )
        features[f"feature_player_minutes_sum_{window}"] = minute_sum
        features[f"feature_player_minutes_mean_{window}"] = _shifted_rolling(
            rows, rows["minutes"], player_group, window, "mean"
        )
        for stat in PLAYER_BINARY_STATS:
            features[f"feature_player_{stat}_rate_{window}"] = _shifted_rolling(
                rows, rows[stat], player_group, window, "mean"
            )
        for stat in PLAYER_RATE_STATS:
            total = _shifted_rolling(rows, rows[stat], player_group, window, "sum")
            available_minutes = rows["minutes"].where(rows[stat].notna())
            minutes = _shifted_rolling(
                rows, available_minutes, player_group, window, "sum"
            )
            features[f"feature_player_{stat}_sum_{window}"] = total
            features[f"feature_player_{stat}_per90_{window}"] = (
                90.0 * total / minutes.replace(0, np.nan)
            )
        features[f"feature_player_xg_observations_{window}"] = _shifted_rolling(
            rows,
            rows["expected_goals"].notna().astype(int),
            player_group,
            window,
            "sum",
        )
        features[
            f"feature_player_defensive_observations_{window}"
        ] = _shifted_rolling(
            rows,
            rows["defensive_contribution"].notna().astype(int),
            player_group,
            window,
            "sum",
        )

    for stat in PLAYER_BINARY_STATS:
        prior_total = _shifted_expanding_sum(rows, rows[stat], season_player_group)
        fixtures = features["feature_player_season_history_fixtures"].replace(0, np.nan)
        features[f"feature_player_season_{stat}_rate"] = prior_total / fixtures
    for stat in PLAYER_RATE_STATS:
        total = _shifted_expanding_sum(rows, rows[stat], season_player_group)
        available_minutes = rows["minutes"].where(rows[stat].notna())
        minutes = _shifted_expanding_sum(rows, available_minutes, season_player_group)
        features[f"feature_player_season_{stat}_total"] = total
        features[f"feature_player_season_{stat}_per90"] = (
            90.0 * total / minutes.replace(0, np.nan)
        )

    feature_frame = pd.DataFrame(features, index=rows.index)
    return pd.concat(
        [feature_frame, _previous_season_player_features(rows)], axis=1
    )


def _team_fixture_rows(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "season",
        "season_start",
        "fixture_id",
        "gameweek",
        "kickoff_time",
        "team_key",
        "opponent_team_key",
        "is_home",
    ]
    grouped = rows.groupby(group_columns, dropna=False)
    teams = grouped.agg(
        home_goals=("home_goals", "first"),
        away_goals=("away_goals", "first"),
        players_used=("appeared", "sum"),
        starters=("started", "sum"),
        players_sixty_plus=("sixty_plus", "sum"),
    ).reset_index()

    summed = grouped[
        [
            "expected_goals",
            "expected_assists",
            "saves",
            "defensive_contribution",
            "bps",
            "yellow_cards",
            "red_cards",
            "total_points",
        ]
    ].sum(min_count=1).reset_index()
    teams = teams.merge(summed, on=group_columns, validate="one_to_one")
    teams["goals_for"] = teams["home_goals"].where(
        teams["is_home"].eq(1), teams["away_goals"]
    )
    teams["goals_against"] = teams["away_goals"].where(
        teams["is_home"].eq(1), teams["home_goals"]
    )
    teams["clean_sheet"] = teams["goals_against"].eq(0).astype(int)
    teams = teams.rename(columns={"expected_goals": "expected_goals_for"})

    opponent_xg = teams[
        ["season", "fixture_id", "team_key", "expected_goals_for"]
    ].rename(
        columns={
            "team_key": "opponent_team_key",
            "expected_goals_for": "expected_goals_against",
        }
    )
    teams = teams.merge(
        opponent_xg,
        on=["season", "fixture_id", "opponent_team_key"],
        how="left",
        validate="many_to_one",
    )
    fixture_team_counts = teams.groupby(["season", "fixture_id"]).size()
    if not fixture_team_counts.eq(2).all():
        raise ValueError("Every fixture must contain player rows for both teams")
    return teams.sort_values(
        ["team_key", "kickoff_time", "fixture_id"], kind="mergesort"
    ).reset_index(drop=True)


def _previous_season_team_features(teams: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["team_key", "season_start"]
    grouped = teams.groupby(group_columns, dropna=False)
    summary = grouped.size().rename("feature_team_previous_season_fixtures").to_frame()
    for stat in TEAM_STATS:
        summary[f"feature_team_previous_season_{stat}_total"] = grouped[stat].sum(
            min_count=1
        )
        summary[f"feature_team_previous_season_{stat}_mean"] = grouped[stat].mean()
    summary = summary.reset_index()
    summary["season_start"] += 1
    return teams[["team_key", "season_start"]].merge(
        summary,
        on=["team_key", "season_start"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["team_key", "season_start"])


def _team_features(teams: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    team_group = ["team_key"]
    season_team_group = ["season", "team_key"]
    venue_team_group = ["team_key", "is_home"]

    features["feature_team_history_fixtures"] = teams.groupby(
        "team_key", sort=False
    ).cumcount()
    features["feature_team_season_history_fixtures"] = teams.groupby(
        season_team_group, sort=False
    ).cumcount()
    previous_kickoff = teams["kickoff_time"].groupby(
        _group_values(teams, team_group), sort=False, dropna=False
    ).shift()
    features["feature_team_rest_days"] = (
        teams["kickoff_time"] - previous_kickoff
    ).dt.total_seconds().div(86400)
    features["feature_team_is_new"] = (
        features["feature_team_history_fixtures"] == 0
    ).astype(int)
    features["feature_team_is_new_season"] = (
        features["feature_team_season_history_fixtures"] == 0
    ).astype(int)

    for window in ROLLING_WINDOWS:
        for stat in TEAM_STATS:
            features[f"feature_team_{stat}_sum_{window}"] = _shifted_rolling(
                teams, teams[stat], team_group, window, "sum"
            )
            features[f"feature_team_{stat}_mean_{window}"] = _shifted_rolling(
                teams, teams[stat], team_group, window, "mean"
            )
    for stat in TEAM_STATS:
        features[f"feature_team_{stat}_venue_mean_5"] = _shifted_rolling(
            teams, teams[stat], venue_team_group, 5, "mean"
        )
        prior_total = _shifted_expanding_sum(
            teams, teams[stat], season_team_group
        )
        fixtures = features["feature_team_season_history_fixtures"].replace(
            0, np.nan
        )
        features[f"feature_team_season_{stat}_total"] = prior_total
        features[f"feature_team_season_{stat}_mean"] = prior_total / fixtures

    features["feature_team_gameweek_fixture_count"] = teams.groupby(
        ["season", "gameweek", "team_key"], sort=False
    )["fixture_id"].transform("size")
    features["feature_team_gameweek_fixture_number"] = (
        teams.groupby(["season", "gameweek", "team_key"], sort=False).cumcount()
        + 1
    )
    feature_frame = pd.DataFrame(features, index=teams.index)
    return pd.concat(
        [feature_frame, _previous_season_team_features(teams)], axis=1
    )


def _add_team_context(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = _team_fixture_rows(rows)
    features = _team_features(teams)
    team_keys = teams[["season", "fixture_id", "team_key"]]
    team_context = pd.concat([team_keys, features], axis=1)

    opponent_context = team_context.rename(
        columns={
            "team_key": "opponent_team_key",
            **{
                column: column.replace("feature_team_", "feature_opponent_", 1)
                for column in features.columns
            },
        }
    )
    row_keys = rows[["season", "fixture_id", "team_key", "opponent_team_key"]]
    team_features = row_keys.merge(
        team_context,
        on=["season", "fixture_id", "team_key"],
        how="left",
        validate="many_to_one",
    ).merge(
        opponent_context,
        on=["season", "fixture_id", "opponent_team_key"],
        how="left",
        validate="many_to_one",
    )
    team_features = team_features.drop(
        columns=["season", "fixture_id", "team_key", "opponent_team_key"]
    )

    team_targets = teams[
        [
            "season",
            "fixture_id",
            "team_key",
            "goals_for",
            "goals_against",
            "clean_sheet",
            "expected_goals_for",
            "expected_goals_against",
        ]
    ].rename(
        columns={
            "goals_for": "target_team_goals",
            "goals_against": "target_opponent_goals",
            "clean_sheet": "target_team_clean_sheet",
            "expected_goals_for": "target_team_expected_goals",
            "expected_goals_against": "target_opponent_expected_goals",
        }
    )
    targets = row_keys[["season", "fixture_id", "team_key"]].merge(
        team_targets,
        on=["season", "fixture_id", "team_key"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["season", "fixture_id", "team_key"])
    return team_features, targets


def _targets(rows: pd.DataFrame, team_targets: pd.DataFrame) -> pd.DataFrame:
    targets = pd.DataFrame(index=rows.index)
    targets["target_outcome_known"] = rows["minutes"].notna().astype(int)
    targets["target_appeared"] = rows["appeared"]
    targets["target_started"] = rows["started"]
    targets["target_sixty_plus"] = rows["sixty_plus"]
    for column in OUTCOME_COLUMNS:
        targets[f"target_{column}"] = rows[column]
    targets["target_expected_goals_available"] = rows["expected_goals"].notna().astype(
        int
    )
    targets["target_expected_assists_available"] = rows[
        "expected_assists"
    ].notna().astype(int)
    targets["target_defensive_contribution_available"] = rows[
        "defensive_contribution"
    ].notna().astype(int)
    return pd.concat([targets, team_targets], axis=1)


def build_training_dataset(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Build the shared all-model training table at player-fixture grain.

    Every ``feature_`` column uses only the current pre-match FPL snapshot or
    fixtures before the target kickoff. Outcome columns are retained only as
    ``target_`` columns so each component trainer can select its own target and
    availability flag without rebuilding the shared history.
    """

    required = set(IDENTIFIER_COLUMNS).union(
        {
            "home_goals",
            "away_goals",
            "price_tenths",
            "selected",
            "transfers_balance",
            "transfers_in",
            "transfers_out",
            *OUTCOME_COLUMNS,
        }
    )
    missing = sorted(required.difference(raw_rows.columns))
    if missing:
        raise ValueError(f"Preprocessing input is missing columns: {', '.join(missing)}")

    rows = raw_rows.copy()
    key = ["season", "player_id", "fixture_id"]
    if rows.duplicated(key).any():
        raise ValueError("Preprocessing input contains duplicate player fixtures")
    rows["kickoff_time"] = pd.to_datetime(rows["kickoff_time"], utc=True, errors="coerce")
    if rows["kickoff_time"].isna().any():
        raise ValueError("Every row must have a valid kickoff time")

    numeric_columns = [
        "price_tenths",
        "selected",
        "transfers_balance",
        "transfers_in",
        "transfers_out",
        "home_goals",
        "away_goals",
        *OUTCOME_COLUMNS,
    ]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["appeared"] = rows["minutes"].gt(0).where(rows["minutes"].notna()).astype(
        "Float64"
    )
    rows["started"] = rows["starts"].gt(0).where(rows["starts"].notna()).astype(
        "Float64"
    )
    rows["sixty_plus"] = rows["minutes"].ge(60).where(
        rows["minutes"].notna()
    ).astype("Float64")
    known_minutes = rows["minutes"].dropna()
    if ((known_minutes < 0) | (known_minutes > 120)).any():
        raise ValueError("Historical minutes must be between 0 and 120")

    rows = rows.sort_values(
        ["player_key", "kickoff_time", "fixture_id"], kind="mergesort"
    ).reset_index(drop=True)
    snapshot_features = _snapshot_features(rows)
    player_features = _player_features(rows)
    team_features, team_targets = _add_team_context(rows)
    targets = _targets(rows, team_targets)

    dataset = pd.concat(
        [
            rows[list(IDENTIFIER_COLUMNS)],
            snapshot_features,
            player_features,
            team_features,
            targets,
        ],
        axis=1,
    )
    return dataset.sort_values(
        ["kickoff_time", "season", "fixture_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
