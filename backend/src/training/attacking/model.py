"""Player xG, xA, goal, and FPL-assist prediction with team reconciliation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_MODEL_DIR = Path("data/models/attacking")
PLAYER_ROW_KEYS = ["season", "fixture_id", "player_key"]
TEAM_ROW_KEYS = ["season", "fixture_id", "team_key"]
IDENTIFIER_COLUMNS = [
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
    "team_key",
    "team_name",
    "opponent_team_key",
    "opponent_team_name",
    "is_home",
]
UPSTREAM_COLUMNS = ["expected_minutes", "team_expected_goals"]
TARGET_COLUMNS = [
    "target_outcome_known",
    "target_minutes",
    "target_goals_scored",
    "target_assists",
    "target_expected_goals",
    "target_expected_assists",
    "target_expected_goals_available",
    "target_expected_assists_available",
    "target_team_goals",
]

USAGE_FEATURES = [
    "gameweek",
    "is_home",
    "feature_player_history_fixtures",
    "feature_player_season_history_fixtures",
    "feature_player_history_minutes",
    "feature_player_season_history_minutes",
    "feature_player_previous_minutes",
    "feature_player_previous_appeared",
    "feature_player_previous_sixty_plus",
    "feature_player_rest_days",
    "feature_player_changed_team",
    "feature_player_is_new",
    "feature_player_is_new_season",
    "feature_player_is_returning",
    "feature_player_minutes_mean_3",
    "feature_player_appeared_rate_3",
    "feature_player_sixty_plus_rate_3",
    "feature_player_minutes_mean_5",
    "feature_player_appeared_rate_5",
    "feature_player_sixty_plus_rate_5",
    "feature_player_minutes_mean_10",
    "feature_player_appeared_rate_10",
    "feature_player_sixty_plus_rate_10",
    "feature_player_season_appeared_rate",
    "feature_player_season_sixty_plus_rate",
    "feature_player_previous_season_fixtures",
    "feature_player_previous_season_minutes_per_fixture",
    "feature_player_previous_season_appeared_rate",
    "feature_player_previous_season_sixty_plus_rate",
    "feature_team_rest_days",
    "feature_team_gameweek_fixture_count",
    "feature_team_gameweek_fixture_number",
]


def _player_rate_features(statistics: Sequence[str]) -> list[str]:
    features = []
    for statistic in statistics:
        features.extend(
            [f"feature_player_{statistic}_per90_{window}" for window in (3, 5, 10)]
        )
        features.extend(
            [
                f"feature_player_season_{statistic}_per90",
                f"feature_player_previous_season_{statistic}_per90",
            ]
        )
    return features


PLAYER_EVENT_FEATURES = _player_rate_features(
    ["goals_scored", "assists", "threat", "creativity", "ict_index"]
)
PLAYER_EXPECTED_FEATURES = _player_rate_features(
    ["expected_goals", "expected_assists", "expected_goal_involvements"]
) + [
    f"feature_player_{statistic}_{window}"
    for statistic in ("xg_observations",)
    for window in (3, 5, 10)
]


def _team_form_features(statistics: Sequence[str]) -> list[str]:
    features = []
    for side in ("team", "opponent"):
        for statistic in statistics:
            features.extend(
                [
                    f"feature_{side}_{statistic}_mean_{window}"
                    for window in (3, 5, 10)
                ]
            )
            features.extend(
                [
                    f"feature_{side}_{statistic}_venue_mean_5",
                    f"feature_{side}_season_{statistic}_mean",
                    f"feature_{side}_previous_season_{statistic}_mean",
                ]
            )
    return features


TEAM_EVENT_FEATURES = _team_form_features(["goals_for", "goals_against"])
TEAM_EXPECTED_FEATURES = _team_form_features(
    ["expected_goals_for", "expected_goals_against", "expected_assists"]
)
MARKET_FEATURES = [
    "feature_price_millions",
    "feature_ownership_percent",
    "feature_price_change_previous_snapshot",
    "feature_price_change_season",
    "feature_team_position_price_rank",
    "feature_team_position_ownership_rank",
]
EXPECTED_HISTORY_FEATURES = PLAYER_EXPECTED_FEATURES + TEAM_EXPECTED_FEATURES
FEATURE_PROFILES = {
    "event_history": {
        "numeric": USAGE_FEATURES + PLAYER_EVENT_FEATURES + TEAM_EVENT_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
    "expected_history": {
        "numeric": USAGE_FEATURES
        + PLAYER_EVENT_FEATURES
        + TEAM_EVENT_FEATURES
        + EXPECTED_HISTORY_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
    "expected_history_and_market": {
        "numeric": USAGE_FEATURES
        + PLAYER_EVENT_FEATURES
        + TEAM_EVENT_FEATURES
        + EXPECTED_HISTORY_FEATURES
        + MARKET_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
}


def get_feature_columns(profile: str) -> tuple[list[str], list[str]]:
    """Return numeric and categorical fields for a feature profile."""
    if profile not in FEATURE_PROFILES:
        choices = ", ".join(sorted(FEATURE_PROFILES))
        raise ValueError(f"Unknown feature profile {profile!r}; choose from {choices}")
    definition = FEATURE_PROFILES[profile]
    return list(definition["numeric"]), list(definition["categorical"])


def all_training_feature_columns() -> list[str]:
    """Return the union of fields required by every feature profile."""
    return sorted(
        {
            column
            for definition in FEATURE_PROFILES.values()
            for feature_type in ("numeric", "categorical")
            for column in definition[feature_type]
        }
    )


def load_attacking_training_data(path: str | Path) -> pd.DataFrame:
    """Load player-fixture rows and mask known false-zero expected-data history."""
    input_path = Path(path)
    required = list(
        dict.fromkeys(
            IDENTIFIER_COLUMNS + TARGET_COLUMNS + all_training_feature_columns()
        )
    )
    available = pd.read_csv(input_path, nrows=0).columns
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"Training dataset is missing columns: {', '.join(missing)}")

    rows = pd.read_csv(input_path, usecols=required, low_memory=False)
    rows = rows.loc[rows["target_outcome_known"].eq(1)].copy()
    if rows.empty:
        raise ValueError("Training dataset has no completed fixture outcomes")
    if rows.duplicated(PLAYER_ROW_KEYS).any():
        raise ValueError("Training dataset contains duplicate player-fixture rows")

    gameweek_expected_totals = rows.groupby(["season", "gameweek"])[
        ["target_expected_goals", "target_expected_assists"]
    ].transform("sum")
    rows["expected_data_available"] = (
        rows["target_expected_goals_available"].eq(1)
        & rows["target_expected_assists_available"].eq(1)
        & gameweek_expected_totals["target_expected_goals"].gt(0)
        & gameweek_expected_totals["target_expected_assists"].gt(0)
    )

    rows.loc[rows["season"].eq("2022-23"), EXPECTED_HISTORY_FEATURES] = np.nan
    previous_expected_features = [
        feature
        for feature in EXPECTED_HISTORY_FEATURES
        if "previous_season" in feature
    ]
    rows.loc[
        rows["season"].eq("2023-24"), previous_expected_features
    ] = np.nan
    rows = rows.sort_values(
        ["season_start", "kickoff_time", "fixture_id", "player_key"]
    ).reset_index(drop=True)
    return rows


def _preprocessor(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_rate_model(
    profile: str,
    parameters: Mapping[str, object],
    *,
    random_state: int,
) -> Pipeline:
    """Build a Poisson gradient-boosting model for an attacking rate per 90."""
    numeric_features, categorical_features = get_feature_columns(profile)
    return Pipeline(
        [
            ("features", _preprocessor(numeric_features, categorical_features)),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    early_stopping=False,
                    random_state=random_state,
                    **parameters,
                ),
            ),
        ]
    )


def position_rate_priors(
    rows: pd.DataFrame, target_column: str
) -> dict[str, float]:
    """Calculate exposure-weighted position rates per 90."""
    training = rows.loc[
        rows["expected_data_available"] & rows["target_minutes"].gt(0)
    ]
    grouped = training.groupby("position").agg(
        events=(target_column, "sum"), minutes=("target_minutes", "sum")
    )
    rates = (90 * grouped["events"] / grouped["minutes"]).to_dict()
    rates["__default__"] = float(
        90 * training[target_column].sum() / training["target_minutes"].sum()
    )
    return {str(position): float(rate) for position, rate in rates.items()}


def fpl_assists_per_team_goal(rows: pd.DataFrame) -> float:
    """Calculate the historical FPL-assist rate per team goal."""
    team_fixtures = rows.groupby(TEAM_ROW_KEYS, as_index=False).agg(
        assists=("target_assists", "sum"), goals=("target_team_goals", "first")
    )
    return float(team_fixtures["assists"].sum() / team_fixtures["goals"].sum())


def _prior_values(
    positions: pd.Series, priors: Mapping[str, float]
) -> np.ndarray:
    default = float(priors["__default__"])
    return positions.map(priors).fillna(default).to_numpy(dtype=float)


def build_attacking_predictions(
    rows: pd.DataFrame,
    raw_goal_rates: Sequence[float],
    raw_assist_rates: Sequence[float],
    goal_priors: Mapping[str, float],
    assist_priors: Mapping[str, float],
    *,
    shrinkage_minutes: float,
    assist_goal_ratio: float,
    minimum_rate: float,
    maximum_rate: float,
) -> pd.DataFrame:
    """Shrink rates, apply exposure, and reconcile player totals to team forecasts."""
    required = set(IDENTIFIER_COLUMNS).union(UPSTREAM_COLUMNS)
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Attacking input is missing columns: {', '.join(missing)}")
    if len(rows) != len(raw_goal_rates) or len(rows) != len(raw_assist_rates):
        raise ValueError("A goal and assist rate are required for every player row")
    if rows.duplicated(PLAYER_ROW_KEYS).any():
        raise ValueError("Attacking input contains duplicate player-fixture rows")
    if not rows["expected_minutes"].between(0, 90).all():
        raise ValueError("expected_minutes must be between zero and 90")

    team_goal_counts = rows.groupby(TEAM_ROW_KEYS)["team_expected_goals"].nunique()
    if not team_goal_counts.eq(1).all():
        raise ValueError("team_expected_goals must be constant within a team fixture")
    team_sizes = rows.groupby(TEAM_ROW_KEYS).size()
    if not team_sizes.ge(11).all():
        raise ValueError(
            "Team reconciliation requires the full player set for every team fixture"
        )

    output_columns = [column for column in IDENTIFIER_COLUMNS if column in rows]
    output = rows[output_columns].reset_index(drop=True).copy()
    expected_minutes = rows["expected_minutes"].to_numpy(dtype=float)
    history_minutes = rows["feature_player_history_minutes"].fillna(0).to_numpy(
        dtype=float
    )
    history_weight = np.divide(
        history_minutes,
        history_minutes + shrinkage_minutes,
        out=np.zeros_like(history_minutes),
        where=(history_minutes + shrinkage_minutes) > 0,
    )
    goal_prior_values = _prior_values(rows["position"], goal_priors)
    assist_prior_values = _prior_values(rows["position"], assist_priors)
    raw_goal_rates_array = np.clip(
        np.asarray(raw_goal_rates, dtype=float), minimum_rate, maximum_rate
    )
    raw_assist_rates_array = np.clip(
        np.asarray(raw_assist_rates, dtype=float), minimum_rate, maximum_rate
    )
    goal_rates = (
        history_weight * raw_goal_rates_array
        + (1 - history_weight) * goal_prior_values
    )
    assist_rates = (
        history_weight * raw_assist_rates_array
        + (1 - history_weight) * assist_prior_values
    )
    pre_reconciliation_xg = goal_rates * expected_minutes / 90
    expected_statistical_xa = assist_rates * expected_minutes / 90

    working = rows[TEAM_ROW_KEYS + ["team_expected_goals"]].reset_index(drop=True)
    working["pre_reconciliation_xg"] = pre_reconciliation_xg
    working["expected_statistical_xa"] = expected_statistical_xa
    goal_totals = working.groupby(TEAM_ROW_KEYS)[
        "pre_reconciliation_xg"
    ].transform("sum")
    assist_totals = working.groupby(TEAM_ROW_KEYS)[
        "expected_statistical_xa"
    ].transform("sum")
    team_expected_goals = working["team_expected_goals"].to_numpy(dtype=float)
    expected_goals = np.divide(
        pre_reconciliation_xg * team_expected_goals,
        goal_totals,
        out=np.zeros_like(pre_reconciliation_xg),
        where=goal_totals > 0,
    )
    expected_fpl_assists = np.divide(
        expected_statistical_xa * team_expected_goals * assist_goal_ratio,
        assist_totals,
        out=np.zeros_like(expected_statistical_xa),
        where=assist_totals > 0,
    )

    output["expected_minutes"] = expected_minutes
    output["team_expected_goals"] = team_expected_goals
    output["history_minutes"] = history_minutes
    output["history_weight"] = history_weight
    output["goal_position_prior_per90"] = goal_prior_values
    output["assist_position_prior_per90"] = assist_prior_values
    output["raw_goal_rate_per90"] = raw_goal_rates_array
    output["raw_xa_rate_per90"] = raw_assist_rates_array
    output["shrunk_goal_rate_per90"] = goal_rates
    output["shrunk_xa_rate_per90"] = assist_rates
    output["pre_reconciliation_xg"] = pre_reconciliation_xg
    output["expected_statistical_xa"] = expected_statistical_xa
    output["expected_goals"] = expected_goals
    output["goal_probability"] = 1 - np.exp(-expected_goals)
    output["expected_fpl_assists"] = expected_fpl_assists
    output["assist_probability"] = 1 - np.exp(-expected_fpl_assists)

    if set(TARGET_COLUMNS).issubset(rows.columns):
        output["actual_goals"] = rows["target_goals_scored"].to_numpy()
        output["actual_assists"] = rows["target_assists"].to_numpy()
        output["actual_xg"] = rows["target_expected_goals"].to_numpy()
        output["actual_xa"] = rows["target_expected_assists"].to_numpy()
        output["expected_data_available"] = rows[
            "expected_data_available"
        ].to_numpy()
    return output


def predict_with_artifact(
    artifact: Mapping[str, object], fixture_rows: pd.DataFrame
) -> pd.DataFrame:
    """Predict player attacking outputs using a saved artifact."""
    feature_columns = list(artifact["feature_columns"])
    required = set(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS + feature_columns)
    missing = sorted(required.difference(fixture_rows.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")
    goal_rates = artifact["goal_rate_model"].predict(
        fixture_rows[feature_columns]
    )
    assist_rates = artifact["assist_rate_model"].predict(
        fixture_rows[feature_columns]
    )
    return build_attacking_predictions(
        fixture_rows,
        goal_rates,
        assist_rates,
        artifact["goal_position_priors"],
        artifact["assist_position_priors"],
        shrinkage_minutes=float(artifact["shrinkage_minutes"]),
        assist_goal_ratio=float(artifact["assist_goal_ratio"]),
        minimum_rate=float(artifact["minimum_rate"]),
        maximum_rate=float(artifact["maximum_rate"]),
    )


def load_artifact(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, object]:
    """Load a trained attacking artifact."""
    return joblib.load(Path(model_dir) / "model.joblib")


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to records containing standard JSON values."""
    return [
        {key: _json_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Player-fixture JSON input.")
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR), help="Saved model directory."
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with Path(args.input).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    fixture_records = (
        payload.get("player_fixture_rows") if isinstance(payload, dict) else payload
    )
    if not isinstance(fixture_records, list):
        raise ValueError(
            "Input JSON must be a list or contain a player_fixture_rows list"
        )

    artifact = load_artifact(args.model_dir)
    predictions = predict_with_artifact(artifact, pd.DataFrame(fixture_records))
    rendered = json.dumps(
        {"predictions": records_for_json(predictions)},
        indent=2,
        allow_nan=False,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote attacking predictions to {output_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
