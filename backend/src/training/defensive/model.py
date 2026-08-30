"""Defensive, clean-sheet, and goalkeeper component prediction."""

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


DEFAULT_MODEL_DIR = Path("data/models/defensive")
PLAYER_KEYS = ["season", "fixture_id", "player_key"]
IDENTIFIER_COLUMNS = [
    "season", "season_start", "fixture_id", "gameweek", "kickoff_time",
    "player_id", "player_code", "player_key", "player_name", "position",
    "team_key", "team_name", "opponent_team_key", "opponent_team_name", "is_home",
]
UPSTREAM_COLUMNS = [
    "expected_minutes", "sixty_plus_probability", "team_expected_goals_against",
    "team_clean_sheet_probability",
]
TARGET_COLUMNS = [
    "target_outcome_known", "target_minutes", "target_saves",
    "target_penalties_saved", "target_defensive_contribution",
    "target_defensive_contribution_available", "target_clean_sheets",
    "target_goals_conceded",
]


def _player_features(statistics: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for statistic in statistics:
        columns.extend(f"feature_player_{statistic}_per90_{window}" for window in (3, 5, 10))
        columns.extend([
            f"feature_player_season_{statistic}_per90",
            f"feature_player_previous_season_{statistic}_per90",
        ])
    return columns


def _team_features(statistics: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for side in ("team", "opponent"):
        for statistic in statistics:
            columns.extend(f"feature_{side}_{statistic}_mean_{window}" for window in (3, 5, 10))
            columns.extend([
                f"feature_{side}_{statistic}_venue_mean_5",
                f"feature_{side}_season_{statistic}_mean",
                f"feature_{side}_previous_season_{statistic}_mean",
            ])
    return columns


USAGE_FEATURES = [
    "gameweek", "is_home", "feature_player_history_fixtures",
    "feature_player_history_minutes", "feature_player_season_history_minutes",
    "feature_player_previous_minutes", "feature_player_previous_appeared",
    "feature_player_rest_days", "feature_player_changed_team",
    "feature_player_is_new", "feature_player_minutes_mean_3",
    "feature_player_minutes_mean_5", "feature_player_minutes_mean_10",
    "feature_player_season_appeared_rate", "feature_player_previous_season_appeared_rate",
    "feature_player_previous_season_minutes",
    "feature_team_rest_days",
]
SAVE_FEATURES = _player_features(["saves", "goals_conceded", "expected_goals_conceded"])
SAVE_FEATURES += [
    "feature_player_season_penalties_saved_total",
    "feature_player_previous_season_penalties_saved_total",
]
DEFCON_FEATURES = _player_features(["defensive_contribution"]) + [
    f"feature_player_defensive_observations_{window}" for window in (3, 5, 10)
]
TEAM_FEATURES = _team_features([
    "goals_for", "goals_against", "expected_goals_for", "expected_goals_against",
    "saves", "defensive_contribution",
])
FEATURE_PROFILES = {
    "history": {
        "numeric": USAGE_FEATURES + SAVE_FEATURES + DEFCON_FEATURES,
        "categorical": ["position"],
    },
    "history_and_fixture": {
        "numeric": USAGE_FEATURES + SAVE_FEATURES + DEFCON_FEATURES + TEAM_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
}


def get_feature_columns(profile: str) -> tuple[list[str], list[str]]:
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown defensive feature profile: {profile}")
    definition = FEATURE_PROFILES[profile]
    return list(definition["numeric"]), list(definition["categorical"])


def all_feature_columns() -> list[str]:
    return sorted({column for definition in FEATURE_PROFILES.values() for kind in ("numeric", "categorical") for column in definition[kind]})


def load_training_data(path: str | Path) -> pd.DataFrame:
    required = list(dict.fromkeys(IDENTIFIER_COLUMNS + TARGET_COLUMNS + all_feature_columns()))
    available = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"Training dataset is missing columns: {', '.join(missing)}")
    rows = pd.read_csv(path, usecols=required, low_memory=False)
    rows = rows.loc[rows["target_outcome_known"].eq(1)].copy()
    if rows.duplicated(PLAYER_KEYS).any():
        raise ValueError("Training dataset contains duplicate player-fixture rows")
    return rows.sort_values(["season_start", "kickoff_time", "fixture_id", "player_key"]).reset_index(drop=True)


def _preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def build_count_model(profile: str, parameters: Mapping[str, object], *, random_state: int) -> Pipeline:
    numeric, categorical = get_feature_columns(profile)
    return Pipeline([
        ("features", _preprocessor(numeric, categorical)),
        ("model", HistGradientBoostingRegressor(
            loss="poisson", early_stopping=False, random_state=random_state, **parameters
        )),
    ])


def position_rate_priors(rows: pd.DataFrame, target: str) -> dict[str, float]:
    appeared = rows.loc[rows["target_minutes"].gt(0)]
    grouped = appeared.groupby("position").agg(events=(target, "sum"), minutes=("target_minutes", "sum"))
    rates = (90 * grouped["events"] / grouped["minutes"]).to_dict()
    rates["__default__"] = 90 * appeared[target].sum() / appeared["target_minutes"].sum()
    return {str(key): float(value) for key, value in rates.items()}


def _poisson_tail(mean: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mean), dtype=float)
    for cutoff in np.unique(threshold):
        selected = threshold == cutoff
        if cutoff <= 0:
            result[selected] = 1.0
            continue
        if cutoff > 100:
            continue
        values = mean[selected]
        term = np.exp(-values)
        cumulative = term.copy()
        for count in range(1, int(cutoff)):
            term *= values / count
            cumulative += term
        result[selected] = np.maximum(0.0, 1.0 - cumulative)
    return result


def expected_grouped_points(mean: np.ndarray, group_size: int, *, maximum_count: int = 30) -> np.ndarray:
    """Return E[floor(Poisson(mean) / group_size)]."""
    result = np.zeros(len(mean), dtype=float)
    probability = np.exp(-mean)
    for count in range(1, maximum_count + 1):
        probability = probability * mean / count
        result += (count // group_size) * probability
    return result


def _prior_values(positions: pd.Series, priors: Mapping[str, float]) -> np.ndarray:
    return positions.map(priors).fillna(float(priors["__default__"])).to_numpy(float)


def build_predictions(
    rows: pd.DataFrame,
    raw_save_rates: Sequence[float],
    raw_defcon_rates: Sequence[float],
    save_priors: Mapping[str, float],
    defcon_priors: Mapping[str, float],
    penalty_save_priors: Mapping[str, float],
    *,
    save_shrinkage_minutes: float,
    defcon_shrinkage_minutes: float,
    penalty_shrinkage_minutes: float,
) -> pd.DataFrame:
    required = set(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS)
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Defensive input is missing columns: {', '.join(missing)}")
    if rows.duplicated(PLAYER_KEYS).any():
        raise ValueError("Defensive input contains duplicate player-fixture rows")
    minutes = rows["expected_minutes"].to_numpy(float)
    if np.any((minutes < 0) | (minutes > 90)):
        raise ValueError("expected_minutes must be between zero and 90")
    history_minutes = rows["feature_player_history_minutes"].fillna(0).to_numpy(float)
    save_weight = np.divide(
        history_minutes,
        history_minutes + save_shrinkage_minutes,
        out=np.zeros_like(history_minutes),
        where=(history_minutes + save_shrinkage_minutes) > 0,
    )
    defcon_weight = np.divide(
        history_minutes,
        history_minutes + defcon_shrinkage_minutes,
        out=np.zeros_like(history_minutes),
        where=(history_minutes + defcon_shrinkage_minutes) > 0,
    )
    save_prior = _prior_values(rows["position"], save_priors)
    save_rate = np.where(rows["position"].eq("GK"), save_weight * np.clip(raw_save_rates, 0, 15) + (1 - save_weight) * save_prior, 0.0)
    defcon_prior = _prior_values(rows["position"], defcon_priors)
    defcon_rate = defcon_weight * np.clip(raw_defcon_rates, 0, 40) + (1 - defcon_weight) * defcon_prior
    defcon_rate = np.where(rows["position"].eq("GK"), 0.0, defcon_rate)
    expected_saves = save_rate * minutes / 90
    expected_defcon = defcon_rate * minutes / 90

    season_events = rows["feature_player_season_penalties_saved_total"].fillna(0).to_numpy(float)
    previous_events = rows["feature_player_previous_season_penalties_saved_total"].fillna(0).to_numpy(float)
    previous_minutes = rows["feature_player_previous_season_minutes"].fillna(0).to_numpy(float)
    event_minutes = rows["feature_player_season_history_minutes"].fillna(0).to_numpy(float) + previous_minutes
    prior_rate = _prior_values(rows["position"], penalty_save_priors)
    penalty_rate = 90 * (season_events + previous_events + prior_rate * penalty_shrinkage_minutes / 90) / (event_minutes + penalty_shrinkage_minutes)
    penalty_rate = np.where(rows["position"].eq("GK"), penalty_rate, 0.0)
    expected_penalty_saves = penalty_rate * minutes / 90

    clean_sheet_probability = rows["team_clean_sheet_probability"].to_numpy(float) * rows["sixty_plus_probability"].to_numpy(float)
    clean_sheet_points = rows["position"].map({"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}).fillna(0).to_numpy(float) * clean_sheet_probability
    expected_goals_conceded = rows["team_expected_goals_against"].to_numpy(float) * minutes / 90
    goals_conceded_points = -expected_grouped_points(expected_goals_conceded, 2)
    goals_conceded_points = np.where(rows["position"].isin(["GK", "DEF"]), goals_conceded_points, 0.0)
    save_points = expected_grouped_points(expected_saves, 3)
    thresholds = rows["position"].map({"DEF": 10, "MID": 12, "FWD": 12}).fillna(999).to_numpy(int)
    defcon_probability = _poisson_tail(expected_defcon, thresholds)
    defcon_points = 2 * defcon_probability

    output = rows[[column for column in IDENTIFIER_COLUMNS if column in rows]].reset_index(drop=True).copy()
    output["expected_minutes"] = minutes
    output["team_clean_sheet_probability"] = rows["team_clean_sheet_probability"].to_numpy(float)
    output["player_clean_sheet_probability"] = clean_sheet_probability
    output["expected_clean_sheet_points"] = clean_sheet_points
    output["expected_goals_conceded"] = expected_goals_conceded
    output["expected_goals_conceded_deduction"] = goals_conceded_points
    output["save_rate_per90"] = save_rate
    output["expected_saves"] = expected_saves
    output["save_points_probability"] = _poisson_tail(expected_saves, np.full(len(rows), 3))
    output["expected_save_points"] = save_points
    output["expected_penalty_saves"] = expected_penalty_saves
    output["expected_penalty_save_points"] = 5 * expected_penalty_saves
    output["defensive_contribution_rate_per90"] = defcon_rate
    output["expected_defensive_contributions"] = expected_defcon
    output["defensive_contribution_threshold"] = np.where(thresholds == 999, np.nan, thresholds)
    output["defensive_contribution_points_probability"] = defcon_probability
    output["expected_defensive_contribution_points"] = defcon_points
    output["expected_defensive_fpl_points"] = clean_sheet_points + goals_conceded_points + save_points + 5 * expected_penalty_saves + defcon_points
    if set(TARGET_COLUMNS).issubset(rows.columns):
        output["actual_saves"] = rows["target_saves"].to_numpy()
        output["actual_penalty_saves"] = rows["target_penalties_saved"].to_numpy()
        output["actual_defensive_contributions"] = rows["target_defensive_contribution"].to_numpy()
        output["defensive_contribution_available"] = rows["target_defensive_contribution_available"].to_numpy()
        output["actual_clean_sheet"] = rows["target_clean_sheets"].to_numpy()
        output["actual_goals_conceded"] = rows["target_goals_conceded"].to_numpy()
        if "defcon_walk_forward_scored" in rows:
            output["defcon_walk_forward_scored"] = rows["defcon_walk_forward_scored"].to_numpy()
    return output


def predict_with_artifact(artifact: Mapping[str, object], fixture_rows: pd.DataFrame) -> pd.DataFrame:
    features = list(artifact["feature_columns"])
    missing = sorted(set(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS + features).difference(fixture_rows.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")
    save_rates = artifact["save_model"].predict(fixture_rows[features])
    defcon_rates = artifact["defcon_model"].predict(fixture_rows[features])
    return build_predictions(
        fixture_rows, save_rates, defcon_rates, artifact["save_priors"],
        artifact["defcon_priors"], artifact["penalty_save_priors"],
        save_shrinkage_minutes=float(artifact["save_shrinkage_minutes"]),
        defcon_shrinkage_minutes=float(artifact["defcon_shrinkage_minutes"]),
        penalty_shrinkage_minutes=float(artifact["penalty_shrinkage_minutes"]),
    )


def load_artifact(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, object]:
    return joblib.load(Path(model_dir) / "model.joblib")


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON containing player_fixture_rows.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    predictions = predict_with_artifact(load_artifact(args.model_dir), pd.DataFrame(payload["player_fixture_rows"]))
    print(json.dumps({"predictions": records_for_json(predictions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
