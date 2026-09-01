"""Yellow-card, red-card, own-goal, and penalty-miss prediction."""

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


DEFAULT_MODEL_DIR = Path("data/models/misc_events")
PLAYER_KEYS = ["season", "fixture_id", "player_key"]
IDENTIFIER_COLUMNS = [
    "season", "season_start", "fixture_id", "gameweek", "kickoff_time",
    "player_id", "player_code", "player_key", "player_name", "position",
    "team_key", "team_name", "opponent_team_key", "opponent_team_name", "is_home",
]
UPSTREAM_COLUMNS = ["expected_minutes"]
EVENTS = {
    "yellow_cards": {"target": "target_yellow_cards", "points": -1, "probability": "yellow_card_probability"},
    "red_cards": {"target": "target_red_cards", "points": -3, "probability": "red_card_probability"},
    "own_goals": {"target": "target_own_goals", "points": -2, "probability": "own_goal_probability"},
    "penalties_missed": {"target": "target_penalties_missed", "points": -2, "probability": "penalty_miss_probability"},
}
TARGET_COLUMNS = ["target_outcome_known", "target_minutes"] + [definition["target"] for definition in EVENTS.values()]


def _player_event_features() -> list[str]:
    columns: list[str] = []
    for event in EVENTS:
        columns.extend(f"feature_player_{event}_per90_{window}" for window in (3, 5, 10))
        columns.extend([
            f"feature_player_season_{event}_total",
            f"feature_player_season_{event}_per90",
            f"feature_player_previous_season_{event}_total",
            f"feature_player_previous_season_{event}_per90",
        ])
    return columns


def _team_features() -> list[str]:
    columns: list[str] = []
    for side in ("team", "opponent"):
        for event in ("yellow_cards", "red_cards"):
            columns.extend(f"feature_{side}_{event}_mean_{window}" for window in (3, 5, 10))
            columns.extend([
                f"feature_{side}_{event}_venue_mean_5",
                f"feature_{side}_season_{event}_mean",
                f"feature_{side}_previous_season_{event}_mean",
            ])
    return columns


USAGE_FEATURES = [
    "gameweek", "is_home", "feature_player_history_fixtures",
    "feature_player_history_minutes", "feature_player_season_history_minutes",
    "feature_player_previous_season_minutes", "feature_player_previous_minutes",
    "feature_player_previous_appeared", "feature_player_rest_days",
    "feature_player_changed_team", "feature_player_is_new",
    "feature_player_minutes_mean_3", "feature_player_minutes_mean_5",
    "feature_player_minutes_mean_10", "feature_player_season_appeared_rate",
    "feature_player_previous_season_appeared_rate", "feature_team_rest_days",
]
EVENT_FEATURES = _player_event_features()
TEAM_FEATURES = _team_features()
MARKET_FEATURES = ["feature_price_millions", "feature_ownership_percent"]
FEATURE_PROFILES = {
    "event_history": {
        "numeric": USAGE_FEATURES + EVENT_FEATURES,
        "categorical": ["position"],
    },
    "event_history_and_fixture": {
        "numeric": USAGE_FEATURES + EVENT_FEATURES + TEAM_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
    "event_history_fixture_and_market": {
        "numeric": USAGE_FEATURES + EVENT_FEATURES + TEAM_FEATURES + MARKET_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
}


def get_feature_columns(profile: str) -> tuple[list[str], list[str]]:
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown miscellaneous-event feature profile: {profile}")
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


def build_yellow_card_model(profile: str, parameters: Mapping[str, object], *, random_state: int) -> Pipeline:
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
    return {str(position): float(rate) for position, rate in rates.items()}


def _prior_values(positions: pd.Series, priors: Mapping[str, float]) -> np.ndarray:
    return positions.map(priors).fillna(float(priors["__default__"])).to_numpy(float)


def _historical_rate(rows: pd.DataFrame, event: str, priors: Mapping[str, float], shrinkage_minutes: float) -> np.ndarray:
    events = (
        rows[f"feature_player_season_{event}_total"].fillna(0).to_numpy(float)
        + rows[f"feature_player_previous_season_{event}_total"].fillna(0).to_numpy(float)
    )
    minutes = (
        rows["feature_player_season_history_minutes"].fillna(0).to_numpy(float)
        + rows["feature_player_previous_season_minutes"].fillna(0).to_numpy(float)
    )
    prior = _prior_values(rows["position"], priors)
    return 90 * (events + prior * shrinkage_minutes / 90) / (minutes + shrinkage_minutes)


def build_predictions(
    rows: pd.DataFrame,
    raw_yellow_rates: Sequence[float],
    priors: Mapping[str, Mapping[str, float]],
    shrinkage_minutes: Mapping[str, float],
) -> pd.DataFrame:
    required = set(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS)
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Miscellaneous-event input is missing columns: {', '.join(missing)}")
    minutes = rows["expected_minutes"].to_numpy(float)
    if np.any((minutes < 0) | (minutes > 90)):
        raise ValueError("expected_minutes must be between zero and 90")
    history_minutes = rows["feature_player_history_minutes"].fillna(0).to_numpy(float)
    yellow_shrinkage = float(shrinkage_minutes["yellow_cards"])
    yellow_weight = np.divide(
        history_minutes, history_minutes + yellow_shrinkage,
        out=np.zeros_like(history_minutes), where=(history_minutes + yellow_shrinkage) > 0,
    )
    yellow_prior = _prior_values(rows["position"], priors["yellow_cards"])
    rates = {
        "yellow_cards": yellow_weight * np.clip(raw_yellow_rates, 0, 2) + (1 - yellow_weight) * yellow_prior,
    }
    for event in ("red_cards", "own_goals", "penalties_missed"):
        rates[event] = _historical_rate(rows, event, priors[event], float(shrinkage_minutes[event]))

    output = rows[[column for column in IDENTIFIER_COLUMNS if column in rows]].reset_index(drop=True).copy()
    output["expected_minutes"] = minutes
    expected_deduction = np.zeros(len(rows), dtype=float)
    for event, definition in EVENTS.items():
        expected = rates[event] * minutes / 90
        output[f"{event}_rate_per90"] = rates[event]
        output[f"expected_{event}"] = expected
        output[str(definition["probability"])] = 1 - np.exp(-expected)
        points = float(definition["points"]) * expected
        output[f"expected_{event}_points"] = points
        expected_deduction += points
        target = str(definition["target"])
        if target in rows:
            output[f"actual_{event}"] = rows[target].to_numpy()
    output["expected_misc_fpl_points"] = expected_deduction
    return output


def predict_with_artifact(artifact: Mapping[str, object], fixture_rows: pd.DataFrame) -> pd.DataFrame:
    features = list(artifact["feature_columns"])
    missing = sorted(set(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS + features).difference(fixture_rows.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")
    raw_rates = artifact["yellow_card_model"].predict(fixture_rows[features])
    return build_predictions(fixture_rows, raw_rates, artifact["position_priors"], artifact["shrinkage_minutes"])


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
