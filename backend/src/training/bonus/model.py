"""Conditional BPS prediction and fixture-level bonus simulation."""

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


DEFAULT_MODEL_DIR = Path("data/models/bonus")
PLAYER_KEYS = ["season", "fixture_id", "player_key"]
FIXTURE_KEYS = ["season", "fixture_id"]
IDENTIFIER_COLUMNS = [
    "season", "season_start", "fixture_id", "gameweek", "kickoff_time",
    "player_id", "player_code", "player_key", "player_name", "position",
    "team_key", "team_name", "opponent_team_key", "opponent_team_name", "is_home",
]
UPSTREAM_COLUMNS = [
    "appearance_probability", "start_probability", "sixty_plus_probability",
    "expected_minutes", "team_expected_goals_for", "team_expected_goals_against",
    "team_clean_sheet_probability", "expected_goals", "expected_fpl_assists",
    "player_clean_sheet_probability", "expected_goals_conceded", "expected_saves",
    "expected_penalty_saves", "expected_defensive_contributions",
    "expected_yellow_cards", "expected_red_cards", "expected_own_goals",
    "expected_penalties_missed",
]
TARGET_COLUMNS = ["target_outcome_known", "target_minutes", "target_bps", "target_bonus"]


def _player_features(statistics: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for statistic in statistics:
        columns.extend(f"feature_player_{statistic}_per90_{window}" for window in (3, 5, 10))
        columns.extend([
            f"feature_player_season_{statistic}_per90",
            f"feature_player_previous_season_{statistic}_per90",
        ])
    return columns


HISTORY_FEATURES = _player_features([
    "bps", "bonus", "influence", "creativity", "threat", "ict_index",
    "goals_scored", "assists", "goals_conceded", "saves",
    "yellow_cards", "red_cards", "defensive_contribution",
]) + [
    "gameweek", "is_home", "feature_player_history_fixtures",
    "feature_player_history_minutes", "feature_player_season_history_minutes",
    "feature_player_previous_season_minutes", "feature_player_previous_minutes",
    "feature_player_previous_appeared", "feature_player_rest_days",
    "feature_player_changed_team", "feature_player_is_new",
]
MARKET_FEATURES = ["feature_price_millions", "feature_ownership_percent"]
FEATURE_PROFILES = {
    "components": {
        "numeric": UPSTREAM_COLUMNS,
        "categorical": ["position"],
    },
    "components_and_history": {
        "numeric": UPSTREAM_COLUMNS + HISTORY_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
    "components_history_and_market": {
        "numeric": UPSTREAM_COLUMNS + HISTORY_FEATURES + MARKET_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
}


def get_feature_columns(profile: str) -> tuple[list[str], list[str]]:
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown bonus feature profile: {profile}")
    definition = FEATURE_PROFILES[profile]
    return list(definition["numeric"]), list(definition["categorical"])


def all_history_columns() -> list[str]:
    return sorted(set(HISTORY_FEATURES + MARKET_FEATURES + ["position", "team_name", "opponent_team_name"]))


def load_training_data(path: str | Path) -> pd.DataFrame:
    required = list(dict.fromkeys(IDENTIFIER_COLUMNS + TARGET_COLUMNS + all_history_columns()))
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


def build_bps_model(profile: str, parameters: Mapping[str, object], *, random_state: int) -> Pipeline:
    numeric, categorical = get_feature_columns(profile)
    return Pipeline([
        ("features", _preprocessor(numeric, categorical)),
        ("model", HistGradientBoostingRegressor(
            early_stopping=False, random_state=random_state, **parameters
        )),
    ])


def build_bps_predictions(rows: pd.DataFrame, conditional_bps: Sequence[float], residual_std: Mapping[str, float]) -> pd.DataFrame:
    missing = sorted(set(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"Bonus input is missing columns: {', '.join(missing)}")
    if rows.duplicated(PLAYER_KEYS).any():
        raise ValueError("Bonus input contains duplicate player-fixture rows")
    conditional = np.asarray(conditional_bps, dtype=float)
    appearance = rows["appearance_probability"].to_numpy(float)
    if np.any((appearance < 0) | (appearance > 1)):
        raise ValueError("appearance_probability must be between zero and one")
    output = rows[[column for column in IDENTIFIER_COLUMNS if column in rows]].reset_index(drop=True).copy()
    output["appearance_probability"] = appearance
    output["conditional_expected_bps"] = conditional
    output["expected_bps"] = appearance * conditional
    output["bps_residual_std"] = rows["position"].map(residual_std).fillna(float(residual_std["__default__"])).to_numpy(float)
    if "target_bps" in rows:
        output["actual_bps"] = rows["target_bps"].to_numpy()
        output["actual_bonus"] = rows["target_bonus"].to_numpy()
        output["actual_appeared"] = rows["target_minutes"].gt(0).astype(int).to_numpy()
    return output


def _allocate_bonus(sampled_bps: np.ndarray, appeared: np.ndarray) -> np.ndarray:
    simulations, players = sampled_bps.shape
    bonus = np.zeros((simulations, players), dtype=np.int8)
    values = np.where(appeared, sampled_bps, -np.inf)
    highest = np.max(values, axis=1)
    first = (values == highest[:, None]) & np.isfinite(values)
    first_count = first.sum(axis=1)
    bonus[first] = 3

    remaining = np.where(first, -np.inf, values)
    second_value = np.max(remaining, axis=1)
    second = (remaining == second_value[:, None]) & np.isfinite(remaining)
    two_tied_first = first_count == 2
    one_first = first_count == 1
    bonus[second & two_tied_first[:, None]] = 1
    bonus[second & one_first[:, None]] = 2

    second_count = second.sum(axis=1)
    third_required = one_first & (second_count == 1)
    remaining = np.where(second, -np.inf, remaining)
    third_value = np.max(remaining, axis=1)
    third = (remaining == third_value[:, None]) & np.isfinite(remaining)
    bonus[third & third_required[:, None]] = 1
    return bonus


def simulate_bonus(
    bps_predictions: pd.DataFrame,
    *,
    simulation_count: int,
    random_state: int,
    residual_scale: float,
) -> pd.DataFrame:
    if simulation_count < 1:
        raise ValueError("simulation_count must be positive")
    if residual_scale <= 0:
        raise ValueError("residual_scale must be positive")
    fixture_sizes = bps_predictions.groupby(FIXTURE_KEYS).size()
    team_counts = bps_predictions.groupby(FIXTURE_KEYS)["team_key"].nunique()
    if not fixture_sizes.ge(22).all() or not team_counts.eq(2).all():
        raise ValueError("Bonus simulation requires the full player set from both teams")
    rng = np.random.default_rng(random_state)
    frames = []
    for _, fixture in bps_predictions.groupby(FIXTURE_KEYS, sort=False):
        appearance_probability = fixture["appearance_probability"].to_numpy(float)
        appeared = rng.random((simulation_count, len(fixture))) < appearance_probability
        conditional = fixture["conditional_expected_bps"].to_numpy(float)
        standard_deviation = fixture["bps_residual_std"].to_numpy(float) * residual_scale
        sampled_bps = np.rint(rng.normal(conditional, standard_deviation, (simulation_count, len(fixture))))
        bonus = _allocate_bonus(sampled_bps, appeared)
        output = fixture.reset_index(drop=True).copy()
        output["expected_bonus"] = bonus.mean(axis=0)
        for points in (0, 1, 2, 3):
            output[f"bonus_{points}_probability"] = (bonus == points).mean(axis=0)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def predict_with_artifact(
    artifact: Mapping[str, object],
    fixture_rows: pd.DataFrame,
    *,
    simulation_count: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    features = list(artifact["feature_columns"])
    missing = sorted(set(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS + features).difference(fixture_rows.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")
    conditional = artifact["bps_model"].predict(fixture_rows[features])
    bps = build_bps_predictions(fixture_rows, conditional, artifact["residual_std_by_position"])
    return simulate_bonus(
        bps,
        simulation_count=int(simulation_count or artifact["simulation_count"]),
        random_state=int(artifact["random_state"] if random_state is None else random_state),
        residual_scale=float(artifact["residual_scale"]),
    )


def load_artifact(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, object]:
    return joblib.load(Path(model_dir) / "model.joblib")


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON containing player_fixture_rows.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--simulations", type=int, default=None)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    predictions = predict_with_artifact(
        load_artifact(args.model_dir), pd.DataFrame(payload["player_fixture_rows"]),
        simulation_count=args.simulations,
    )
    print(json.dumps({"predictions": records_for_json(predictions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
