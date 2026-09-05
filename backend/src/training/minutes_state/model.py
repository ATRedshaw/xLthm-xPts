"""Consistent player appearance, role, and minutes-state prediction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_MODEL_DIR = Path("data/models/minutes_state")
PLAYER_ROW_KEYS = ["season", "fixture_id", "player_key"]
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
TARGET_COLUMNS = [
    "target_outcome_known",
    "target_appeared",
    "target_started",
    "target_sixty_plus",
    "target_minutes",
]

DID_NOT_PLAY = "did_not_play"
APPEARED_STATES = [
    "sub_under_60",
    "sub_60_plus",
    "start_under_60",
    "start_60_plus",
]
MINUTES_STATES = [DID_NOT_PLAY, *APPEARED_STATES]
STATE_MINUTE_BOUNDS = {
    "sub_under_60": (1.0, 59.0),
    "sub_60_plus": (60.0, 90.0),
    "start_under_60": (1.0, 59.0),
    "start_60_plus": (60.0, 90.0),
}

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
STABLE_MARKET_FEATURES = [
    "feature_price_millions",
    "feature_ownership_percent",
    "feature_price_change_previous_snapshot",
    "feature_price_change_season",
    "feature_team_position_price_rank",
    "feature_team_position_ownership_rank",
]
VOLATILE_MARKET_FEATURES = [
    "feature_transfers_in_percent",
    "feature_transfers_out_percent",
    "feature_transfers_balance_percent",
]
FEATURE_PROFILES = {
    "usage_context": {
        "numeric": USAGE_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
    "usage_context_and_stable_market": {
        "numeric": USAGE_FEATURES + STABLE_MARKET_FEATURES,
        "categorical": ["position", "team_name", "opponent_team_name"],
    },
    "usage_context_and_market": {
        "numeric": USAGE_FEATURES
        + STABLE_MARKET_FEATURES
        + VOLATILE_MARKET_FEATURES,
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


def load_minutes_training_data(path: str | Path) -> pd.DataFrame:
    """Load player-fixture rows and mark whether each start label is reliable."""
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

    appeared_from_minutes = rows["target_minutes"].gt(0).astype(int)
    sixty_from_minutes = rows["target_minutes"].ge(60).astype(int)
    if not rows["target_appeared"].eq(appeared_from_minutes).all():
        raise ValueError("Appearance labels disagree with target minutes")
    if not rows["target_sixty_plus"].eq(sixty_from_minutes).all():
        raise ValueError("60+ labels disagree with target minutes")

    starter_counts = rows.groupby(
        ["season", "fixture_id", "team_key"], dropna=False
    )["target_started"].transform("sum")
    rows["start_label_available"] = starter_counts.eq(11)
    invalid_starts = rows["start_label_available"] & rows["target_started"].gt(
        rows["target_appeared"]
    )
    if invalid_starts.any():
        raise ValueError("Reliable start labels contain a start without an appearance")

    conditions = [
        rows["target_appeared"].eq(1)
        & rows["target_started"].eq(0)
        & rows["target_sixty_plus"].eq(0),
        rows["target_appeared"].eq(1)
        & rows["target_started"].eq(0)
        & rows["target_sixty_plus"].eq(1),
        rows["target_appeared"].eq(1)
        & rows["target_started"].eq(1)
        & rows["target_sixty_plus"].eq(0),
        rows["target_appeared"].eq(1)
        & rows["target_started"].eq(1)
        & rows["target_sixty_plus"].eq(1),
    ]
    rows["minutes_state"] = np.select(
        conditions, APPEARED_STATES, default=DID_NOT_PLAY
    )
    rows.loc[~rows["start_label_available"], "minutes_state"] = pd.NA
    rows = rows.sort_values(
        ["season_start", "kickoff_time", "fixture_id", "player_key"]
    ).reset_index(drop=True)
    return rows


def _preprocessor(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, object, Sequence[str]]] = [
        (
            "numeric",
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            numeric_features,
        )
    ]
    if categorical_features:
        transformers.append(
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
            )
        )
    return ColumnTransformer(transformers)


def build_state_classifier(
    profile: str,
    parameters: Mapping[str, object],
    *,
    random_state: int,
) -> Pipeline:
    """Build a gradient-boosting classifier for appearance or appeared state."""
    numeric_features, categorical_features = get_feature_columns(profile)
    return Pipeline(
        [
            ("features", _preprocessor(numeric_features, categorical_features)),
            (
                "model",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    early_stopping=False,
                    random_state=random_state,
                    **parameters,
                ),
            ),
        ]
    )


def build_duration_regressor(
    profile: str,
    parameters: Mapping[str, object],
    *,
    random_state: int,
) -> Pipeline:
    """Build the state-conditioned minutes regressor."""
    numeric_features, categorical_features = get_feature_columns(profile)
    return Pipeline(
        [
            (
                "features",
                _preprocessor(
                    numeric_features,
                    [*categorical_features, "minutes_state"],
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    early_stopping=False,
                    random_state=random_state,
                    **parameters,
                ),
            ),
        ]
    )


def apply_temperature(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Apply multiclass temperature scaling while preserving probability sums."""
    if temperature <= 0:
        raise ValueError("Probability temperature must be positive")
    log_probabilities = np.log(np.clip(probabilities, 1e-15, 1)) / temperature
    log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
    scaled = np.exp(log_probabilities)
    return scaled / scaled.sum(axis=1, keepdims=True)


def predict_state_probabilities(
    appearance_model: Pipeline,
    appeared_state_model: Pipeline,
    rows: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Predict coherent probabilities for DNP and four appeared states."""
    appearance_classes = list(appearance_model.named_steps["model"].classes_)
    appeared_index = appearance_classes.index(1)
    appeared_probability = appearance_model.predict_proba(
        rows[list(feature_columns)]
    )[:, appeared_index]

    conditional_classes = list(
        appeared_state_model.named_steps["model"].classes_
    )
    conditional_raw = appeared_state_model.predict_proba(rows[list(feature_columns)])
    conditional_probabilities = np.column_stack(
        [
            conditional_raw[:, conditional_classes.index(state)]
            for state in APPEARED_STATES
        ]
    )
    probabilities = np.column_stack(
        [
            1 - appeared_probability,
            appeared_probability[:, None] * conditional_probabilities,
        ]
    )
    return apply_temperature(probabilities, temperature)


def predict_state_durations(
    duration_model: Pipeline,
    rows: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    duration_offset: float = 0.0,
) -> np.ndarray:
    """Predict a feasible duration for every appeared state."""
    predictions = []
    for state in APPEARED_STATES:
        state_rows = rows[list(feature_columns)].copy()
        state_rows["minutes_state"] = state
        lower, upper = STATE_MINUTE_BOUNDS[state]
        predictions.append(
            np.clip(
                duration_model.predict(state_rows) + duration_offset,
                lower,
                upper,
            )
        )
    return np.column_stack(predictions)


def build_prediction_frame(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    state_durations: np.ndarray,
) -> pd.DataFrame:
    """Return player-fixture outputs derived from coherent state probabilities."""
    if probabilities.shape != (len(rows), len(MINUTES_STATES)):
        raise ValueError("State probabilities do not match the prediction rows")
    if state_durations.shape != (len(rows), len(APPEARED_STATES)):
        raise ValueError("State durations do not match the prediction rows")
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Minutes-state probabilities must sum to one")

    output_columns = [column for column in IDENTIFIER_COLUMNS if column in rows]
    output = rows[output_columns].reset_index(drop=True).copy()
    for index, state in enumerate(MINUTES_STATES):
        output[f"probability_{state}"] = probabilities[:, index]
    for index, state in enumerate(APPEARED_STATES):
        output[f"minutes_if_{state}"] = state_durations[:, index]

    appeared_probability = probabilities[:, 1:].sum(axis=1)
    start_probability = probabilities[:, 3:].sum(axis=1)
    sixty_plus_probability = probabilities[:, [2, 4]].sum(axis=1)
    expected_minutes = (probabilities[:, 1:] * state_durations).sum(axis=1)
    output["appearance_probability"] = appeared_probability
    output["start_probability"] = start_probability
    output["sixty_plus_probability"] = sixty_plus_probability
    output["expected_minutes"] = expected_minutes
    output["expected_minutes_if_appears"] = np.divide(
        expected_minutes,
        appeared_probability,
        out=np.zeros_like(expected_minutes),
        where=appeared_probability > 0,
    )
    output["most_likely_state"] = np.asarray(MINUTES_STATES)[
        probabilities.argmax(axis=1)
    ]

    if set(TARGET_COLUMNS).issubset(rows.columns):
        output["actual_appeared"] = rows["target_appeared"].to_numpy()
        output["actual_started"] = rows["target_started"].to_numpy()
        output["actual_sixty_plus"] = rows["target_sixty_plus"].to_numpy()
        output["actual_minutes"] = rows["target_minutes"].to_numpy()
        if "start_label_available" in rows:
            output["start_label_available"] = rows[
                "start_label_available"
            ].to_numpy()
        if "minutes_state" in rows:
            output["actual_minutes_state"] = rows["minutes_state"].to_numpy()
    return output


def predict_with_artifact(
    artifact: Mapping[str, object], fixture_rows: pd.DataFrame
) -> pd.DataFrame:
    """Predict player minutes states using a saved artifact."""
    feature_columns = list(artifact["feature_columns"])
    required = set(IDENTIFIER_COLUMNS).union(feature_columns)
    missing = sorted(required.difference(fixture_rows.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")

    probabilities = predict_state_probabilities(
        artifact["appearance_model"],
        artifact["appeared_state_model"],
        fixture_rows,
        feature_columns,
        temperature=float(artifact["probability_temperature"]),
    )
    durations = predict_state_durations(
        artifact["duration_model"],
        fixture_rows,
        feature_columns,
        duration_offset=float(artifact["duration_offset"]),
    )
    return build_prediction_frame(fixture_rows, probabilities, durations)


def load_artifact(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, object]:
    """Load a trained minutes-state artifact."""
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
        print(f"Wrote minutes-state predictions to {output_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
