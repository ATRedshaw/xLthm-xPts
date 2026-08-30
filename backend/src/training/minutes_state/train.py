"""Tune and train the consistent player minutes-state model."""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .model import (
    APPEARED_STATES,
    DEFAULT_MODEL_DIR,
    DID_NOT_PLAY,
    IDENTIFIER_COLUMNS,
    MINUTES_STATES,
    STABLE_MARKET_FEATURES,
    STATE_MINUTE_BOUNDS,
    VOLATILE_MARKET_FEATURES,
    apply_temperature,
    build_duration_regressor,
    build_prediction_frame,
    build_state_classifier,
    get_feature_columns,
    load_minutes_training_data,
    predict_state_durations,
    predict_state_probabilities,
    predict_with_artifact,
    records_for_json,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "minutes_state.yaml"
)
PROBABILITY_COLUMNS = [f"probability_{state}" for state in MINUTES_STATES]


def load_settings(path: str | Path) -> dict[str, object]:
    """Load and validate the minutes_state configuration section."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = config.get("minutes_state") if isinstance(config, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must contain a minutes_state mapping: {path}")
    return settings


def _expand_grid(grid: object, name: str) -> list[dict[str, object]]:
    if not isinstance(grid, dict) or not grid:
        raise ValueError(f"minutes_state.{name} must be a non-empty mapping")
    parameter_names = list(grid)
    parameter_values = []
    for parameter_name in parameter_names:
        values = grid[parameter_name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}.{parameter_name} must be a non-empty list")
        parameter_values.append(values)
    return [
        dict(zip(parameter_names, values))
        for values in itertools.product(*parameter_values)
    ]


def classifier_candidates(settings: Mapping[str, object]) -> list[dict[str, object]]:
    profiles = settings.get("feature_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("minutes_state.feature_profiles must be a non-empty list")
    parameters = _expand_grid(
        settings.get("classifier_parameter_grid"), "classifier_parameter_grid"
    )
    return [
        {"feature_profile": str(profile), "parameters": candidate}
        for profile in profiles
        for candidate in parameters
    ]


def duration_candidates(settings: Mapping[str, object]) -> list[dict[str, object]]:
    return _expand_grid(
        settings.get("duration_parameter_grid"), "duration_parameter_grid"
    )


def chronological_splits(
    rows: pd.DataFrame, validation_seasons: Sequence[str]
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window season splits without future leakage."""
    season_starts = (
        rows[["season", "season_start"]]
        .drop_duplicates()
        .set_index("season")["season_start"]
        .to_dict()
    )
    splits = []
    for season in validation_seasons:
        if season not in season_starts:
            raise ValueError(f"Validation season is absent from the dataset: {season}")
        training = rows.loc[rows["season_start"].lt(season_starts[season])].copy()
        validation = rows.loc[rows["season"].eq(season)].copy()
        if training.empty or validation.empty:
            raise ValueError(f"Validation season {season} does not have a usable split")
        splits.append((season, training, validation))
    return splits


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    probability_floor = 1e-15
    probabilities = np.clip(predicted.astype(float), probability_floor, 1 - probability_floor)
    targets = actual.astype(int)
    bins = pd.cut(probabilities, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = pd.DataFrame(
        {"actual": targets, "predicted": probabilities, "band": bins}
    ).groupby("band", observed=True).agg(
        rows=("actual", "size"),
        mean_probability=("predicted", "mean"),
        observed_rate=("actual", "mean"),
    )
    return {
        "log_loss": float(
            -np.mean(
                targets * np.log(probabilities)
                + (1 - targets) * np.log(1 - probabilities)
            )
        ),
        "brier": float(np.mean(np.square(probabilities - targets))),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "calibration_error": float(
            np.average(
                np.abs(
                    calibration["mean_probability"] - calibration["observed_rate"]
                ),
                weights=calibration["rows"],
            )
        ),
        "actual_rate": float(targets.mean()),
        "predicted_rate": float(probabilities.mean()),
    }


def calculate_probability_metrics(
    rows: pd.DataFrame, probabilities: np.ndarray
) -> dict[str, float]:
    """Evaluate coherent state and derived binary probabilities."""
    if probabilities.shape != (len(rows), len(MINUTES_STATES)):
        raise ValueError("Probability rows do not match validation rows")
    appeared_column = (
        "target_appeared" if "target_appeared" in rows else "actual_appeared"
    )
    started_column = (
        "target_started" if "target_started" in rows else "actual_started"
    )
    sixty_plus_column = (
        "target_sixty_plus" if "target_sixty_plus" in rows else "actual_sixty_plus"
    )
    state_column = (
        "minutes_state" if "minutes_state" in rows else "actual_minutes_state"
    )
    available_state = rows["start_label_available"].to_numpy(dtype=bool)
    actual_states = rows.loc[available_state, state_column].to_numpy()
    actual_indices = np.array(
        [MINUTES_STATES.index(str(state)) for state in actual_states]
    )
    available_probabilities = probabilities[available_state]
    state_log_loss = -np.log(
        np.clip(
            available_probabilities[np.arange(len(actual_indices)), actual_indices],
            1e-15,
            1,
        )
    ).mean()
    state_targets = np.eye(len(MINUTES_STATES))[actual_indices]

    appearance_probability = probabilities[:, 1:].sum(axis=1)
    start_probability = probabilities[:, 3:].sum(axis=1)
    sixty_plus_probability = probabilities[:, [2, 4]].sum(axis=1)
    appearance = _binary_metrics(
        rows[appeared_column].to_numpy(), appearance_probability
    )
    starts = _binary_metrics(
        rows.loc[available_state, started_column].to_numpy(),
        start_probability[available_state],
    )
    sixty_plus = _binary_metrics(
        rows[sixty_plus_column].to_numpy(), sixty_plus_probability
    )
    return {
        "rows": int(len(rows)),
        "state_label_rows": int(available_state.sum()),
        "state_log_loss": float(state_log_loss),
        "state_brier": float(
            np.mean(np.sum(np.square(available_probabilities - state_targets), axis=1))
        ),
        "state_accuracy": float(
            (available_probabilities.argmax(axis=1) == actual_indices).mean()
        ),
        **{f"appearance_{key}": value for key, value in appearance.items()},
        **{f"start_{key}": value for key, value in starts.items()},
        **{f"sixty_plus_{key}": value for key, value in sixty_plus.items()},
    }


def calculate_all_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Evaluate states, probabilities, and expected minutes together."""
    metrics = calculate_probability_metrics(
        predictions,
        predictions[PROBABILITY_COLUMNS].to_numpy(),
    )
    minute_error = predictions["expected_minutes"] - predictions["actual_minutes"]
    appeared = predictions["actual_appeared"].eq(1)
    metrics.update(
        {
            "minutes_mae": float(minute_error.abs().mean()),
            "minutes_rmse": float(np.sqrt(np.mean(np.square(minute_error)))),
            "minutes_bias": float(minute_error.mean()),
            "mean_actual_minutes": float(predictions["actual_minutes"].mean()),
            "mean_expected_minutes": float(predictions["expected_minutes"].mean()),
            "appeared_minutes_mae": float(
                (
                    predictions.loc[appeared, "expected_minutes_if_appears"]
                    - predictions.loc[appeared, "actual_minutes"]
                )
                .abs()
                .mean()
            ),
        }
    )
    return metrics


def fit_classifier_folds(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    candidate: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Fit appearance and appeared-state classifiers on each fold."""
    profile = str(candidate["feature_profile"])
    parameters = candidate["parameters"]
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    predicted_inputs = []
    fold_metrics = []

    for season, training, validation in splits:
        appearance_model = build_state_classifier(
            profile, parameters, random_state=int(settings["random_state"])
        )
        appearance_model.fit(
            training[feature_columns], training["target_appeared"]
        )
        state_training = training.loc[
            training["start_label_available"] & training["target_appeared"].eq(1)
        ]
        appeared_state_model = build_state_classifier(
            profile, parameters, random_state=int(settings["random_state"])
        )
        appeared_state_model.fit(
            state_training[feature_columns], state_training["minutes_state"]
        )
        probabilities = predict_state_probabilities(
            appearance_model,
            appeared_state_model,
            validation,
            feature_columns,
        )
        fold_inputs = validation.copy()
        for index, column in enumerate(PROBABILITY_COLUMNS):
            fold_inputs[column] = probabilities[:, index]
        predicted_inputs.append(fold_inputs)
        fold_metrics.append(
            {
                "validation_season": season,
                **calculate_probability_metrics(validation, probabilities),
            }
        )
    return pd.concat(predicted_inputs, ignore_index=True), fold_metrics


def tune_classifiers(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    candidates: Sequence[dict[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every hierarchical classifier candidate."""
    results = []
    fold_results = []
    for index, candidate in enumerate(candidates, start=1):
        predictions, folds = fit_classifier_folds(splits, candidate, settings)
        metrics = calculate_probability_metrics(
            predictions, predictions[PROBABILITY_COLUMNS].to_numpy()
        )
        parameters = candidate["parameters"]
        results.append(
            {
                "candidate": index,
                "feature_profile": candidate["feature_profile"],
                **{f"parameter_{key}": value for key, value in parameters.items()},
                **metrics,
            }
        )
        fold_results.extend(
            {
                "candidate": index,
                "feature_profile": candidate["feature_profile"],
                **{f"parameter_{key}": value for key, value in parameters.items()},
                **fold,
            }
            for fold in folds
        )
        print(
            f"Classifier {index:>2}/{len(candidates)}: "
            f"{candidate['feature_profile']} state log loss="
            f"{metrics['state_log_loss']:.4f}, appearance Brier="
            f"{metrics['appearance_brier']:.4f}"
        )
    results_frame = pd.DataFrame(results).sort_values(
        ["state_log_loss", "appearance_log_loss", "sixty_plus_log_loss"]
    )
    results_frame.insert(0, "rank", range(1, len(results_frame) + 1))
    return results_frame, pd.DataFrame(fold_results)


def tune_temperature(
    probability_inputs: pd.DataFrame, settings: Mapping[str, object]
) -> pd.DataFrame:
    """Tune one probability temperature on walk-forward state forecasts."""
    temperature_settings = settings["probability_temperature"]
    minimum = float(temperature_settings["minimum"])
    maximum = float(temperature_settings["maximum"])
    step = float(temperature_settings["step"])
    raw_probabilities = probability_inputs[PROBABILITY_COLUMNS].to_numpy()
    results = []
    for temperature in np.arange(minimum, maximum + step / 2, step):
        probabilities = apply_temperature(raw_probabilities, float(temperature))
        metrics = calculate_probability_metrics(probability_inputs, probabilities)
        results.append(
            {
                "temperature": round(float(temperature), 10),
                "state_log_loss": metrics["state_log_loss"],
                "appearance_brier": metrics["appearance_brier"],
                "start_brier": metrics["start_brier"],
                "sixty_plus_brier": metrics["sixty_plus_brier"],
            }
        )
    return pd.DataFrame(results).sort_values(
        ["state_log_loss", "appearance_brier"]
    )


def apply_probability_temperature(
    probability_inputs: pd.DataFrame, temperature: float
) -> pd.DataFrame:
    output = probability_inputs.copy()
    probabilities = apply_temperature(
        output[PROBABILITY_COLUMNS].to_numpy(), temperature
    )
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probabilities[:, index]
    return output


def fit_duration_folds(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    probability_inputs: pd.DataFrame,
    profile: str,
    parameters: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Fit state-conditioned duration models and form complete predictions."""
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    predictions = []
    fold_metrics = []
    for season, training, _ in splits:
        duration_training = training.loc[
            training["start_label_available"] & training["target_appeared"].eq(1)
        ].copy()
        duration_model = build_duration_regressor(
            profile, parameters, random_state=int(settings["random_state"])
        )
        duration_model.fit(
            duration_training[[*feature_columns, "minutes_state"]],
            duration_training["target_minutes"],
        )
        validation = probability_inputs.loc[
            probability_inputs["season"].eq(season)
        ].copy()
        durations = predict_state_durations(
            duration_model, validation, feature_columns
        )
        fold_predictions = build_prediction_frame(
            validation,
            validation[PROBABILITY_COLUMNS].to_numpy(),
            durations,
        )
        predictions.append(fold_predictions)
        fold_metrics.append(
            {"validation_season": season, **calculate_all_metrics(fold_predictions)}
        )
    return pd.concat(predictions, ignore_index=True), fold_metrics


def tune_durations(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    probability_inputs: pd.DataFrame,
    profile: str,
    candidates: Sequence[dict[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune the duration regressor against expected-minutes accuracy."""
    results = []
    fold_results = []
    for index, parameters in enumerate(candidates, start=1):
        predictions, folds = fit_duration_folds(
            splits, probability_inputs, profile, parameters, settings
        )
        metrics = calculate_all_metrics(predictions)
        results.append(
            {
                "candidate": index,
                **{f"parameter_{key}": value for key, value in parameters.items()},
                **metrics,
            }
        )
        fold_results.extend(
            {
                "candidate": index,
                **{f"parameter_{key}": value for key, value in parameters.items()},
                **fold,
            }
            for fold in folds
        )
        print(
            f"Duration {index:>2}/{len(candidates)}: minutes RMSE="
            f"{metrics['minutes_rmse']:.4f}, minutes MAE="
            f"{metrics['minutes_mae']:.4f}"
        )
    results_frame = pd.DataFrame(results).sort_values(
        ["minutes_rmse", "minutes_mae", "appeared_minutes_mae"]
    )
    results_frame.insert(0, "rank", range(1, len(results_frame) + 1))
    return results_frame, pd.DataFrame(fold_results)


def apply_duration_offset(
    predictions: pd.DataFrame, offset: float
) -> pd.DataFrame:
    """Apply one bounded offset to all appeared-state conditional means."""
    output = predictions.copy()
    expected_minutes = np.zeros(len(output), dtype=float)
    for state in APPEARED_STATES:
        lower, upper = STATE_MINUTE_BOUNDS[state]
        duration_column = f"minutes_if_{state}"
        output[duration_column] = np.clip(
            output[duration_column] + offset, lower, upper
        )
        expected_minutes += (
            output[f"probability_{state}"] * output[duration_column]
        )
    output["expected_minutes"] = expected_minutes
    output["expected_minutes_if_appears"] = np.divide(
        expected_minutes,
        output["appearance_probability"],
        out=np.zeros_like(expected_minutes),
        where=output["appearance_probability"].to_numpy() > 0,
    )
    return output


def tune_duration_offset(
    predictions: pd.DataFrame, settings: Mapping[str, object]
) -> pd.DataFrame:
    """Tune a bounded duration offset against walk-forward minutes RMSE."""
    offset_settings = settings["duration_offset"]
    minimum = float(offset_settings["minimum"])
    maximum = float(offset_settings["maximum"])
    step = float(offset_settings["step"])
    results = []
    for offset in np.arange(minimum, maximum + step / 2, step):
        metrics = calculate_all_metrics(
            apply_duration_offset(predictions, float(offset))
        )
        results.append(
            {
                "duration_offset": round(float(offset), 10),
                "minutes_rmse": metrics["minutes_rmse"],
                "minutes_mae": metrics["minutes_mae"],
                "minutes_bias": metrics["minutes_bias"],
            }
        )
    return pd.DataFrame(results).sort_values(
        ["minutes_rmse", "minutes_mae"]
    )


def baseline_walk_forward(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Build a smoothed position-level empirical state benchmark."""
    predictions = []
    for _, training, validation in splits:
        appeared_counts = training.groupby("position")["target_appeared"].agg(
            ["sum", "count"]
        )
        appeared_rates = (appeared_counts["sum"] + 1) / (
            appeared_counts["count"] + 2
        )
        overall_appeared = (training["target_appeared"].sum() + 1) / (
            len(training) + 2
        )

        reliable_appeared = training.loc[
            training["start_label_available"] & training["target_appeared"].eq(1)
        ]
        state_counts = pd.crosstab(
            reliable_appeared["position"], reliable_appeared["minutes_state"]
        ).reindex(columns=APPEARED_STATES, fill_value=0)
        state_rates = (state_counts + 1).div(
            state_counts.sum(axis=1) + len(APPEARED_STATES), axis=0
        )
        overall_state = (
            reliable_appeared["minutes_state"]
            .value_counts()
            .reindex(APPEARED_STATES, fill_value=0)
            .add(1)
        )
        overall_state = overall_state / overall_state.sum()

        appeared_probability = (
            validation["position"].map(appeared_rates).fillna(overall_appeared)
        ).to_numpy()
        conditional_probabilities = np.column_stack(
            [
                validation["position"]
                .map(state_rates[state])
                .fillna(overall_state[state])
                .to_numpy()
                for state in APPEARED_STATES
            ]
        )
        probabilities = np.column_stack(
            [
                1 - appeared_probability,
                appeared_probability[:, None] * conditional_probabilities,
            ]
        )

        duration_means = reliable_appeared.groupby(
            ["position", "minutes_state"]
        )["target_minutes"].mean()
        overall_durations = reliable_appeared.groupby("minutes_state")[
            "target_minutes"
        ].mean()
        durations = np.column_stack(
            [
                np.clip(
                    [
                        duration_means.get(
                            (position, state), overall_durations[state]
                        )
                        for position in validation["position"]
                    ],
                    *STATE_MINUTE_BOUNDS[state],
                )
                for state in APPEARED_STATES
            ]
        )
        predictions.append(
            build_prediction_frame(validation, probabilities, durations)
        )
    return pd.concat(predictions, ignore_index=True)


def previous_minutes_baseline(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> dict[str, object]:
    """Evaluate carrying the player's previous fixture minutes forward."""
    predictions = pd.concat(
        [
            validation[["season", "target_minutes"]].assign(
                expected_minutes=validation["feature_player_previous_minutes"]
                .fillna(0)
                .clip(0, 90)
            )
            for _, _, validation in splits
        ],
        ignore_index=True,
    )

    def metrics(frame: pd.DataFrame) -> dict[str, float]:
        error = frame["expected_minutes"] - frame["target_minutes"]
        return {
            "rows": int(len(frame)),
            "minutes_mae": float(error.abs().mean()),
            "minutes_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "minutes_bias": float(error.mean()),
            "mean_actual_minutes": float(frame["target_minutes"].mean()),
            "mean_expected_minutes": float(frame["expected_minutes"].mean()),
        }

    return {
        "all_validation_seasons": metrics(predictions),
        "by_season": {
            str(season): metrics(
                predictions.loc[predictions["season"].eq(season)]
            )
            for season in predictions["season"].drop_duplicates()
        },
    }


def metrics_by_season(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(season): calculate_all_metrics(
            predictions.loc[predictions["season"].eq(season)]
        )
        for season in predictions["season"].drop_duplicates()
    }


def probability_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    targets = [
        ("appearance", "appearance_probability", "actual_appeared"),
        ("start", "start_probability", "actual_started"),
        ("sixty_plus", "sixty_plus_probability", "actual_sixty_plus"),
    ]
    records = []
    for target, prediction_column, actual_column in targets:
        target_rows = predictions
        if target == "start":
            target_rows = predictions.loc[predictions["start_label_available"]]
        bands = pd.cut(
            target_rows[prediction_column],
            bins=np.linspace(0, 1, 11),
            include_lowest=True,
        )
        summary = (
            target_rows.assign(probability_band=bands)
            .groupby("probability_band", observed=True)
            .agg(
                rows=(actual_column, "size"),
                mean_probability=(prediction_column, "mean"),
                observed_rate=(actual_column, "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "target", target)
        summary["probability_band"] = summary["probability_band"].astype(str)
        records.append(summary)
    return pd.concat(records, ignore_index=True)


def minutes_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[["expected_minutes", "actual_minutes"]].copy()
    frame["prediction_decile"] = pd.qcut(
        frame["expected_minutes"], 10, duplicates="drop"
    )
    summary = (
        frame.groupby("prediction_decile", observed=True)
        .agg(
            rows=("actual_minutes", "size"),
            mean_expected_minutes=("expected_minutes", "mean"),
            mean_actual_minutes=("actual_minutes", "mean"),
        )
        .reset_index()
    )
    summary["prediction_decile"] = summary["prediction_decile"].astype(str)
    return summary


def feature_coverage(rows: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "feature": feature,
                "available_rows": int(season_rows[feature].notna().sum()),
                "player_fixture_rows": int(len(season_rows)),
                "coverage": float(season_rows[feature].notna().mean()),
            }
            for season, season_rows in rows.groupby("season", sort=False)
            for feature in feature_columns
        ]
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_artifacts(
    rows: pd.DataFrame,
    settings: Mapping[str, object],
    profile: str,
    classifier_parameters: Mapping[str, object],
    duration_parameters: Mapping[str, object],
    temperature: float,
    appearance_model: object,
    appeared_state_model: object,
    duration_model: object,
    classifier_results: pd.DataFrame,
    classifier_fold_results: pd.DataFrame,
    temperature_results: pd.DataFrame,
    duration_results: pd.DataFrame,
    duration_fold_results: pd.DataFrame,
    duration_offset_results: pd.DataFrame,
    duration_offset: float,
    predictions: pd.DataFrame,
    model_metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
    previous_minutes_metrics: Mapping[str, object],
) -> Path:
    """Write the model and all reproducibility and evaluation artifacts."""
    model_dir = Path(str(settings.get("model_dir", DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_version": 1,
        "model_type": "minutes_state",
        "trained_at": trained_at,
        "appearance_model": appearance_model,
        "appeared_state_model": appeared_state_model,
        "duration_model": duration_model,
        "feature_profile": profile,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classifier_parameters": dict(classifier_parameters),
        "duration_parameters": dict(duration_parameters),
        "duration_offset": duration_offset,
        "probability_temperature": temperature,
        "states": MINUTES_STATES,
    }
    joblib.dump(artifact, model_dir / "model.joblib")

    classifier_results.to_csv(model_dir / "classifier_tuning_results.csv", index=False)
    classifier_fold_results.to_csv(
        model_dir / "classifier_tuning_fold_results.csv", index=False
    )
    temperature_results.to_csv(model_dir / "temperature_tuning.csv", index=False)
    duration_results.to_csv(model_dir / "duration_tuning_results.csv", index=False)
    duration_fold_results.to_csv(
        model_dir / "duration_tuning_fold_results.csv", index=False
    )
    duration_offset_results.to_csv(
        model_dir / "duration_offset_tuning.csv", index=False
    )
    predictions.to_csv(model_dir / "walk_forward_predictions.csv", index=False)
    probability_calibration(predictions).to_csv(
        model_dir / "probability_calibration.csv", index=False
    )
    minutes_calibration(predictions).to_csv(
        model_dir / "minutes_calibration.csv", index=False
    )
    feature_coverage(rows, feature_columns).to_csv(
        model_dir / "feature_coverage.csv", index=False
    )
    market_features = [
        feature
        for feature in [*STABLE_MARKET_FEATURES, *VOLATILE_MARKET_FEATURES]
        if feature in feature_columns
    ]
    if market_features:
        market_distribution = (
            rows.groupby("season")[market_features]
            .agg(["mean", "std", "min", "max"])
            .stack(level=0, future_stack=True)
            .rename_axis(["season", "feature"])
            .reset_index()
        )
    else:
        market_distribution = pd.DataFrame(
            columns=["season", "feature", "mean", "std", "min", "max"]
        )
    market_distribution.to_csv(
        model_dir / "market_feature_distribution.csv", index=False
    )

    state_distribution = pd.DataFrame(
        {
            "state": MINUTES_STATES,
            "actual_rate": [
                predictions["actual_minutes_state"].eq(state).mean()
                for state in MINUTES_STATES
            ],
            "predicted_rate": [
                predictions[f"probability_{state}"].mean()
                for state in MINUTES_STATES
            ],
        }
    )
    state_distribution.to_csv(model_dir / "state_distribution.csv", index=False)

    write_json(
        model_dir / "best_parameters.json",
        {
            "feature_profile": profile,
            "appearance_and_state_estimator": "HistGradientBoostingClassifier",
            "classifier_parameters": dict(classifier_parameters),
            "duration_estimator": "HistGradientBoostingRegressor",
            "duration_loss": "squared_error",
            "duration_parameters": dict(duration_parameters),
            "duration_offset": duration_offset,
            "probability_temperature": temperature,
            "classifier_selection_metric": "walk-forward state log loss",
            "duration_selection_metric": "walk-forward expected-minutes RMSE",
            "duration_offset_selection_metric": "walk-forward expected-minutes RMSE",
        },
    )
    write_json(
        model_dir / "metrics.json",
        {
            "walk_forward_model": model_metrics,
            "walk_forward_position_baseline": baseline_metrics,
            "walk_forward_previous_minutes_baseline": previous_minutes_metrics,
        },
    )
    write_json(
        model_dir / "feature_schema.json",
        {
            "required_identifiers": IDENTIFIER_COLUMNS,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "states": MINUTES_STATES,
        },
    )

    unreliable = rows.loc[~rows["start_label_available"]]
    unavailable_gameweeks = (
        unreliable[["season", "gameweek"]]
        .drop_duplicates()
        .groupby("season")["gameweek"]
        .apply(lambda values: sorted(int(value) for value in values))
        .to_dict()
    )
    metadata = {
        "model_type": "minutes_state",
        "trained_at": trained_at,
        "source_dataset": str(settings["input_path"]),
        "source": "Vaastav Fantasy Premier League repository",
        "training_player_fixture_rows": int(len(rows)),
        "training_seasons": rows["season"].drop_duplicates().tolist(),
        "training_cutoff": str(rows["kickoff_time"].max()),
        "validation_seasons": list(settings["validation_seasons"]),
        "validation_method": "expanding-window season walk-forward",
        "feature_profile": profile,
        "feature_count": len(feature_columns),
        "outputs": [
            "appearance probability",
            "start probability",
            "60+ probability",
            "five mutually exclusive state probabilities",
            "expected minutes",
            "expected minutes conditional on appearance",
        ],
        "state_consistency": (
            "DNP and four appeared states form one probability distribution. Start "
            "and 60+ probabilities are sums of compatible states, and expected "
            "minutes is the state-probability-weighted feasible duration."
        ),
        "unavailable_start_label_gameweeks": unavailable_gameweeks,
        "unavailable_start_label_rows": int(len(unreliable)),
        "start_label_handling": (
            "A team-fixture start label is used only when exactly 11 starters are "
            "present. Unreliable rows remain available to the appearance model but "
            "are excluded from appeared-state and duration fitting."
        ),
        "excluded_start_features": (
            "Historical started-rate features are excluded because the 2022/23 "
            "start-data gap would encode false zeroes. Minutes, appearance and 60+ "
            "history remain available."
        ),
        "availability_caveat": (
            "The processed Vaastav rows do not contain explicit injury status or "
            "chance-of-playing fields. FPL expected points, weekly price, ownership "
            "and transfers provide limited pre-match market context but do not "
            "replace a current availability feed."
        ),
        "fpl_expected_points_mean_by_season": rows.groupby("season")[
            "feature_fpl_expected_points"
        ]
        .mean()
        .to_dict(),
        "market_feature_caveat": (
            "FPL expected-points and transfer fields change distribution between "
            "seasons. Their per-season distributions and all no-market/stable-market "
            "candidate results are retained for drift review."
        ),
    }
    write_json(model_dir / "metadata.json", metadata)

    latest_fixture_id = rows.sort_values(["kickoff_time", "fixture_id"]).iloc[-1][
        ["season", "fixture_id"]
    ]
    latest_fixture = rows.loc[
        rows["season"].eq(latest_fixture_id["season"])
        & rows["fixture_id"].eq(latest_fixture_id["fixture_id"])
    ].copy()
    latest_fixture = (
        latest_fixture.sort_values("feature_price_millions", ascending=False)
        .groupby("position", sort=False)
        .head(1)
        .head(4)
    )
    example_columns = list(dict.fromkeys(IDENTIFIER_COLUMNS + feature_columns))
    write_json(
        model_dir / "example_input.json",
        {
            "player_fixture_rows": records_for_json(
                latest_fixture[example_columns]
            )
        },
    )
    write_json(
        model_dir / "example_output.json",
        {
            "predictions": records_for_json(
                predict_with_artifact(artifact, latest_fixture)
            )
        },
    )
    return model_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="Training configuration."
    )
    parser.add_argument("--input", default=None, help="Override the processed CSV.")
    parser.add_argument("--model-dir", default=None, help="Override artifact output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.input:
        settings["input_path"] = args.input
    if args.model_dir:
        settings["model_dir"] = args.model_dir

    rows = load_minutes_training_data(str(settings["input_path"]))
    validation_seasons = [str(season) for season in settings["validation_seasons"]]
    splits = chronological_splits(rows, validation_seasons)
    candidates = classifier_candidates(settings)
    print(
        f"Loaded {len(rows):,} player-fixture rows; tuning {len(candidates)} "
        f"hierarchical classifier candidates across {len(splits)} seasons"
    )
    unavailable = rows.loc[~rows["start_label_available"]]
    print(
        f"Excluded {len(unavailable):,} rows from start-state fitting because "
        "their team-fixture did not contain 11 start labels"
    )

    classifier_results, classifier_fold_results = tune_classifiers(
        splits, candidates, settings
    )
    best_classifier_row = classifier_results.iloc[0]
    best_classifier_number = int(best_classifier_row["candidate"])
    best_classifier = candidates[best_classifier_number - 1]
    probability_inputs, _ = fit_classifier_folds(
        splits, best_classifier, settings
    )
    temperature_results = tune_temperature(probability_inputs, settings)
    best_temperature = float(temperature_results.iloc[0]["temperature"])
    probability_inputs = apply_probability_temperature(
        probability_inputs, best_temperature
    )

    profile = str(best_classifier["feature_profile"])
    duration_parameter_candidates = duration_candidates(settings)
    duration_results, duration_fold_results = tune_durations(
        splits,
        probability_inputs,
        profile,
        duration_parameter_candidates,
        settings,
    )
    best_duration_row = duration_results.iloc[0]
    best_duration_number = int(best_duration_row["candidate"])
    best_duration_parameters = duration_parameter_candidates[
        best_duration_number - 1
    ]
    predictions, _ = fit_duration_folds(
        splits,
        probability_inputs,
        profile,
        best_duration_parameters,
        settings,
    )
    duration_offset_results = tune_duration_offset(predictions, settings)
    best_duration_offset = float(
        duration_offset_results.iloc[0]["duration_offset"]
    )
    predictions = apply_duration_offset(predictions, best_duration_offset)

    model_metrics = {
        "all_validation_seasons": calculate_all_metrics(predictions),
        "by_season": metrics_by_season(predictions),
    }
    baseline_predictions = baseline_walk_forward(splits)
    baseline_metrics = {
        "all_validation_seasons": calculate_all_metrics(baseline_predictions),
        "by_season": metrics_by_season(baseline_predictions),
    }
    previous_minutes_metrics = previous_minutes_baseline(splits)

    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    classifier_parameters = best_classifier["parameters"]
    appearance_model = build_state_classifier(
        profile, classifier_parameters, random_state=int(settings["random_state"])
    )
    appearance_model.fit(rows[feature_columns], rows["target_appeared"])
    reliable_appeared = rows.loc[
        rows["start_label_available"] & rows["target_appeared"].eq(1)
    ].copy()
    appeared_state_model = build_state_classifier(
        profile, classifier_parameters, random_state=int(settings["random_state"])
    )
    appeared_state_model.fit(
        reliable_appeared[feature_columns], reliable_appeared["minutes_state"]
    )
    duration_model = build_duration_regressor(
        profile,
        best_duration_parameters,
        random_state=int(settings["random_state"]),
    )
    duration_model.fit(
        reliable_appeared[[*feature_columns, "minutes_state"]],
        reliable_appeared["target_minutes"],
    )
    model_dir = write_artifacts(
        rows,
        settings,
        profile,
        classifier_parameters,
        best_duration_parameters,
        best_temperature,
        appearance_model,
        appeared_state_model,
        duration_model,
        classifier_results,
        classifier_fold_results,
        temperature_results,
        duration_results,
        duration_fold_results,
        duration_offset_results,
        best_duration_offset,
        predictions,
        model_metrics,
        baseline_metrics,
        previous_minutes_metrics,
    )

    metrics = model_metrics["all_validation_seasons"]
    print(f"Best feature profile: {profile}")
    print(
        f"Best classifier parameters: {json.dumps(classifier_parameters, sort_keys=True)}"
    )
    print(
        f"Best duration parameters: {json.dumps(best_duration_parameters, sort_keys=True)}"
    )
    print(f"Best probability temperature: {best_temperature:.2f}")
    print(f"Best duration offset: {best_duration_offset:.2f} minutes")
    print(
        f"Walk-forward: minutes RMSE={metrics['minutes_rmse']:.4f}, "
        f"minutes MAE={metrics['minutes_mae']:.4f}, "
        f"appearance Brier={metrics['appearance_brier']:.4f}, "
        f"start Brier={metrics['start_brier']:.4f}, "
        f"60+ Brier={metrics['sixty_plus_brier']:.4f}"
    )
    print(f"Wrote model and reports to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
