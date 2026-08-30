"""Tune and train disciplinary and rare-event components."""

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
from sklearn.metrics import mean_poisson_deviance

from .model import (
    DEFAULT_MODEL_DIR,
    EVENTS,
    IDENTIFIER_COLUMNS,
    PLAYER_KEYS,
    UPSTREAM_COLUMNS,
    build_predictions,
    build_yellow_card_model,
    get_feature_columns,
    load_training_data,
    position_rate_priors,
    predict_with_artifact,
    records_for_json,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "misc_events.yaml"


def load_settings(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = config.get("misc_events") if isinstance(config, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must contain a misc_events mapping: {path}")
    return settings


def candidates(settings: Mapping[str, object]) -> list[dict[str, object]]:
    grid = settings["parameter_grid"]
    names = list(grid)
    return [
        {"feature_profile": str(profile), "parameters": dict(zip(names, values))}
        for profile in settings["feature_profiles"]
        for values in itertools.product(*(grid[name] for name in names))
    ]


def chronological_splits(rows: pd.DataFrame, seasons: Sequence[str]) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    starts = rows[["season", "season_start"]].drop_duplicates().set_index("season")["season_start"].to_dict()
    return [
        (
            season,
            rows.loc[rows["season_start"].lt(starts[season])].copy(),
            rows.loc[rows["season"].eq(season)].copy(),
        )
        for season in seasons
    ]


def attach_minutes(rows: pd.DataFrame, settings: Mapping[str, object]) -> pd.DataFrame:
    minutes = pd.read_csv(
        str(settings["minutes_predictions_path"]),
        usecols=PLAYER_KEYS + ["expected_minutes"],
    )
    joined = rows.merge(minutes, on=PLAYER_KEYS, how="left", validate="one_to_one")
    missing = joined["expected_minutes"].isna()
    if missing.any():
        raise ValueError(f"Minutes predictions are missing for {int(missing.sum())} miscellaneous-event rows")
    return joined


def _count_metrics(actual: Sequence[float], expected: Sequence[float]) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.clip(np.asarray(expected, dtype=float), 1e-8, None)
    error = expected_array - actual_array
    probability = 1 - np.exp(-expected_array)
    actual_event = actual_array > 0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "poisson_deviance": float(mean_poisson_deviance(actual_array, expected_array)),
        "brier": float(np.mean(np.square(probability - actual_event))),
        "bias": float(error.mean()),
        "mean_actual": float(actual_array.mean()),
        "mean_expected": float(expected_array.mean()),
    }


def fit_yellow_folds(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    upstream: pd.DataFrame,
    candidate: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    predictions, fold_metrics = [], []
    profile = str(candidate["feature_profile"])
    features = sum((list(part) for part in get_feature_columns(profile)), [])
    for season, training, _ in splits:
        fit_rows = training.loc[training["target_minutes"].gt(0)].copy()
        fit_rows["rate_target"] = 90 * fit_rows["target_yellow_cards"] / fit_rows["target_minutes"]
        model = build_yellow_card_model(profile, candidate["parameters"], random_state=int(settings["random_state"]))
        model.fit(fit_rows[features], fit_rows["rate_target"])
        validation = upstream.loc[upstream["season"].eq(season)].copy()
        validation["raw_yellow_rate"] = model.predict(validation[features])
        predictions.append(validation)
        metrics = _count_metrics(
            validation["target_yellow_cards"],
            np.clip(validation["raw_yellow_rate"], 0, None) * validation["expected_minutes"] / 90,
        )
        fold_metrics.append({"validation_season": season, **metrics})
    return pd.concat(predictions, ignore_index=True), fold_metrics


def tune_yellow_models(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    upstream: pd.DataFrame,
    model_candidates: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    summaries, folds, prediction_sets = [], [], {}
    for number, candidate in enumerate(model_candidates, 1):
        predictions, candidate_folds = fit_yellow_folds(splits, upstream, candidate, settings)
        prediction_sets[number] = predictions
        expected = np.clip(predictions["raw_yellow_rate"], 0, None) * predictions["expected_minutes"] / 90
        metrics = _count_metrics(predictions["target_yellow_cards"], expected)
        summaries.append({"candidate": number, **candidate, **metrics})
        folds.extend({"candidate": number, **fold} for fold in candidate_folds)
        print(f"Yellow-card candidate {number}/{len(model_candidates)}: deviance={metrics['poisson_deviance']:.4f}")
    return (
        pd.DataFrame(summaries).sort_values(["poisson_deviance", "brier"]).reset_index(drop=True),
        pd.DataFrame(folds),
        prediction_sets,
    )


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        event: _count_metrics(predictions[f"actual_{event}"], predictions[f"expected_{event}"])
        for event in EVENTS
    }


def flatten_metrics(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {f"{event}_{name}": value for event, event_metrics in metrics.items() for name, value in event_metrics.items()}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, (np.integer, np.floating)) else str(value)) + "\n", encoding="utf-8")


def write_artifacts(
    rows: pd.DataFrame,
    inputs: pd.DataFrame,
    predictions: pd.DataFrame,
    settings: Mapping[str, object],
    candidate: Mapping[str, object],
    model: object,
    priors: Mapping[str, Mapping[str, float]],
    shrinkage: Mapping[str, float],
    tuning_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    shrinkage_results: pd.DataFrame,
    metrics: Mapping[str, Mapping[str, float]],
    baseline_metrics: Mapping[str, Mapping[str, float]],
) -> Path:
    model_dir = Path(str(settings.get("model_dir", DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    profile = str(candidate["feature_profile"])
    numeric, categorical = get_feature_columns(profile)
    feature_columns = numeric + categorical
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_version": 1,
        "model_type": "misc_events",
        "trained_at": trained_at,
        "yellow_card_model": model,
        "feature_profile": profile,
        "feature_columns": feature_columns,
        "position_priors": {event: dict(event_priors) for event, event_priors in priors.items()},
        "shrinkage_minutes": dict(shrinkage),
    }
    joblib.dump(artifact, model_dir / "model.joblib")
    tuning_results.to_csv(model_dir / "yellow_card_model_tuning.csv", index=False)
    fold_results.to_csv(model_dir / "yellow_card_model_tuning_folds.csv", index=False)
    shrinkage_results.to_csv(model_dir / "event_shrinkage_tuning.csv", index=False)
    predictions.to_csv(model_dir / "walk_forward_predictions.csv", index=False)
    pd.DataFrame([
        {"season": season, "target": definition["target"], "rows": len(group), "events": float(group[str(definition["target"])].sum()), "positive_rows": int(group[str(definition["target"])].gt(0).sum())}
        for season, group in rows.groupby("season")
        for definition in EVENTS.values()
    ]).to_csv(model_dir / "target_coverage.csv", index=False)
    write_json(model_dir / "best_parameters.json", {
        "yellow_card_model": candidate,
        "shrinkage_minutes": shrinkage,
        "selection_metric": "walk-forward Poisson deviance per event",
    })
    write_json(model_dir / "metrics.json", {
        "walk_forward_model": metrics,
        "walk_forward_position_prior_baseline": baseline_metrics,
    })
    write_json(model_dir / "feature_schema.json", {
        "required_identifiers": IDENTIFIER_COLUMNS,
        "required_upstream_predictions": UPSTREAM_COLUMNS,
        "numeric_features": numeric,
        "categorical_features": categorical,
    })
    write_json(model_dir / "metadata.json", {
        "model_type": "misc_events",
        "trained_at": trained_at,
        "source_dataset": str(settings["input_path"]),
        "source": "Vaastav Fantasy Premier League repository",
        "validation_method": "expanding-window season walk-forward",
        "yellow_card_method": "Poisson gradient boosting per 90, shrunk to position rates and scaled by forecast minutes.",
        "rare_event_method": "Pre-match current- and previous-season player counts are shrunk to position priors, then scaled by forecast minutes.",
        "penalty_miss_caveat": "Vaastav does not provide a current penalty-taker assignment or future penalty opportunity. This is a heavily shrunk per-minute event forecast, not a conditional miss probability.",
        "simulation_caveat": "Red-card effects on minutes and mutual dependence between cards are deferred to the joint simulation layer.",
        "outputs": ["yellow cards", "red cards", "own goals", "penalties missed", "expected FPL deductions"],
    })
    latest = inputs.sort_values(["kickoff_time", "fixture_id"]).iloc[-1]
    example = inputs.loc[inputs["season"].eq(latest["season"]) & inputs["fixture_id"].eq(latest["fixture_id"])].copy()
    example_columns = list(dict.fromkeys(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS + feature_columns))
    write_json(model_dir / "example_input.json", {"player_fixture_rows": records_for_json(example[example_columns])})
    write_json(model_dir / "example_output.json", {"predictions": records_for_json(predict_with_artifact(artifact, example))})
    return model_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input", default=None)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.input:
        settings["input_path"] = args.input
    if args.model_dir:
        settings["model_dir"] = args.model_dir

    rows = load_training_data(str(settings["input_path"]))
    validation_seasons = [str(season) for season in settings["validation_seasons"]]
    upstream = attach_minutes(rows.loc[rows["season"].isin(validation_seasons)].copy(), settings)
    model_candidates = candidates(settings)
    print(f"Loaded {len(rows):,} rows; tuning {len(model_candidates)} yellow-card candidates")
    tuning_results, fold_results, prediction_sets = tune_yellow_models(
        chronological_splits(rows, validation_seasons), upstream, model_candidates, settings
    )
    best_number = int(tuning_results.iloc[0]["candidate"])
    best_candidate = model_candidates[best_number - 1]
    inputs = prediction_sets[best_number]

    prior_training = rows.loc[rows["season"].eq("2022-23")]
    evaluation_priors = {
        event: position_rate_priors(prior_training, str(definition["target"]))
        for event, definition in EVENTS.items()
    }
    final_priors = {
        event: position_rate_priors(rows, str(definition["target"]))
        for event, definition in EVENTS.items()
    }
    tuning_rows = []
    selected_shrinkage: dict[str, float] = {}
    event_predictions: dict[str, tuple[float, float]] = {}
    for yellow_minutes in settings["yellow_card_shrinkage_minutes"]:
        shrinkage = {event: float(yellow_minutes if event == "yellow_cards" else 1800) for event in EVENTS}
        predicted = build_predictions(inputs, inputs["raw_yellow_rate"], evaluation_priors, shrinkage)
        metrics = calculate_metrics(predicted)["yellow_cards"]
        tuning_rows.append({"event": "yellow_cards", "shrinkage_minutes": yellow_minutes, **metrics})
        event_predictions[f"yellow_cards:{yellow_minutes}"] = (metrics["poisson_deviance"], float(yellow_minutes))
    selected_shrinkage["yellow_cards"] = min(event_predictions.values())[1]
    for event in ("red_cards", "own_goals", "penalties_missed"):
        choices = []
        for event_minutes in settings["rare_event_shrinkage_minutes"]:
            shrinkage = {name: float(selected_shrinkage.get(name, 1800)) for name in EVENTS}
            shrinkage[event] = float(event_minutes)
            predicted = build_predictions(inputs, inputs["raw_yellow_rate"], evaluation_priors, shrinkage)
            metrics = calculate_metrics(predicted)[event]
            tuning_rows.append({"event": event, "shrinkage_minutes": event_minutes, **metrics})
            choices.append((metrics["poisson_deviance"], float(event_minutes)))
        selected_shrinkage[event] = min(choices)[1]
    shrinkage_results = pd.DataFrame(tuning_rows).sort_values(["event", "poisson_deviance"]).reset_index(drop=True)
    predictions = build_predictions(inputs, inputs["raw_yellow_rate"], evaluation_priors, selected_shrinkage)
    metrics = calculate_metrics(predictions)
    baseline = build_predictions(
        inputs,
        inputs["position"].map(evaluation_priors["yellow_cards"]).fillna(evaluation_priors["yellow_cards"]["__default__"]),
        evaluation_priors,
        {event: 1e9 for event in EVENTS},
    )
    baseline_metrics = calculate_metrics(baseline)

    profile = str(best_candidate["feature_profile"])
    features = sum((list(part) for part in get_feature_columns(profile)), [])
    fit_rows = rows.loc[rows["target_minutes"].gt(0)].copy()
    fit_rows["rate_target"] = 90 * fit_rows["target_yellow_cards"] / fit_rows["target_minutes"]
    model = build_yellow_card_model(profile, best_candidate["parameters"], random_state=int(settings["random_state"]))
    model.fit(fit_rows[features], fit_rows["rate_target"])
    model_dir = write_artifacts(
        rows, inputs, predictions, settings, best_candidate, model, final_priors,
        selected_shrinkage, tuning_results, fold_results, shrinkage_results,
        metrics, baseline_metrics,
    )
    print(f"Best yellow-card profile: {profile}")
    for event, event_metrics in metrics.items():
        print(f"{event}: deviance={event_metrics['poisson_deviance']:.4f}, Brier={event_metrics['brier']:.4f}")
    print(f"Wrote model and reports to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
