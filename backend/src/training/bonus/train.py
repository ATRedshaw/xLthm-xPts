"""Tune conditional BPS prediction and fixture-level bonus allocation."""

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

from .model import (
    DEFAULT_MODEL_DIR,
    FIXTURE_KEYS,
    IDENTIFIER_COLUMNS,
    PLAYER_KEYS,
    UPSTREAM_COLUMNS,
    build_bps_model,
    build_bps_predictions,
    get_feature_columns,
    load_training_data,
    predict_with_artifact,
    records_for_json,
    simulate_bonus,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "bonus.yaml"


def load_settings(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = config.get("bonus") if isinstance(config, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must contain a bonus mapping: {path}")
    return settings


def candidates(settings: Mapping[str, object]) -> list[dict[str, object]]:
    grid = settings["parameter_grid"]
    names = list(grid)
    return [
        {"feature_profile": str(profile), "parameters": dict(zip(names, values))}
        for profile in settings["feature_profiles"]
        for values in itertools.product(*(grid[name] for name in names))
    ]


def attach_upstream(rows: pd.DataFrame, settings: Mapping[str, object]) -> pd.DataFrame:
    minutes = pd.read_csv(
        str(settings["minutes_predictions_path"]),
        usecols=PLAYER_KEYS + ["appearance_probability", "start_probability", "sixty_plus_probability", "expected_minutes"],
    )
    team = pd.read_csv(
        str(settings["team_predictions_path"]),
        usecols=["season", "fixture_id", "team_key", "expected_goals_for", "expected_goals_against", "clean_sheet_probability"],
    ).rename(columns={
        "expected_goals_for": "team_expected_goals_for",
        "expected_goals_against": "team_expected_goals_against",
        "clean_sheet_probability": "team_clean_sheet_probability",
    })
    attacking = pd.read_csv(
        str(settings["attacking_predictions_path"]),
        usecols=PLAYER_KEYS + ["expected_goals", "expected_fpl_assists"],
    )
    defensive = pd.read_csv(
        str(settings["defensive_predictions_path"]),
        usecols=PLAYER_KEYS + [
            "player_clean_sheet_probability", "expected_goals_conceded",
            "expected_saves", "expected_penalty_saves", "expected_defensive_contributions",
        ],
    )
    miscellaneous = pd.read_csv(
        str(settings["misc_events_predictions_path"]),
        usecols=PLAYER_KEYS + [
            "expected_yellow_cards", "expected_red_cards", "expected_own_goals",
            "expected_penalties_missed",
        ],
    )
    joined = rows.merge(minutes, on=PLAYER_KEYS, how="left", validate="one_to_one")
    joined = joined.merge(team, on=["season", "fixture_id", "team_key"], how="left", validate="many_to_one")
    for upstream in (attacking, defensive, miscellaneous):
        joined = joined.merge(upstream, on=PLAYER_KEYS, how="left", validate="one_to_one")
    missing = joined[UPSTREAM_COLUMNS].isna().any(axis=1)
    if missing.any():
        missing_columns = joined.loc[missing, UPSTREAM_COLUMNS].isna().sum()
        missing_columns = missing_columns.loc[missing_columns.gt(0)].to_dict()
        raise ValueError(f"Component predictions are incomplete: {missing_columns}")
    fixture_sizes = joined.groupby(FIXTURE_KEYS).size()
    if not fixture_sizes.ge(22).all():
        raise ValueError("Bonus allocation requires the full player set for every fixture")
    return joined


def chronological_splits(rows: pd.DataFrame, validation_seasons: Sequence[str]) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    starts = rows[["season", "season_start"]].drop_duplicates().set_index("season")["season_start"].to_dict()
    return [
        (
            season,
            rows.loc[rows["season_start"].lt(starts[season])].copy(),
            rows.loc[rows["season"].eq(season)].copy(),
        )
        for season in validation_seasons
    ]


def residual_std_by_position(rows: pd.DataFrame, predicted: Sequence[float]) -> dict[str, float]:
    appeared = rows["target_minutes"].gt(0).to_numpy()
    residuals = rows.loc[appeared, ["position"]].copy()
    residuals["residual"] = rows.loc[appeared, "target_bps"].to_numpy(float) - np.asarray(predicted, dtype=float)[appeared]
    values = residuals.groupby("position")["residual"].std().clip(lower=3).to_dict()
    values["__default__"] = max(3.0, float(residuals["residual"].std()))
    return {str(position): float(value) for position, value in values.items()}


def bps_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    error = predictions["expected_bps"].to_numpy() - predictions["actual_bps"].to_numpy()
    appeared = predictions["actual_appeared"].eq(1)
    appeared_error = (
        predictions.loc[appeared, "conditional_expected_bps"].to_numpy()
        - predictions.loc[appeared, "actual_bps"].to_numpy()
    )
    return {
        "bps_mae": float(np.mean(np.abs(error))),
        "bps_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bps_bias": float(error.mean()),
        "appeared_bps_mae": float(np.mean(np.abs(appeared_error))),
        "appeared_bps_rmse": float(np.sqrt(np.mean(np.square(appeared_error)))),
        "mean_actual_bps": float(predictions["actual_bps"].mean()),
        "mean_expected_bps": float(predictions["expected_bps"].mean()),
    }


def bonus_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["actual_bonus"].to_numpy(int)
    expected = predictions["expected_bonus"].to_numpy(float)
    probabilities = np.column_stack([
        predictions[f"bonus_{points}_probability"].to_numpy(float) for points in range(4)
    ])
    chosen = np.clip(probabilities[np.arange(len(actual)), actual], 1e-12, 1)
    any_probability = 1 - probabilities[:, 0]
    return {
        "bonus_mae": float(np.mean(np.abs(expected - actual))),
        "bonus_rmse": float(np.sqrt(np.mean(np.square(expected - actual)))),
        "bonus_bias": float((expected - actual).mean()),
        "bonus_log_loss": float(-np.log(chosen).mean()),
        "any_bonus_brier": float(np.mean(np.square(any_probability - (actual > 0)))),
        "mean_actual_bonus": float(actual.mean()),
        "mean_expected_bonus": float(expected.mean()),
    }


def fit_candidate_folds(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    candidate: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    predictions, fold_metrics = [], []
    profile = str(candidate["feature_profile"])
    features = sum((list(part) for part in get_feature_columns(profile)), [])
    for season, training, validation in splits:
        appeared = training["target_minutes"].gt(0)
        model = build_bps_model(profile, candidate["parameters"], random_state=int(settings["random_state"]))
        model.fit(training.loc[appeared, features], training.loc[appeared, "target_bps"])
        training_conditional = model.predict(training[features])
        residual_std = residual_std_by_position(training, training_conditional)
        conditional = model.predict(validation[features])
        predicted = build_bps_predictions(validation, conditional, residual_std)
        predictions.append(predicted)
        fold_metrics.append({"validation_season": season, **bps_metrics(predicted)})
    return pd.concat(predictions, ignore_index=True), fold_metrics


def tune_bps_models(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    model_candidates: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    summaries, folds, prediction_sets = [], [], {}
    for number, candidate in enumerate(model_candidates, 1):
        predictions, candidate_folds = fit_candidate_folds(splits, candidate, settings)
        prediction_sets[number] = predictions
        metrics = bps_metrics(predictions)
        summaries.append({"candidate": number, **candidate, **metrics})
        folds.extend({"candidate": number, **fold} for fold in candidate_folds)
        print(f"BPS candidate {number}/{len(model_candidates)}: appeared RMSE={metrics['appeared_bps_rmse']:.4f}")
    return (
        pd.DataFrame(summaries).sort_values(["appeared_bps_rmse", "appeared_bps_mae"]).reset_index(drop=True),
        pd.DataFrame(folds),
        prediction_sets,
    )


def metrics_by_season(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(season): {**bps_metrics(group), **bonus_metrics(group)}
        for season, group in predictions.groupby("season")
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, (np.integer, np.floating)) else str(value)) + "\n", encoding="utf-8")


def write_artifacts(
    rows: pd.DataFrame,
    inputs: pd.DataFrame,
    predictions: pd.DataFrame,
    settings: Mapping[str, object],
    candidate: Mapping[str, object],
    model: object,
    residual_std: Mapping[str, float],
    residual_scale: float,
    tuning_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    simulation_results: pd.DataFrame,
    metrics: Mapping[str, object],
) -> Path:
    model_dir = Path(str(settings.get("model_dir", DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    profile = str(candidate["feature_profile"])
    numeric, categorical = get_feature_columns(profile)
    feature_columns = numeric + categorical
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_version": 1,
        "model_type": "bonus",
        "trained_at": trained_at,
        "bps_model": model,
        "feature_profile": profile,
        "feature_columns": feature_columns,
        "residual_std_by_position": dict(residual_std),
        "residual_scale": residual_scale,
        "simulation_count": int(settings["final_simulations"]),
        "random_state": int(settings["random_state"]),
    }
    joblib.dump(artifact, model_dir / "model.joblib")
    tuning_results.to_csv(model_dir / "bps_model_tuning.csv", index=False)
    fold_results.to_csv(model_dir / "bps_model_tuning_folds.csv", index=False)
    simulation_results.to_csv(model_dir / "simulation_tuning.csv", index=False)
    predictions.to_csv(model_dir / "walk_forward_predictions.csv", index=False)
    predictions.groupby(FIXTURE_KEYS, as_index=False).agg(
        player_rows=("player_key", "size"),
        expected_bonus_points=("expected_bonus", "sum"),
        actual_bonus_points=("actual_bonus", "sum"),
    ).to_csv(model_dir / "walk_forward_fixture_totals.csv", index=False)
    pd.DataFrame([
        {"season": season, "rows": len(group), "appeared_rows": int(group["target_minutes"].gt(0).sum()), "bps_min": float(group["target_bps"].min()), "bps_max": float(group["target_bps"].max()), "bonus_points": float(group["target_bonus"].sum())}
        for season, group in rows.groupby("season")
    ]).to_csv(model_dir / "target_coverage.csv", index=False)
    write_json(model_dir / "best_parameters.json", {
        "bps_model": candidate,
        "residual_std_by_position": residual_std,
        "residual_scale": residual_scale,
        "tuning_simulations": int(settings["tuning_simulations"]),
        "final_simulations": int(settings["final_simulations"]),
        "selection_metrics": {"bps_model": "appeared-player BPS RMSE", "simulation": "bonus multiclass log loss"},
    })
    write_json(model_dir / "metrics.json", metrics)
    write_json(model_dir / "feature_schema.json", {
        "required_identifiers": IDENTIFIER_COLUMNS,
        "required_upstream_predictions": UPSTREAM_COLUMNS,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "batch_requirement": "All players from both teams in each fixture are required for bonus allocation.",
    })
    write_json(model_dir / "metadata.json", {
        "model_type": "bonus",
        "trained_at": trained_at,
        "source_dataset": str(settings["input_path"]),
        "source": "Vaastav Fantasy Premier League repository",
        "training_rows": int(len(inputs)),
        "training_cutoff": str(inputs["kickoff_time"].max()),
        "validation_seasons": list(settings["validation_seasons"]),
        "validation_method": "expanding-window season walk-forward using only upstream walk-forward component forecasts",
        "bps_method": "BPS is modelled conditional on appearance; unconditional expected BPS multiplies this by appearance probability.",
        "bonus_method": "Fixture-level Monte Carlo samples appearances and rounded BPS, then applies FPL 3/2/1 allocation including ties.",
        "simulation_caveat": "BPS residuals are independently sampled by player and position. A later joint event simulation should replace this approximation to capture correlated scorelines, minutes, and BPS actions.",
        "current_season_caveat": "Leakage-safe upstream component forecasts currently end at 2025/26, so supervised BPS fitting does too. The tie allocator is explicit, but scoring-rule changes should be represented in the later rule-based joint simulator and this model retrained when current-season walk-forward inputs exist.",
        "outputs": ["conditional and unconditional expected BPS", "expected bonus", "probability of zero, one, two, or three bonus points"],
    })
    latest = inputs.sort_values(["kickoff_time", "fixture_id"]).iloc[-1]
    example = inputs.loc[inputs["season"].eq(latest["season"]) & inputs["fixture_id"].eq(latest["fixture_id"])].copy()
    example_columns = list(dict.fromkeys(IDENTIFIER_COLUMNS + feature_columns))
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

    source_rows = load_training_data(str(settings["input_path"]))
    upstream_seasons = ["2023-24", "2024-25", "2025-26"]
    inputs = attach_upstream(source_rows.loc[source_rows["season"].isin(upstream_seasons)].copy(), settings)
    validation_seasons = [str(season) for season in settings["validation_seasons"]]
    splits = chronological_splits(inputs, validation_seasons)
    model_candidates = candidates(settings)
    print(f"Loaded {len(inputs):,} component-complete rows; tuning {len(model_candidates)} BPS candidates")
    tuning_results, fold_results, prediction_sets = tune_bps_models(splits, model_candidates, settings)
    best_number = int(tuning_results.iloc[0]["candidate"])
    best_candidate = model_candidates[best_number - 1]
    bps_predictions = prediction_sets[best_number]

    simulation_rows = []
    best_scale = None
    best_log_loss = float("inf")
    for scale in settings["residual_scales"]:
        simulated = simulate_bonus(
            bps_predictions,
            simulation_count=int(settings["tuning_simulations"]),
            random_state=int(settings["random_state"]),
            residual_scale=float(scale),
        )
        metrics = bonus_metrics(simulated)
        simulation_rows.append({"residual_scale": scale, **metrics})
        if metrics["bonus_log_loss"] < best_log_loss:
            best_log_loss = metrics["bonus_log_loss"]
            best_scale = float(scale)
    simulation_results = pd.DataFrame(simulation_rows).sort_values("bonus_log_loss").reset_index(drop=True)
    predictions = simulate_bonus(
        bps_predictions,
        simulation_count=int(settings["final_simulations"]),
        random_state=int(settings["random_state"]),
        residual_scale=best_scale,
    )
    metrics = {
        "walk_forward_model": {**bps_metrics(predictions), **bonus_metrics(predictions)},
        "by_season": metrics_by_season(predictions),
    }

    profile = str(best_candidate["feature_profile"])
    features = sum((list(part) for part in get_feature_columns(profile)), [])
    appeared = inputs["target_minutes"].gt(0)
    model = build_bps_model(profile, best_candidate["parameters"], random_state=int(settings["random_state"]))
    model.fit(inputs.loc[appeared, features], inputs.loc[appeared, "target_bps"])
    fitted = model.predict(inputs[features])
    residual_std = residual_std_by_position(inputs, fitted)
    model_dir = write_artifacts(
        source_rows, inputs, predictions, settings, best_candidate, model,
        residual_std, best_scale, tuning_results, fold_results,
        simulation_results, metrics,
    )
    summary = metrics["walk_forward_model"]
    print(f"Best BPS profile: {profile}; appeared RMSE={summary['appeared_bps_rmse']:.4f}")
    print(f"Best residual scale: {best_scale:.2f}; bonus log loss={summary['bonus_log_loss']:.4f}")
    print(f"Bonus MAE={summary['bonus_mae']:.4f}; any-bonus Brier={summary['any_bonus_brier']:.4f}")
    print(f"Wrote model and reports to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
