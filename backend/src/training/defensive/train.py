"""Tune and train defensive, clean-sheet, and goalkeeper components."""

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
    IDENTIFIER_COLUMNS,
    PLAYER_KEYS,
    UPSTREAM_COLUMNS,
    build_count_model,
    build_predictions,
    get_feature_columns,
    load_training_data,
    position_rate_priors,
    predict_with_artifact,
    records_for_json,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "defensive.yaml"


def load_settings(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = config.get("defensive") if isinstance(config, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must contain a defensive mapping: {path}")
    return settings


def candidates(settings: Mapping[str, object]) -> list[dict[str, object]]:
    grid = settings["parameter_grid"]
    profiles = settings["feature_profiles"]
    names = list(grid)
    return [
        {"feature_profile": str(profile), "parameters": dict(zip(names, values))}
        for profile in profiles
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


def attach_upstream(rows: pd.DataFrame, settings: Mapping[str, object]) -> pd.DataFrame:
    minutes = pd.read_csv(
        str(settings["minutes_predictions_path"]),
        usecols=PLAYER_KEYS + ["expected_minutes", "sixty_plus_probability"],
    )
    team = pd.read_csv(
        str(settings["team_predictions_path"]),
        usecols=["season", "fixture_id", "team_key", "expected_goals_against", "clean_sheet_probability"],
    ).rename(columns={
        "expected_goals_against": "team_expected_goals_against",
        "clean_sheet_probability": "team_clean_sheet_probability",
    })
    joined = rows.merge(minutes, on=PLAYER_KEYS, how="left", validate="one_to_one").merge(
        team, on=["season", "fixture_id", "team_key"], how="left", validate="many_to_one"
    )
    missing = joined[UPSTREAM_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Upstream predictions are missing for {int(missing.sum())} defensive rows")
    return joined


def _rate_rows(rows: pd.DataFrame, target: str, *, goalkeeper: bool) -> pd.DataFrame:
    mask = rows["target_minutes"].gt(0)
    mask &= rows["position"].eq("GK") if goalkeeper else rows["position"].ne("GK")
    if target == "target_defensive_contribution":
        mask &= rows["target_defensive_contribution_available"].eq(1)
    output = rows.loc[mask].copy()
    output["rate_target"] = 90 * output[target] / output["target_minutes"]
    return output


def _count_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    expected = np.clip(np.asarray(expected, dtype=float), 1e-8, None)
    actual = np.asarray(actual, dtype=float)
    error = expected - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "poisson_deviance": float(mean_poisson_deviance(actual, expected)),
        "bias": float(error.mean()),
        "mean_actual": float(actual.mean()),
        "mean_expected": float(expected.mean()),
    }


def tune_save_models(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    upstream: pd.DataFrame,
    model_candidates: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    summaries, folds, prediction_sets = [], [], {}
    for number, candidate in enumerate(model_candidates, 1):
        candidate_predictions = []
        for season, training, validation in splits:
            model = build_count_model(
                str(candidate["feature_profile"]), candidate["parameters"],
                random_state=int(settings["random_state"]),
            )
            fit_rows = _rate_rows(training, "target_saves", goalkeeper=True)
            features = sum((list(part) for part in get_feature_columns(str(candidate["feature_profile"]))), [])
            model.fit(fit_rows[features], fit_rows["rate_target"])
            validation_upstream = upstream.loc[upstream["season"].eq(season)].copy()
            validation_upstream["raw_rate"] = model.predict(validation_upstream[features])
            candidate_predictions.append(validation_upstream)
            keeper = validation_upstream["position"].eq("GK")
            metric = _count_metrics(
                validation_upstream.loc[keeper, "target_saves"],
                np.clip(validation_upstream.loc[keeper, "raw_rate"], 0, None)
                * validation_upstream.loc[keeper, "expected_minutes"] / 90,
            )
            folds.append({"candidate": number, "validation_season": season, **metric})
        predictions = pd.concat(candidate_predictions, ignore_index=True)
        prediction_sets[number] = predictions
        keeper = predictions["position"].eq("GK")
        metric = _count_metrics(
            predictions.loc[keeper, "target_saves"],
            np.clip(predictions.loc[keeper, "raw_rate"], 0, None) * predictions.loc[keeper, "expected_minutes"] / 90,
        )
        summaries.append({"candidate": number, **candidate, **metric})
        print(f"Save candidate {number}/{len(model_candidates)}: deviance={metric['poisson_deviance']:.4f}")
    return (
        pd.DataFrame(summaries).sort_values(["poisson_deviance", "mae"]).reset_index(drop=True),
        pd.DataFrame(folds),
        prediction_sets,
    )


def defcon_splits(rows: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    available = rows.loc[rows["target_defensive_contribution_available"].eq(1) & rows["season"].eq("2025-26")]
    return [
        (
            f"2025-26 GW{start}-{end}",
            available.loc[available["gameweek"].lt(start)].copy(),
            available.loc[available["gameweek"].between(start, end)].copy(),
        )
        for start, end in ((11, 19), (20, 28), (29, 38))
    ]


def tune_defcon_models(
    rows: pd.DataFrame,
    upstream: pd.DataFrame,
    model_candidates: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    summaries, folds, prediction_sets = [], [], {}
    for number, candidate in enumerate(model_candidates, 1):
        candidate_predictions = []
        for label, training, validation in defcon_splits(rows):
            model = build_count_model(
                str(candidate["feature_profile"]), candidate["parameters"],
                random_state=int(settings["random_state"]),
            )
            fit_rows = _rate_rows(training, "target_defensive_contribution", goalkeeper=False)
            features = sum((list(part) for part in get_feature_columns(str(candidate["feature_profile"]))), [])
            model.fit(fit_rows[features], fit_rows["rate_target"])
            predicted = validation.copy()
            predicted["raw_rate"] = model.predict(predicted[features])
            predicted = predicted.merge(
                upstream[PLAYER_KEYS + ["expected_minutes"]],
                on=PLAYER_KEYS,
                how="left",
                validate="one_to_one",
            )
            candidate_predictions.append(predicted)
            scored = predicted["position"].ne("GK") & predicted["target_minutes"].gt(0)
            metric = _count_metrics(
                predicted.loc[scored, "target_defensive_contribution"],
                np.clip(predicted.loc[scored, "raw_rate"], 0, None) * predicted.loc[scored, "expected_minutes"] / 90,
            )
            folds.append({"candidate": number, "validation_block": label, **metric})
        predictions = pd.concat(candidate_predictions, ignore_index=True)
        prediction_sets[number] = predictions
        scored = predictions["position"].ne("GK") & predictions["target_minutes"].gt(0)
        metric = _count_metrics(
            predictions.loc[scored, "target_defensive_contribution"],
            np.clip(predictions.loc[scored, "raw_rate"], 0, None) * predictions.loc[scored, "expected_minutes"] / 90,
        )
        summaries.append({"candidate": number, **candidate, **metric})
        print(f"Defcon candidate {number}/{len(model_candidates)}: deviance={metric['poisson_deviance']:.4f}")
    return (
        pd.DataFrame(summaries).sort_values(["poisson_deviance", "mae"]).reset_index(drop=True),
        pd.DataFrame(folds),
        prediction_sets,
    )


def _available_priors(rows: pd.DataFrame, target: str) -> dict[str, float]:
    if target == "target_defensive_contribution":
        rows = rows.loc[rows["target_defensive_contribution_available"].eq(1)]
    return position_rate_priors(rows, target)


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    keeper = predictions["position"].eq("GK")
    defcon = predictions["defensive_contribution_available"].eq(1) & predictions["position"].ne("GK")
    if "defcon_walk_forward_scored" in predictions:
        defcon &= predictions["defcon_walk_forward_scored"].eq(1)
    clean_sheet_error = predictions["player_clean_sheet_probability"] - predictions["actual_clean_sheet"]
    thresholds = predictions.loc[defcon, "defensive_contribution_threshold"]
    defcon_actual = predictions.loc[defcon, "actual_defensive_contributions"].to_numpy() >= thresholds.to_numpy()
    defcon_probability = predictions.loc[defcon, "defensive_contribution_points_probability"].to_numpy()
    metrics = {
        **{f"saves_{key}": value for key, value in _count_metrics(predictions.loc[keeper, "actual_saves"], predictions.loc[keeper, "expected_saves"]).items()},
        **{f"defcon_{key}": value for key, value in _count_metrics(predictions.loc[defcon, "actual_defensive_contributions"], predictions.loc[defcon, "expected_defensive_contributions"]).items()},
        "clean_sheet_brier": float(np.mean(np.square(clean_sheet_error))),
        "clean_sheet_actual_rate": float(predictions["actual_clean_sheet"].mean()),
        "clean_sheet_predicted_rate": float(predictions["player_clean_sheet_probability"].mean()),
        "defcon_threshold_brier": float(np.mean(np.square(defcon_probability - defcon_actual))),
        "goals_conceded_mae": float(np.mean(np.abs(predictions["expected_goals_conceded"] - predictions["actual_goals_conceded"]))),
        "penalty_saves_mae": float(np.mean(np.abs(predictions.loc[keeper, "expected_penalty_saves"] - predictions.loc[keeper, "actual_penalty_saves"]))),
        "penalty_saves_poisson_deviance": _count_metrics(
            predictions.loc[keeper, "actual_penalty_saves"],
            predictions.loc[keeper, "expected_penalty_saves"],
        )["poisson_deviance"],
    }
    return metrics


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, (np.integer, np.floating)) else str(value)) + "\n", encoding="utf-8")


def write_artifacts(
    rows: pd.DataFrame,
    prediction_inputs: pd.DataFrame,
    predictions: pd.DataFrame,
    settings: Mapping[str, object],
    save_candidate: Mapping[str, object],
    defcon_candidate: Mapping[str, object],
    save_model: object,
    defcon_model: object,
    priors: Mapping[str, Mapping[str, float]],
    shrinkage: Mapping[str, float],
    save_results: pd.DataFrame,
    save_folds: pd.DataFrame,
    defcon_results: pd.DataFrame,
    defcon_folds: pd.DataFrame,
    shrinkage_results: pd.DataFrame,
    metrics: Mapping[str, float],
) -> Path:
    model_dir = Path(str(settings.get("model_dir", DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    save_profile = str(save_candidate["feature_profile"])
    defcon_profile = str(defcon_candidate["feature_profile"])
    feature_columns = sorted(set(sum((list(part) for part in get_feature_columns(save_profile)), []) + sum((list(part) for part in get_feature_columns(defcon_profile)), [])))
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_version": 1,
        "model_type": "defensive",
        "trained_at": trained_at,
        "save_model": save_model,
        "defcon_model": defcon_model,
        "feature_columns": feature_columns,
        "save_feature_profile": save_profile,
        "defcon_feature_profile": defcon_profile,
        "save_priors": dict(priors["saves"]),
        "defcon_priors": dict(priors["defcon"]),
        "penalty_save_priors": dict(priors["penalty_saves"]),
        "save_shrinkage_minutes": shrinkage["saves"],
        "defcon_shrinkage_minutes": shrinkage["defcon"],
        "penalty_shrinkage_minutes": shrinkage["penalty_saves"],
    }
    joblib.dump(artifact, model_dir / "model.joblib")
    save_results.to_csv(model_dir / "save_model_tuning.csv", index=False)
    save_folds.to_csv(model_dir / "save_model_tuning_folds.csv", index=False)
    defcon_results.to_csv(model_dir / "defcon_model_tuning.csv", index=False)
    defcon_folds.to_csv(model_dir / "defcon_model_tuning_folds.csv", index=False)
    shrinkage_results.to_csv(model_dir / "shrinkage_tuning.csv", index=False)
    predictions.to_csv(model_dir / "walk_forward_predictions.csv", index=False)
    pd.DataFrame([
        {"season": season, "target": target, "rows": len(group), "positive_rows": int(group[target].fillna(0).gt(0).sum()), "available_rows": int(group[target].notna().sum())}
        for season, group in rows.groupby("season")
        for target in ("target_saves", "target_penalties_saved", "target_defensive_contribution", "target_clean_sheets", "target_goals_conceded")
    ]).to_csv(model_dir / "target_coverage.csv", index=False)
    write_json(model_dir / "best_parameters.json", {
        "save_model": save_candidate,
        "defensive_contribution_model": defcon_candidate,
        "shrinkage_minutes": shrinkage,
        "selection_metrics": {"models": "walk-forward Poisson deviance", "shrinkage": "combined saves, defensive-contribution, and penalty-save Poisson deviance"},
    })
    write_json(model_dir / "metrics.json", {"walk_forward_model": metrics})
    write_json(model_dir / "feature_schema.json", {
        "required_identifiers": IDENTIFIER_COLUMNS,
        "required_upstream_predictions": UPSTREAM_COLUMNS,
        "feature_columns": feature_columns,
    })
    write_json(model_dir / "metadata.json", {
        "model_type": "defensive",
        "trained_at": trained_at,
        "source_dataset": str(settings["input_path"]),
        "source": "Vaastav Fantasy Premier League repository",
        "validation_method": "season walk-forward for saves; expanding gameweek blocks for defensive contributions",
        "defensive_contribution_data_gap": "Vaastav defensive-contribution targets start in 2025/26. Earlier rows use position priors and are excluded from defcon scoring.",
        "distribution_method": "Poisson count distributions produce save-point expectations, defensive-contribution threshold probabilities, and goals-conceded deductions.",
        "clean_sheet_method": "Team clean-sheet probability is multiplied by the minutes model's probability of reaching 60 minutes.",
        "outputs": ["player clean-sheet probability and points", "goal-conceded exposure and expected deduction", "goalkeeper saves and save points", "penalty saves", "defensive contributions and threshold points"],
    })
    latest = prediction_inputs.sort_values(["kickoff_time", "fixture_id"]).iloc[-1]
    example = prediction_inputs.loc[prediction_inputs["season"].eq(latest["season"]) & prediction_inputs["fixture_id"].eq(latest["fixture_id"])].copy()
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
    validation = rows.loc[rows["season"].isin(validation_seasons)].copy()
    upstream = attach_upstream(validation, settings)
    model_candidates = candidates(settings)
    print(f"Loaded {len(rows):,} rows; tuning {len(model_candidates)} candidates")
    save_results, save_folds, save_sets = tune_save_models(
        chronological_splits(rows, validation_seasons), upstream, model_candidates, settings
    )
    defcon_results, defcon_folds, defcon_sets = tune_defcon_models(rows, upstream, model_candidates, settings)
    best_save_number = int(save_results.iloc[0]["candidate"])
    best_defcon_number = int(defcon_results.iloc[0]["candidate"])
    best_save = model_candidates[best_save_number - 1]
    best_defcon = model_candidates[best_defcon_number - 1]

    priors = {
        "saves": _available_priors(rows, "target_saves"),
        "defcon": _available_priors(rows, "target_defensive_contribution"),
        "penalty_saves": _available_priors(rows, "target_penalties_saved"),
    }
    evaluation_priors = {
        "saves": _available_priors(rows.loc[rows["season"].eq("2022-23")], "target_saves"),
        "defcon": _available_priors(
            rows.loc[rows["season"].eq("2025-26") & rows["gameweek"].le(10)],
            "target_defensive_contribution",
        ),
        "penalty_saves": _available_priors(
            rows.loc[rows["season"].eq("2022-23")], "target_penalties_saved"
        ),
    }
    save_inputs = save_sets[best_save_number]
    defcon_rates = save_inputs["position"].map(evaluation_priors["defcon"]).fillna(evaluation_priors["defcon"]["__default__"]).to_numpy(float)
    defcon_oof = defcon_sets[best_defcon_number][PLAYER_KEYS + ["raw_rate"]]
    rate_lookup = defcon_oof.set_index(PLAYER_KEYS)["raw_rate"]
    keys = pd.MultiIndex.from_frame(save_inputs[PLAYER_KEYS])
    available_rates = rate_lookup.reindex(keys).to_numpy(float)
    save_inputs["defcon_walk_forward_scored"] = ~np.isnan(available_rates)
    defcon_rates = np.where(np.isnan(available_rates), defcon_rates, available_rates)

    tuning_rows = []
    best_score = float("inf")
    best_shrinkage = None
    for save_minutes, defcon_minutes, penalty_minutes in itertools.product(
        settings["shrinkage_minutes"], settings["shrinkage_minutes"], settings["penalty_shrinkage_minutes"]
    ):
        candidate_predictions = build_predictions(
            save_inputs, save_inputs["raw_rate"], defcon_rates,
            evaluation_priors["saves"], evaluation_priors["defcon"], evaluation_priors["penalty_saves"],
            save_shrinkage_minutes=float(save_minutes),
            defcon_shrinkage_minutes=float(defcon_minutes),
            penalty_shrinkage_minutes=float(penalty_minutes),
        )
        metric = calculate_metrics(candidate_predictions)
        score = (
            metric["saves_poisson_deviance"]
            + metric["defcon_poisson_deviance"]
            + metric["penalty_saves_poisson_deviance"]
        )
        tuning_rows.append({
            "save_shrinkage_minutes": save_minutes,
            "defcon_shrinkage_minutes": defcon_minutes,
            "penalty_shrinkage_minutes": penalty_minutes,
            "selection_score": score,
            **metric,
        })
        if score < best_score:
            best_score = score
            best_shrinkage = {"saves": float(save_minutes), "defcon": float(defcon_minutes), "penalty_saves": float(penalty_minutes)}
    shrinkage_results = pd.DataFrame(tuning_rows).sort_values("selection_score").reset_index(drop=True)
    predictions = build_predictions(
        save_inputs, save_inputs["raw_rate"], defcon_rates,
        evaluation_priors["saves"], evaluation_priors["defcon"], evaluation_priors["penalty_saves"],
        save_shrinkage_minutes=best_shrinkage["saves"],
        defcon_shrinkage_minutes=best_shrinkage["defcon"],
        penalty_shrinkage_minutes=best_shrinkage["penalty_saves"],
    )
    metrics = calculate_metrics(predictions)

    save_features = sum((list(part) for part in get_feature_columns(str(best_save["feature_profile"]))), [])
    defcon_features = sum((list(part) for part in get_feature_columns(str(best_defcon["feature_profile"]))), [])
    save_model = build_count_model(str(best_save["feature_profile"]), best_save["parameters"], random_state=int(settings["random_state"]))
    save_fit = _rate_rows(rows, "target_saves", goalkeeper=True)
    save_model.fit(save_fit[save_features], save_fit["rate_target"])
    defcon_model = build_count_model(str(best_defcon["feature_profile"]), best_defcon["parameters"], random_state=int(settings["random_state"]))
    defcon_fit = _rate_rows(rows, "target_defensive_contribution", goalkeeper=False)
    defcon_model.fit(defcon_fit[defcon_features], defcon_fit["rate_target"])
    model_dir = write_artifacts(
        rows, save_inputs, predictions, settings, best_save, best_defcon,
        save_model, defcon_model, priors, best_shrinkage, save_results, save_folds,
        defcon_results, defcon_folds, shrinkage_results, metrics,
    )
    print(f"Best save profile: {best_save['feature_profile']}; deviance={metrics['saves_poisson_deviance']:.4f}")
    print(f"Best defcon profile: {best_defcon['feature_profile']}; deviance={metrics['defcon_poisson_deviance']:.4f}")
    print(f"Clean-sheet Brier={metrics['clean_sheet_brier']:.4f}; defcon threshold Brier={metrics['defcon_threshold_brier']:.4f}")
    print(f"Wrote model and reports to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
