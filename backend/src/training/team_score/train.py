"""Tune and train the team score model from the processed Vaastav dataset."""

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
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from .model import (
    DEFAULT_MODEL_DIR,
    IDENTIFIER_COLUMNS,
    build_goal_pipeline,
    forecast_team_rows,
    get_feature_columns,
    load_team_training_data,
    predict_with_artifact,
    records_for_json,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "team_score.yaml"


def load_settings(path: str | Path) -> dict[str, object]:
    """Load and validate the team_score configuration section."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = config.get("team_score") if isinstance(config, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must contain a team_score mapping: {path}")
    return settings


def parameter_candidates(settings: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand feature profiles and estimator parameter values into candidates."""
    grid = settings.get("parameter_grid")
    profiles = settings.get("feature_profiles")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("team_score.parameter_grid must be a non-empty mapping")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("team_score.feature_profiles must be a non-empty list")

    parameter_names = list(grid)
    parameter_values = []
    for name in parameter_names:
        values = grid[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"parameter_grid.{name} must be a non-empty list")
        parameter_values.append(values)

    return [
        {
            "feature_profile": str(profile),
            "parameters": dict(zip(parameter_names, values)),
        }
        for profile in profiles
        for values in itertools.product(*parameter_values)
    ]


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
        validation_start = season_starts[season]
        training = rows.loc[rows["season_start"].lt(validation_start)].copy()
        validation = rows.loc[rows["season"].eq(season)].copy()
        if training.empty or validation.empty:
            raise ValueError(f"Validation season {season} does not have a usable split")
        splits.append((season, training, validation))
    return splits


def clean_sheet_calibration(team_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summarise clean-sheet calibration in ten probability bands."""
    calibration = team_forecasts[
        ["clean_sheet_probability", "actual_clean_sheet"]
    ].copy()
    calibration["probability_band"] = pd.cut(
        calibration["clean_sheet_probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    return (
        calibration.groupby("probability_band", observed=True)
        .agg(
            predictions=("actual_clean_sheet", "size"),
            mean_probability=("clean_sheet_probability", "mean"),
            observed_rate=("actual_clean_sheet", "mean"),
        )
        .reset_index()
        .assign(
            probability_band=lambda frame: frame["probability_band"].astype(str)
        )
    )


def calculate_metrics(
    team_forecasts: pd.DataFrame, fixture_forecasts: pd.DataFrame
) -> dict[str, float]:
    """Calculate the metrics used for tuning and model reporting."""
    goals_for_error = (
        team_forecasts["expected_goals_for"] - team_forecasts["actual_goals_for"]
    )
    goals_against_error = (
        team_forecasts["expected_goals_against"]
        - team_forecasts["actual_goals_against"]
    )
    clean_sheet_error = (
        team_forecasts["clean_sheet_probability"]
        - team_forecasts["actual_clean_sheet"]
    )
    calibration = clean_sheet_calibration(team_forecasts)
    calibration_error = np.average(
        np.abs(calibration["mean_probability"] - calibration["observed_rate"]),
        weights=calibration["predictions"],
    )

    probability_floor = 1e-15
    clean_sheet_probabilities = np.clip(
        team_forecasts["clean_sheet_probability"].to_numpy(),
        probability_floor,
        1 - probability_floor,
    )
    clean_sheet_targets = team_forecasts["actual_clean_sheet"].to_numpy()
    clean_sheet_log_loss = -np.mean(
        clean_sheet_targets * np.log(clean_sheet_probabilities)
        + (1 - clean_sheet_targets) * np.log(1 - clean_sheet_probabilities)
    )

    result_probabilities = fixture_forecasts[
        ["home_win_probability", "draw_probability", "away_win_probability"]
    ].to_numpy()
    result_targets = np.zeros_like(result_probabilities)
    home_goals = fixture_forecasts["actual_home_goals"].to_numpy()
    away_goals = fixture_forecasts["actual_away_goals"].to_numpy()
    result_targets[home_goals > away_goals, 0] = 1
    result_targets[home_goals == away_goals, 1] = 1
    result_targets[home_goals < away_goals, 2] = 1

    return {
        "team_rows": int(len(team_forecasts)),
        "fixtures": int(len(fixture_forecasts)),
        "goals_for_mae": float(
            mean_absolute_error(
                team_forecasts["actual_goals_for"],
                team_forecasts["expected_goals_for"],
            )
        ),
        "goals_for_rmse": float(np.sqrt(np.mean(np.square(goals_for_error)))),
        "goals_for_poisson_deviance": float(
            mean_poisson_deviance(
                team_forecasts["actual_goals_for"],
                team_forecasts["expected_goals_for"],
            )
        ),
        "goals_for_bias": float(goals_for_error.mean()),
        "mean_actual_goals_for": float(team_forecasts["actual_goals_for"].mean()),
        "mean_expected_goals_for": float(
            team_forecasts["expected_goals_for"].mean()
        ),
        "goals_against_mae": float(
            mean_absolute_error(
                team_forecasts["actual_goals_against"],
                team_forecasts["expected_goals_against"],
            )
        ),
        "goals_against_rmse": float(np.sqrt(np.mean(np.square(goals_against_error)))),
        "goals_against_poisson_deviance": float(
            mean_poisson_deviance(
                team_forecasts["actual_goals_against"],
                team_forecasts["expected_goals_against"],
            )
        ),
        "goals_against_bias": float(goals_against_error.mean()),
        "clean_sheet_brier": float(np.mean(np.square(clean_sheet_error))),
        "clean_sheet_log_loss": float(clean_sheet_log_loss),
        "clean_sheet_calibration_error": float(calibration_error),
        "mean_actual_clean_sheet": float(
            team_forecasts["actual_clean_sheet"].mean()
        ),
        "mean_predicted_clean_sheet": float(
            team_forecasts["clean_sheet_probability"].mean()
        ),
        "exact_score_log_loss": float(
            -np.log(
                np.clip(
                    fixture_forecasts["actual_score_probability"],
                    probability_floor,
                    1,
                )
            ).mean()
        ),
        "result_log_loss": float(
            -np.log(
                np.clip(
                    fixture_forecasts["actual_result_probability"],
                    probability_floor,
                    1,
                )
            ).mean()
        ),
        "result_brier": float(
            np.mean(np.sum(np.square(result_probabilities - result_targets), axis=1))
        ),
        "most_likely_score_accuracy": float(
            (
                fixture_forecasts["most_likely_home_goals"].eq(home_goals)
                & fixture_forecasts["most_likely_away_goals"].eq(away_goals)
            ).mean()
        ),
    }


def fit_candidate_folds(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    candidate: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Fit one candidate on each fold and return its walk-forward predictions."""
    profile = str(candidate["feature_profile"])
    parameters = candidate["parameters"]
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    random_state = int(settings["random_state"])

    predicted_inputs = []
    team_forecasts = []
    fixture_forecasts = []
    fold_metrics = []
    for season, training, validation in splits:
        pipeline = build_goal_pipeline(
            profile, parameters, random_state=random_state
        )
        pipeline.fit(training[feature_columns], training["target_team_goals"])
        raw_rates = pipeline.predict(validation[feature_columns])
        fold_inputs = validation.copy()
        fold_inputs["predicted_raw_goal_rate"] = raw_rates
        predicted_inputs.append(fold_inputs)
        teams, fixtures, _ = forecast_team_rows(
            validation,
            raw_rates,
            rho=0.0,
            max_goals=int(settings["max_goals"]),
            minimum_goal_rate=float(settings["minimum_goal_rate"]),
            maximum_goal_rate=float(settings["maximum_goal_rate"]),
        )
        team_forecasts.append(teams)
        fixture_forecasts.append(fixtures)
        metrics = calculate_metrics(teams, fixtures)
        fold_metrics.append({"validation_season": season, **metrics})

    return (
        pd.concat(predicted_inputs, ignore_index=True),
        pd.concat(team_forecasts, ignore_index=True),
        pd.concat(fixture_forecasts, ignore_index=True),
        fold_metrics,
    )


def tune_candidates(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    candidates: Sequence[dict[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all model candidates and return aggregate and fold results."""
    results = []
    fold_results = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        _, teams, fixtures, folds = fit_candidate_folds(splits, candidate, settings)
        metrics = calculate_metrics(teams, fixtures)
        parameters = candidate["parameters"]
        result = {
            "candidate": index,
            "feature_profile": candidate["feature_profile"],
            **{f"parameter_{key}": value for key, value in parameters.items()},
            **metrics,
        }
        results.append(result)
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
            f"Candidate {index:>2}/{total}: {candidate['feature_profile']} "
            f"score log loss={metrics['exact_score_log_loss']:.4f}, "
            f"clean-sheet Brier={metrics['clean_sheet_brier']:.4f}"
        )

    results_frame = pd.DataFrame(results).sort_values(
        ["exact_score_log_loss", "clean_sheet_brier", "goals_for_mae"]
    )
    results_frame.insert(0, "rank", range(1, len(results_frame) + 1))
    return results_frame, pd.DataFrame(fold_results)


def tune_dixon_coles_rho(
    predicted_inputs: pd.DataFrame, settings: Mapping[str, object]
) -> pd.DataFrame:
    """Tune the low-score dependency correction on walk-forward predictions."""
    rho_settings = settings["dixon_coles_rho"]
    minimum = float(rho_settings["minimum"])
    maximum = float(rho_settings["maximum"])
    step = float(rho_settings["step"])
    rho_values = np.arange(minimum, maximum + step / 2, step)

    results = []
    for rho in rho_values:
        teams, fixtures, _ = forecast_team_rows(
            predicted_inputs,
            predicted_inputs["predicted_raw_goal_rate"],
            rho=float(rho),
            max_goals=int(settings["max_goals"]),
            minimum_goal_rate=float(settings["minimum_goal_rate"]),
            maximum_goal_rate=float(settings["maximum_goal_rate"]),
        )
        metrics = calculate_metrics(teams, fixtures)
        results.append(
            {
                "rho": round(float(rho), 10),
                "exact_score_log_loss": metrics["exact_score_log_loss"],
                "clean_sheet_brier": metrics["clean_sheet_brier"],
                "result_log_loss": metrics["result_log_loss"],
            }
        )
    return pd.DataFrame(results).sort_values(
        ["exact_score_log_loss", "clean_sheet_brier"]
    )


def baseline_walk_forward(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict historical home/away scoring means as a simple benchmark."""
    teams = []
    fixtures = []
    for _, training, validation in splits:
        venue_means = training.groupby("is_home")["target_team_goals"].mean()
        overall_mean = float(training["target_team_goals"].mean())
        raw_rates = validation["is_home"].map(venue_means).fillna(overall_mean)
        fold_teams, fold_fixtures, _ = forecast_team_rows(
            validation,
            raw_rates,
            rho=0.0,
            max_goals=int(settings["max_goals"]),
            minimum_goal_rate=float(settings["minimum_goal_rate"]),
            maximum_goal_rate=float(settings["maximum_goal_rate"]),
        )
        teams.append(fold_teams)
        fixtures.append(fold_fixtures)
    return pd.concat(teams, ignore_index=True), pd.concat(fixtures, ignore_index=True)


def metrics_by_season(
    team_forecasts: pd.DataFrame, fixture_forecasts: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Calculate the complete metric set for each validation season."""
    return {
        str(season): calculate_metrics(
            team_forecasts.loc[team_forecasts["season"].eq(season)],
            fixture_forecasts.loc[fixture_forecasts["season"].eq(season)],
        )
        for season in team_forecasts["season"].drop_duplicates()
    }


def feature_coverage(rows: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Report non-null feature coverage by season."""
    return pd.DataFrame(
        [
            {
                "season": season,
                "feature": feature,
                "available_rows": int(season_rows[feature].notna().sum()),
                "team_rows": int(len(season_rows)),
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
    best_candidate: Mapping[str, object],
    best_parameters: Mapping[str, object],
    pipeline: object,
    rho: float,
    tuning_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    rho_results: pd.DataFrame,
    team_forecasts: pd.DataFrame,
    fixture_forecasts: pd.DataFrame,
    model_metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
) -> Path:
    """Write the trained model and its reproducibility/reporting artifacts."""
    model_dir = Path(str(settings.get("model_dir", DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    profile = str(best_candidate["feature_profile"])
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_version": 1,
        "model_type": "team_score",
        "trained_at": trained_at,
        "pipeline": pipeline,
        "feature_profile": profile,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "estimator_parameters": dict(best_parameters),
        "dixon_coles_rho": rho,
        "max_goals": int(settings["max_goals"]),
        "minimum_goal_rate": float(settings["minimum_goal_rate"]),
        "maximum_goal_rate": float(settings["maximum_goal_rate"]),
    }
    joblib.dump(artifact, model_dir / "model.joblib")

    coverage = feature_coverage(rows, feature_columns)
    coverage.to_csv(model_dir / "feature_coverage.csv", index=False)
    tuning_results.to_csv(model_dir / "tuning_results.csv", index=False)
    fold_results.to_csv(model_dir / "tuning_fold_results.csv", index=False)
    rho_results.to_csv(model_dir / "rho_tuning.csv", index=False)
    team_forecasts.to_csv(
        model_dir / "walk_forward_team_predictions.csv", index=False
    )
    fixture_forecasts.to_csv(
        model_dir / "walk_forward_fixture_predictions.csv", index=False
    )
    clean_sheet_calibration(team_forecasts).to_csv(
        model_dir / "clean_sheet_calibration.csv", index=False
    )

    best_parameter_payload = {
        "feature_profile": profile,
        "estimator": "HistGradientBoostingRegressor",
        "loss": "poisson",
        "estimator_parameters": dict(best_parameters),
        "dixon_coles_rho": rho,
        "selection_metric": "walk-forward exact-score log loss",
    }
    write_json(model_dir / "best_parameters.json", best_parameter_payload)
    write_json(
        model_dir / "metrics.json",
        {
            "walk_forward_model": model_metrics,
            "walk_forward_home_away_mean_baseline": baseline_metrics,
        },
    )

    xg_features = [feature for feature in feature_columns if "expected_goals" in feature]
    xg_coverage = (
        coverage.loc[coverage["feature"].isin(xg_features)]
        .groupby("season")["coverage"]
        .mean()
        .to_dict()
        if xg_features
        else {}
    )
    metadata = {
        "model_type": "team_score",
        "trained_at": trained_at,
        "source_dataset": str(settings["input_path"]),
        "source": "Vaastav Fantasy Premier League repository",
        "training_team_rows": int(len(rows)),
        "training_fixtures": int(len(rows) / 2),
        "training_seasons": rows["season"].drop_duplicates().tolist(),
        "training_cutoff": str(rows["kickoff_time"].max()),
        "validation_seasons": list(settings["validation_seasons"]),
        "validation_method": "expanding-window season walk-forward",
        "target": "team goals in the fixture",
        "outputs": [
            "expected goals for each team",
            "expected goals against each team",
            "home/draw/away probabilities",
            "clean-sheet probabilities",
            "scoreline probability distribution",
        ],
        "feature_profile": profile,
        "feature_count": len(feature_columns),
        "xg_feature_coverage_by_season": xg_coverage,
        "missing_values": (
            "Numeric missing values are median-imputed within each training fold, "
            "with missingness indicators. This covers early-season history and "
            "historical xG gaps without using future values."
        ),
        "score_model": (
            "Home and away Poisson goal rates are paired and adjusted for low-score "
            "dependence with a tuned Dixon-Coles rho."
        ),
    }
    write_json(model_dir / "metadata.json", metadata)
    write_json(
        model_dir / "feature_schema.json",
        {
            "required_identifiers": IDENTIFIER_COLUMNS,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
    )

    latest_fixture = (
        rows.sort_values(["kickoff_time", "fixture_id"])
        .groupby(["season", "fixture_id"], sort=False)
        .tail(2)
        .tail(2)
        .copy()
    )
    example_columns = list(dict.fromkeys(IDENTIFIER_COLUMNS + feature_columns))
    example_input = {"fixture_rows": records_for_json(latest_fixture[example_columns])}
    write_json(model_dir / "example_input.json", example_input)
    example_teams, example_fixtures, example_scorelines = predict_with_artifact(
        artifact, latest_fixture, include_scorelines=True
    )
    write_json(
        model_dir / "example_output.json",
        {
            "teams": records_for_json(example_teams),
            "fixtures": records_for_json(example_fixtures),
            "scorelines": records_for_json(example_scorelines),
        },
    )
    example_scorelines.to_csv(
        model_dir / "example_score_probabilities.csv", index=False
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

    rows = load_team_training_data(str(settings["input_path"]))
    validation_seasons = [str(season) for season in settings["validation_seasons"]]
    splits = chronological_splits(rows, validation_seasons)
    candidates = parameter_candidates(settings)
    print(
        f"Loaded {len(rows):,} team rows ({len(rows) // 2:,} fixtures); "
        f"tuning {len(candidates)} candidates across {len(splits)} seasons"
    )

    tuning_results, fold_results = tune_candidates(splits, candidates, settings)
    best_row = tuning_results.iloc[0]
    best_candidate_number = int(best_row["candidate"])
    best_candidate = candidates[best_candidate_number - 1]
    predicted_inputs, _, _, _ = fit_candidate_folds(
        splits, best_candidate, settings
    )
    rho_results = tune_dixon_coles_rho(predicted_inputs, settings)
    best_rho = float(rho_results.iloc[0]["rho"])
    team_forecasts, fixture_forecasts, _ = forecast_team_rows(
        predicted_inputs,
        predicted_inputs["predicted_raw_goal_rate"],
        rho=best_rho,
        max_goals=int(settings["max_goals"]),
        minimum_goal_rate=float(settings["minimum_goal_rate"]),
        maximum_goal_rate=float(settings["maximum_goal_rate"]),
    )
    model_metrics = {
        "all_validation_seasons": calculate_metrics(
            team_forecasts, fixture_forecasts
        ),
        "by_season": metrics_by_season(team_forecasts, fixture_forecasts),
    }
    baseline_teams, baseline_fixtures = baseline_walk_forward(splits, settings)
    baseline_metrics = {
        "all_validation_seasons": calculate_metrics(
            baseline_teams, baseline_fixtures
        ),
        "by_season": metrics_by_season(baseline_teams, baseline_fixtures),
    }

    profile = str(best_candidate["feature_profile"])
    best_parameters = best_candidate["parameters"]
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    final_pipeline = build_goal_pipeline(
        profile, best_parameters, random_state=int(settings["random_state"])
    )
    final_pipeline.fit(rows[feature_columns], rows["target_team_goals"])
    model_dir = write_artifacts(
        rows,
        settings,
        best_candidate,
        best_parameters,
        final_pipeline,
        best_rho,
        tuning_results,
        fold_results,
        rho_results,
        team_forecasts,
        fixture_forecasts,
        model_metrics,
        baseline_metrics,
    )

    metrics = model_metrics["all_validation_seasons"]
    print(f"Best feature profile: {profile}")
    print(f"Best estimator parameters: {json.dumps(best_parameters, sort_keys=True)}")
    print(f"Best Dixon-Coles rho: {best_rho:.2f}")
    print(
        f"Walk-forward: goals MAE={metrics['goals_for_mae']:.4f}, "
        f"clean-sheet Brier={metrics['clean_sheet_brier']:.4f}, "
        f"score log loss={metrics['exact_score_log_loss']:.4f}"
    )
    print(f"Wrote model and reports to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
