"""Tune and train player attacking rates with team-total reconciliation."""

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
    TEAM_ROW_KEYS,
    UPSTREAM_COLUMNS,
    build_attacking_predictions,
    build_rate_model,
    fpl_assists_per_team_goal,
    get_feature_columns,
    load_attacking_training_data,
    position_rate_priors,
    predict_with_artifact,
    records_for_json,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "attacking.yaml"


def load_settings(path: str | Path) -> dict[str, object]:
    """Load and validate the attacking configuration section."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = config.get("attacking") if isinstance(config, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must contain an attacking mapping: {path}")
    return settings


def parameter_candidates(settings: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand feature profiles and estimator parameter values."""
    grid = settings.get("parameter_grid")
    profiles = settings.get("feature_profiles")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("attacking.parameter_grid must be a non-empty mapping")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("attacking.feature_profiles must be a non-empty list")
    names = list(grid)
    values = []
    for name in names:
        choices = grid[name]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"parameter_grid.{name} must be a non-empty list")
        values.append(choices)
    return [
        {
            "feature_profile": str(profile),
            "parameters": dict(zip(names, combination)),
        }
        for profile in profiles
        for combination in itertools.product(*values)
    ]


def chronological_splits(
    rows: pd.DataFrame, validation_seasons: Sequence[str]
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
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


def attach_upstream_predictions(
    rows: pd.DataFrame, settings: Mapping[str, object]
) -> pd.DataFrame:
    """Join upstream forecasts to their exact player and team keys."""
    team = pd.read_csv(
        str(settings["team_predictions_path"]),
        usecols=["season", "fixture_id", "team_key", "expected_goals_for"],
    ).rename(columns={"expected_goals_for": "team_expected_goals"})
    minutes = pd.read_csv(
        str(settings["minutes_predictions_path"]),
        usecols=["season", "fixture_id", "player_key", "expected_minutes"],
    )
    joined = rows.merge(
        team,
        on=["season", "fixture_id", "team_key"],
        how="left",
        validate="many_to_one",
    ).merge(
        minutes,
        on=["season", "fixture_id", "player_key"],
        how="left",
        validate="one_to_one",
    )
    missing = joined[UPSTREAM_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"Upstream walk-forward predictions are missing for {int(missing.sum())} rows"
        )
    return joined


def _binary_metrics(actual: np.ndarray, expected_count: np.ndarray) -> dict[str, float]:
    actual_event = actual > 0
    probability = np.clip(1 - np.exp(-expected_count), 1e-15, 1 - 1e-15)
    return {
        "brier": float(np.mean(np.square(probability - actual_event))),
        "log_loss": float(
            -np.mean(
                actual_event * np.log(probability)
                + (~actual_event) * np.log(1 - probability)
            )
        ),
        "actual_rate": float(actual_event.mean()),
        "predicted_rate": float(probability.mean()),
    }


def _count_metrics(actual: pd.Series, expected: pd.Series) -> dict[str, float]:
    error = expected.to_numpy() - actual.to_numpy()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "poisson_deviance": float(mean_poisson_deviance(actual, expected)),
        "bias": float(error.mean()),
        "mean_actual": float(actual.mean()),
        "mean_expected": float(expected.mean()),
    }


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Evaluate statistical rates, FPL events, and team reconciliation."""
    goals = _count_metrics(predictions["actual_goals"], predictions["expected_goals"])
    assists = _count_metrics(
        predictions["actual_assists"], predictions["expected_fpl_assists"]
    )
    goal_events = _binary_metrics(
        predictions["actual_goals"].to_numpy(),
        predictions["expected_goals"].to_numpy(),
    )
    assist_events = _binary_metrics(
        predictions["actual_assists"].to_numpy(),
        predictions["expected_fpl_assists"].to_numpy(),
    )
    expected_rows = predictions["expected_data_available"].astype(bool)
    xg = _count_metrics(
        predictions.loc[expected_rows, "actual_xg"],
        predictions.loc[expected_rows, "pre_reconciliation_xg"],
    )
    xa = _count_metrics(
        predictions.loc[expected_rows, "actual_xa"],
        predictions.loc[expected_rows, "expected_statistical_xa"],
    )
    team_totals = predictions.groupby(TEAM_ROW_KEYS, as_index=False).agg(
        team_expected_goals=("team_expected_goals", "first"),
        allocated_goals=("expected_goals", "sum"),
        allocated_assists=("expected_fpl_assists", "sum"),
        actual_goals=("actual_goals", "sum"),
        actual_assists=("actual_assists", "sum"),
    )
    reconciliation_error = (
        team_totals["allocated_goals"] - team_totals["team_expected_goals"]
    ).abs()
    team_goal_metrics = _count_metrics(
        team_totals["actual_goals"], team_totals["allocated_goals"]
    )
    team_assist_metrics = _count_metrics(
        team_totals["actual_assists"], team_totals["allocated_assists"]
    )
    return {
        "player_fixture_rows": int(len(predictions)),
        "team_fixtures": int(len(team_totals)),
        "attacking_poisson_deviance": float(
            goals["poisson_deviance"] + assists["poisson_deviance"]
        ),
        **{f"goals_{key}": value for key, value in goals.items()},
        **{f"goal_event_{key}": value for key, value in goal_events.items()},
        **{f"assists_{key}": value for key, value in assists.items()},
        **{f"assist_event_{key}": value for key, value in assist_events.items()},
        **{f"xg_{key}": value for key, value in xg.items()},
        **{f"xa_{key}": value for key, value in xa.items()},
        **{f"team_goals_{key}": value for key, value in team_goal_metrics.items()},
        **{f"team_assists_{key}": value for key, value in team_assist_metrics.items()},
        "maximum_team_goal_reconciliation_error": float(
            reconciliation_error.max()
        ),
        "mean_team_goal_reconciliation_error": float(reconciliation_error.mean()),
    }


def _rate_training_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[
        rows["expected_data_available"] & rows["target_minutes"].gt(0)
    ].copy()


def _fit_rate_models(
    training: pd.DataFrame,
    profile: str,
    parameters: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[object, object]:
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    rate_training = _rate_training_rows(training)
    sample_weight = rate_training["target_minutes"] / 90

    goal_model = build_rate_model(
        profile, parameters, random_state=int(settings["random_state"])
    )
    goal_model.fit(
        rate_training[feature_columns],
        rate_training["target_expected_goals"]
        * 90
        / rate_training["target_minutes"],
        model__sample_weight=sample_weight,
    )
    assist_model = build_rate_model(
        profile, parameters, random_state=int(settings["random_state"])
    )
    assist_model.fit(
        rate_training[feature_columns],
        rate_training["target_expected_assists"]
        * 90
        / rate_training["target_minutes"],
        model__sample_weight=sample_weight,
    )
    return goal_model, assist_model


def _add_fold_inputs(
    validation: pd.DataFrame,
    goal_rates: np.ndarray,
    assist_rates: np.ndarray,
    goal_priors: Mapping[str, float],
    assist_priors: Mapping[str, float],
    assist_ratio: float,
) -> pd.DataFrame:
    output = validation.copy()
    output["fold_raw_goal_rate"] = goal_rates
    output["fold_raw_assist_rate"] = assist_rates
    output["fold_goal_prior"] = output["position"].map(goal_priors).fillna(
        goal_priors["__default__"]
    )
    output["fold_assist_prior"] = output["position"].map(assist_priors).fillna(
        assist_priors["__default__"]
    )
    output["fold_goal_default_prior"] = goal_priors["__default__"]
    output["fold_assist_default_prior"] = assist_priors["__default__"]
    output["fold_assist_goal_ratio"] = assist_ratio
    return output


def predictions_from_fold_inputs(
    fold_inputs: pd.DataFrame,
    settings: Mapping[str, object],
    *,
    shrinkage_minutes: float,
    assist_ratio_multiplier: float,
    use_position_baseline: bool = False,
) -> pd.DataFrame:
    predictions = []
    for season, rows in fold_inputs.groupby("season", sort=False):
        goal_priors = rows.groupby("position")["fold_goal_prior"].first().to_dict()
        assist_priors = (
            rows.groupby("position")["fold_assist_prior"].first().to_dict()
        )
        goal_priors["__default__"] = float(rows["fold_goal_default_prior"].iloc[0])
        assist_priors["__default__"] = float(
            rows["fold_assist_default_prior"].iloc[0]
        )
        goal_rates = (
            rows["fold_goal_prior"]
            if use_position_baseline
            else rows["fold_raw_goal_rate"]
        )
        assist_rates = (
            rows["fold_assist_prior"]
            if use_position_baseline
            else rows["fold_raw_assist_rate"]
        )
        predictions.append(
            build_attacking_predictions(
                rows,
                goal_rates,
                assist_rates,
                goal_priors,
                assist_priors,
                shrinkage_minutes=shrinkage_minutes,
                assist_goal_ratio=float(rows["fold_assist_goal_ratio"].iloc[0])
                * assist_ratio_multiplier,
                minimum_rate=float(settings["minimum_rate"]),
                maximum_rate=float(settings["maximum_rate"]),
            )
        )
    return pd.concat(predictions, ignore_index=True)


def fit_candidate_folds(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    upstream_rows: pd.DataFrame,
    candidate: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Fit one xG/xA candidate across chronological folds."""
    profile = str(candidate["feature_profile"])
    parameters = candidate["parameters"]
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    fold_inputs = []
    fold_metrics = []
    fold_predictions = []

    for season, training, validation in splits:
        goal_model, assist_model = _fit_rate_models(
            training, profile, parameters, settings
        )
        validation_with_upstream = upstream_rows.loc[
            upstream_rows["season"].eq(season)
        ].copy()
        goal_rates = goal_model.predict(validation_with_upstream[feature_columns])
        assist_rates = assist_model.predict(
            validation_with_upstream[feature_columns]
        )
        goal_priors = position_rate_priors(training, "target_expected_goals")
        assist_priors = position_rate_priors(training, "target_expected_assists")
        assist_ratio = fpl_assists_per_team_goal(training)
        inputs = _add_fold_inputs(
            validation_with_upstream,
            goal_rates,
            assist_rates,
            goal_priors,
            assist_priors,
            assist_ratio,
        )
        predictions = predictions_from_fold_inputs(
            inputs,
            settings,
            shrinkage_minutes=float(settings["candidate_shrinkage_minutes"]),
            assist_ratio_multiplier=1.0,
        )
        fold_inputs.append(inputs)
        fold_predictions.append(predictions)
        fold_metrics.append(
            {"validation_season": season, **calculate_metrics(predictions)}
        )
    return (
        pd.concat(fold_inputs, ignore_index=True),
        pd.concat(fold_predictions, ignore_index=True),
        fold_metrics,
    )


def tune_candidates(
    splits: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    upstream_rows: pd.DataFrame,
    candidates: Sequence[dict[str, object]],
    settings: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    fold_results = []
    for index, candidate in enumerate(candidates, start=1):
        _, predictions, folds = fit_candidate_folds(
            splits, upstream_rows, candidate, settings
        )
        metrics = calculate_metrics(predictions)
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
            f"Candidate {index:>2}/{len(candidates)}: "
            f"{candidate['feature_profile']} attack deviance="
            f"{metrics['attacking_poisson_deviance']:.4f}, xG MAE="
            f"{metrics['xg_mae']:.4f}, xA MAE={metrics['xa_mae']:.4f}"
        )
    results_frame = pd.DataFrame(results).sort_values(
        ["attacking_poisson_deviance", "goals_poisson_deviance", "xg_mae"]
    )
    results_frame.insert(0, "rank", range(1, len(results_frame) + 1))
    return results_frame, pd.DataFrame(fold_results)


def tune_allocation(
    fold_inputs: pd.DataFrame, settings: Mapping[str, object]
) -> pd.DataFrame:
    """Tune empirical-Bayes strength and FPL-assist calibration without refitting."""
    shrinkage_values = settings["shrinkage_minutes"]
    ratio_settings = settings["assist_ratio_multiplier"]
    ratio_values = np.arange(
        float(ratio_settings["minimum"]),
        float(ratio_settings["maximum"])
        + float(ratio_settings["step"]) / 2,
        float(ratio_settings["step"]),
    )
    results = []
    for shrinkage, ratio_multiplier in itertools.product(
        shrinkage_values, ratio_values
    ):
        predictions = predictions_from_fold_inputs(
            fold_inputs,
            settings,
            shrinkage_minutes=float(shrinkage),
            assist_ratio_multiplier=float(ratio_multiplier),
        )
        metrics = calculate_metrics(predictions)
        results.append(
            {
                "shrinkage_minutes": float(shrinkage),
                "assist_ratio_multiplier": round(float(ratio_multiplier), 10),
                "attacking_poisson_deviance": metrics[
                    "attacking_poisson_deviance"
                ],
                "goals_poisson_deviance": metrics["goals_poisson_deviance"],
                "assists_poisson_deviance": metrics[
                    "assists_poisson_deviance"
                ],
                "goal_event_brier": metrics["goal_event_brier"],
                "assist_event_brier": metrics["assist_event_brier"],
                "xg_mae": metrics["xg_mae"],
                "xa_mae": metrics["xa_mae"],
            }
        )
    return pd.DataFrame(results).sort_values(
        ["attacking_poisson_deviance", "goals_poisson_deviance", "xg_mae"]
    )


def metrics_by_season(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(season): calculate_metrics(
            predictions.loc[predictions["season"].eq(season)]
        )
        for season in predictions["season"].drop_duplicates()
    }


def event_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for event, probability_column, actual_column in [
        ("goal", "goal_probability", "actual_goals"),
        ("assist", "assist_probability", "actual_assists"),
    ]:
        frame = predictions[[probability_column, actual_column]].copy()
        frame["actual_event"] = frame[actual_column].gt(0).astype(int)
        frame["probability_band"] = pd.cut(
            frame[probability_column],
            bins=[0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 1],
            include_lowest=True,
        )
        summary = (
            frame.groupby("probability_band", observed=True)
            .agg(
                rows=("actual_event", "size"),
                mean_probability=(probability_column, "mean"),
                observed_rate=("actual_event", "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "event", event)
        summary["probability_band"] = summary["probability_band"].astype(str)
        records.append(summary)
    return pd.concat(records, ignore_index=True)


def team_totals(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(TEAM_ROW_KEYS, as_index=False).agg(
        team_name=("team_name", "first"),
        opponent_team_name=("opponent_team_name", "first"),
        is_home=("is_home", "first"),
        team_expected_goals=("team_expected_goals", "first"),
        allocated_player_goals=("expected_goals", "sum"),
        allocated_fpl_assists=("expected_fpl_assists", "sum"),
        actual_goals=("actual_goals", "sum"),
        actual_assists=("actual_assists", "sum"),
    )


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


def target_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("season", as_index=False)
        .agg(
            player_fixture_rows=("player_key", "size"),
            expected_data_rows=("expected_data_available", "sum"),
            goals=("target_goals_scored", "sum"),
            assists=("target_assists", "sum"),
            xg=("target_expected_goals", "sum"),
            xa=("target_expected_assists", "sum"),
        )
        .assign(
            expected_data_coverage=lambda frame: frame["expected_data_rows"]
            / frame["player_fixture_rows"]
        )
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
    parameters: Mapping[str, object],
    shrinkage_minutes: float,
    assist_ratio_multiplier: float,
    goal_model: object,
    assist_model: object,
    goal_priors: Mapping[str, float],
    assist_priors: Mapping[str, float],
    base_assist_ratio: float,
    tuning_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    allocation_results: pd.DataFrame,
    predictions: pd.DataFrame,
    model_metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
    example_inputs: pd.DataFrame,
) -> Path:
    """Write the trained models and reporting/reproducibility artifacts."""
    model_dir = Path(str(settings.get("model_dir", DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    numeric_features, categorical_features = get_feature_columns(profile)
    feature_columns = numeric_features + categorical_features
    trained_at = datetime.now(timezone.utc).isoformat()
    final_assist_ratio = base_assist_ratio * assist_ratio_multiplier
    artifact = {
        "artifact_version": 1,
        "model_type": "attacking",
        "trained_at": trained_at,
        "goal_rate_model": goal_model,
        "assist_rate_model": assist_model,
        "feature_profile": profile,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "estimator_parameters": dict(parameters),
        "goal_position_priors": dict(goal_priors),
        "assist_position_priors": dict(assist_priors),
        "shrinkage_minutes": shrinkage_minutes,
        "base_assist_goal_ratio": base_assist_ratio,
        "assist_ratio_multiplier": assist_ratio_multiplier,
        "assist_goal_ratio": final_assist_ratio,
        "minimum_rate": float(settings["minimum_rate"]),
        "maximum_rate": float(settings["maximum_rate"]),
    }
    joblib.dump(artifact, model_dir / "model.joblib")

    tuning_results.to_csv(model_dir / "rate_model_tuning_results.csv", index=False)
    fold_results.to_csv(model_dir / "rate_model_tuning_fold_results.csv", index=False)
    allocation_results.to_csv(model_dir / "allocation_tuning.csv", index=False)
    predictions.to_csv(model_dir / "walk_forward_predictions.csv", index=False)
    team_totals(predictions).to_csv(
        model_dir / "walk_forward_team_totals.csv", index=False
    )
    event_calibration(predictions).to_csv(
        model_dir / "event_calibration.csv", index=False
    )
    feature_coverage(rows, feature_columns).to_csv(
        model_dir / "feature_coverage.csv", index=False
    )
    target_coverage(rows).to_csv(model_dir / "target_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "position": position,
                "goal_prior_per90": goal_priors[position],
                "xa_prior_per90": assist_priors[position],
            }
            for position in sorted(set(goal_priors).difference({"__default__"}))
        ]
    ).to_csv(model_dir / "position_priors.csv", index=False)

    write_json(
        model_dir / "best_parameters.json",
        {
            "feature_profile": profile,
            "estimator": "HistGradientBoostingRegressor",
            "loss": "poisson",
            "estimator_parameters": dict(parameters),
            "shrinkage_minutes": shrinkage_minutes,
            "base_fpl_assists_per_team_goal": base_assist_ratio,
            "assist_ratio_multiplier": assist_ratio_multiplier,
            "final_fpl_assists_per_team_goal": final_assist_ratio,
            "selection_metric": "walk-forward combined goal and assist Poisson deviance",
        },
    )
    write_json(
        model_dir / "metrics.json",
        {
            "walk_forward_model": model_metrics,
            "walk_forward_position_prior_baseline": baseline_metrics,
        },
    )
    write_json(
        model_dir / "feature_schema.json",
        {
            "required_identifiers": IDENTIFIER_COLUMNS,
            "required_upstream_predictions": UPSTREAM_COLUMNS,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
    )

    unavailable = rows.loc[~rows["expected_data_available"]]
    unavailable_gameweeks = (
        unavailable[["season", "gameweek"]]
        .drop_duplicates()
        .groupby("season")["gameweek"]
        .apply(lambda values: sorted(int(value) for value in values))
        .to_dict()
    )
    metadata = {
        "model_type": "attacking",
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
            "shrunk player xG and xA rates per 90",
            "statistical xG and xA for expected minutes",
            "team-reconciled expected goals and goal probability",
            "calibrated expected FPL assists and assist probability",
            "history sample minutes, position prior, and shrinkage weight",
        ],
        "upstream_contract": (
            "Each prediction batch must contain the full player set for each team "
            "fixture, expected_minutes from the minutes-state model, and "
            "team_expected_goals from the team-score model."
        ),
        "reconciliation": (
            "Player expected goals sum exactly to the upstream team expected-goals "
            "forecast. FPL assists use xA shares and a separately calibrated "
            "assists-per-team-goal total."
        ),
        "unavailable_expected_data_gameweeks": unavailable_gameweeks,
        "unavailable_expected_data_rows": int(len(unavailable)),
        "expected_data_handling": (
            "Vaastav contains false-zero xG/xA values before 2022/23 GW16. Those "
            "targets are excluded; 2022/23 expected-history features and affected "
            "2023/24 previous-season expected features are masked."
        ),
        "set_piece_caveat": (
            "The processed Vaastav data has no explicit current penalty, corner, or "
            "free-kick taker assignment. Price, role history, threat and creativity "
            "provide proxies but cannot encode a newly assigned set-piece role."
        ),
    }
    write_json(model_dir / "metadata.json", metadata)

    example_columns = list(
        dict.fromkeys(IDENTIFIER_COLUMNS + UPSTREAM_COLUMNS + feature_columns)
    )
    write_json(
        model_dir / "example_input.json",
        {
            "player_fixture_rows": records_for_json(
                example_inputs[example_columns]
            )
        },
    )
    example_output = predict_with_artifact(artifact, example_inputs)
    write_json(
        model_dir / "example_output.json",
        {"predictions": records_for_json(example_output)},
    )
    team_totals(
        example_output.assign(actual_goals=0, actual_assists=0)
        if "actual_goals" not in example_output
        else example_output
    ).to_csv(model_dir / "example_team_totals.csv", index=False)
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

    rows = load_attacking_training_data(str(settings["input_path"]))
    validation_seasons = [str(season) for season in settings["validation_seasons"]]
    splits = chronological_splits(rows, validation_seasons)
    validation_rows = rows.loc[rows["season"].isin(validation_seasons)].copy()
    upstream_rows = attach_upstream_predictions(validation_rows, settings)
    candidates = parameter_candidates(settings)
    unavailable = rows.loc[~rows["expected_data_available"]]
    print(
        f"Loaded {len(rows):,} player-fixture rows; tuning {len(candidates)} "
        f"xG/xA candidates across {len(splits)} seasons"
    )
    print(
        f"Excluded {len(unavailable):,} false-zero xG/xA rows from expected-rate fitting"
    )

    tuning_results, fold_results = tune_candidates(
        splits, upstream_rows, candidates, settings
    )
    best_row = tuning_results.iloc[0]
    best_candidate_number = int(best_row["candidate"])
    best_candidate = candidates[best_candidate_number - 1]
    best_fold_inputs, _, _ = fit_candidate_folds(
        splits, upstream_rows, best_candidate, settings
    )
    allocation_results = tune_allocation(best_fold_inputs, settings)
    best_allocation = allocation_results.iloc[0]
    best_shrinkage = float(best_allocation["shrinkage_minutes"])
    best_assist_multiplier = float(
        best_allocation["assist_ratio_multiplier"]
    )
    predictions = predictions_from_fold_inputs(
        best_fold_inputs,
        settings,
        shrinkage_minutes=best_shrinkage,
        assist_ratio_multiplier=best_assist_multiplier,
    )
    baseline_predictions = predictions_from_fold_inputs(
        best_fold_inputs,
        settings,
        shrinkage_minutes=0,
        assist_ratio_multiplier=1.0,
        use_position_baseline=True,
    )
    model_metrics = {
        "all_validation_seasons": calculate_metrics(predictions),
        "by_season": metrics_by_season(predictions),
    }
    baseline_metrics = {
        "all_validation_seasons": calculate_metrics(baseline_predictions),
        "by_season": metrics_by_season(baseline_predictions),
    }

    profile = str(best_candidate["feature_profile"])
    parameters = best_candidate["parameters"]
    goal_model, assist_model = _fit_rate_models(rows, profile, parameters, settings)
    goal_priors = position_rate_priors(rows, "target_expected_goals")
    assist_priors = position_rate_priors(rows, "target_expected_assists")
    base_assist_ratio = fpl_assists_per_team_goal(rows)
    latest_example_key = (
        best_fold_inputs.sort_values(["kickoff_time", "fixture_id"])
        .iloc[-1][["season", "fixture_id"]]
    )
    example_inputs = best_fold_inputs.loc[
        best_fold_inputs["season"].eq(latest_example_key["season"])
        & best_fold_inputs["fixture_id"].eq(latest_example_key["fixture_id"])
    ].copy()
    model_dir = write_artifacts(
        rows,
        settings,
        profile,
        parameters,
        best_shrinkage,
        best_assist_multiplier,
        goal_model,
        assist_model,
        goal_priors,
        assist_priors,
        base_assist_ratio,
        tuning_results,
        fold_results,
        allocation_results,
        predictions,
        model_metrics,
        baseline_metrics,
        example_inputs,
    )

    metrics = model_metrics["all_validation_seasons"]
    print(f"Best feature profile: {profile}")
    print(f"Best estimator parameters: {json.dumps(parameters, sort_keys=True)}")
    print(f"Best shrinkage: {best_shrinkage:.0f} history minutes")
    print(f"Best assist-ratio multiplier: {best_assist_multiplier:.2f}")
    print(
        f"Walk-forward: attack deviance={metrics['attacking_poisson_deviance']:.4f}, "
        f"goal Brier={metrics['goal_event_brier']:.4f}, "
        f"assist Brier={metrics['assist_event_brier']:.4f}, "
        f"xG MAE={metrics['xg_mae']:.4f}, xA MAE={metrics['xa_mae']:.4f}"
    )
    print(f"Wrote model and reports to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
