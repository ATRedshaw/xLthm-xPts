"""Team score model, probability calculations, and inference command."""

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


DEFAULT_MODEL_DIR = Path("data/models/team_score")
TEAM_ROW_KEYS = ["season", "fixture_id", "team_key"]
IDENTIFIER_COLUMNS = [
    "season",
    "season_start",
    "fixture_id",
    "gameweek",
    "kickoff_time",
    "team_key",
    "team_name",
    "opponent_team_key",
    "opponent_team_name",
    "is_home",
]
TARGET_COLUMNS = [
    "target_outcome_known",
    "target_team_goals",
    "target_opponent_goals",
    "target_team_clean_sheet",
]


def _side_features(suffixes: Sequence[str]) -> list[str]:
    return [
        f"feature_{side}_{suffix}"
        for side in ("team", "opponent")
        for suffix in suffixes
    ]


CONTEXT_FEATURES = [
    "gameweek",
    "is_home",
    *_side_features(
        [
            "history_fixtures",
            "season_history_fixtures",
            "rest_days",
            "is_new",
            "is_new_season",
            "gameweek_fixture_count",
            "gameweek_fixture_number",
            "previous_season_fixtures",
        ]
    ),
]


def _form_features(statistics: Sequence[str]) -> list[str]:
    suffixes: list[str] = []
    for statistic in statistics:
        suffixes.extend(
            [f"{statistic}_mean_{window}" for window in (3, 5, 10)]
        )
        suffixes.extend(
            [
                f"{statistic}_venue_mean_5",
                f"season_{statistic}_mean",
                f"previous_season_{statistic}_mean",
            ]
        )
    return _side_features(suffixes)


SCORE_FORM_FEATURES = _form_features(
    ["goals_for", "goals_against", "clean_sheet"]
)
EXPECTED_GOALS_FEATURES = _form_features(
    ["expected_goals_for", "expected_goals_against"]
)
FEATURE_PROFILES = {
    "score_form": {
        "numeric": CONTEXT_FEATURES + SCORE_FORM_FEATURES,
        "categorical": [],
    },
    "score_and_xg": {
        "numeric": CONTEXT_FEATURES + SCORE_FORM_FEATURES + EXPECTED_GOALS_FEATURES,
        "categorical": [],
    },
    "score_xg_and_team_identity": {
        "numeric": CONTEXT_FEATURES + SCORE_FORM_FEATURES + EXPECTED_GOALS_FEATURES,
        "categorical": ["team_name", "opponent_team_name"],
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
    """Return the union of fields required by every candidate feature profile."""
    columns = {
        column
        for profile in FEATURE_PROFILES
        for feature_type in ("numeric", "categorical")
        for column in FEATURE_PROFILES[profile][feature_type]
    }
    return sorted(columns)


def load_team_training_data(path: str | Path) -> pd.DataFrame:
    """Collapse player-fixture training data to one row per team and fixture."""
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

    player_rows = pd.read_csv(input_path, usecols=required, low_memory=False)
    player_rows = player_rows.loc[
        player_rows["target_outcome_known"].eq(1)
        & player_rows["target_team_goals"].notna()
        & player_rows["target_opponent_goals"].notna()
    ].copy()
    if player_rows.empty:
        raise ValueError("Training dataset has no completed fixture outcomes")

    checked_columns = [column for column in required if column not in TEAM_ROW_KEYS]
    distinct_counts = player_rows.groupby(TEAM_ROW_KEYS, dropna=False)[
        checked_columns
    ].nunique(dropna=False)
    conflicts = distinct_counts.gt(1)
    if conflicts.any().any():
        examples = [
            f"{conflict[:-1]}: {conflict[-1]}"
            for conflict in conflicts.stack().loc[lambda values: values].index[:5]
        ]
        raise ValueError(
            "Selected team fields vary within a team-fixture row: "
            + "; ".join(examples)
        )

    team_rows = (
        player_rows.groupby(TEAM_ROW_KEYS, as_index=False, dropna=False)
        .first()
        .sort_values(["season_start", "kickoff_time", "fixture_id", "is_home"])
        .reset_index(drop=True)
    )
    _validate_fixture_pairs(team_rows)
    return team_rows


def _validate_fixture_pairs(rows: pd.DataFrame) -> None:
    fixture_keys = ["season", "fixture_id"]
    group_sizes = rows.groupby(fixture_keys, dropna=False).size()
    if not group_sizes.eq(2).all():
        raise ValueError("Every fixture must have exactly two team-perspective rows")

    home_counts = rows.groupby(fixture_keys, dropna=False)["is_home"].sum()
    if not home_counts.eq(1).all():
        raise ValueError("Every fixture must have one home and one away team")

    if not set(rows["is_home"].dropna().unique()).issubset({0, 1, False, True}):
        raise ValueError("is_home must contain only zero/one values")


def build_goal_pipeline(
    profile: str,
    parameters: Mapping[str, object],
    *,
    random_state: int,
) -> Pipeline:
    """Build a Poisson gradient-boosting pipeline for team goals."""
    numeric_features, categorical_features = get_feature_columns(profile)
    transformers: list[tuple[str, object, list[str]]] = [
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

    estimator = HistGradientBoostingRegressor(
        loss="poisson",
        early_stopping=False,
        random_state=random_state,
        **parameters,
    )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("goals", estimator),
        ]
    )


def poisson_probabilities(expected_goals: float, max_goals: int) -> np.ndarray:
    """Return Poisson probabilities from zero through max_goals."""
    probabilities = np.empty(max_goals + 1, dtype=float)
    probabilities[0] = np.exp(-expected_goals)
    for goals in range(1, max_goals + 1):
        probabilities[goals] = (
            probabilities[goals - 1] * expected_goals / goals
        )
    return probabilities


def score_probability_matrix(
    home_rate: float,
    away_rate: float,
    *,
    rho: float,
    max_goals: int,
) -> np.ndarray:
    """Build a truncated, normalised Dixon-Coles score probability matrix."""
    home_probabilities = poisson_probabilities(home_rate, max_goals)
    away_probabilities = poisson_probabilities(away_rate, max_goals)
    matrix = np.outer(home_probabilities, away_probabilities)

    adjustments = {
        (0, 0): 1 - home_rate * away_rate * rho,
        (0, 1): 1 + home_rate * rho,
        (1, 0): 1 + away_rate * rho,
        (1, 1): 1 - rho,
    }
    if any(adjustment <= 0 for adjustment in adjustments.values()):
        raise ValueError("Dixon-Coles rho produces a non-positive probability")
    for score, adjustment in adjustments.items():
        matrix[score] *= adjustment

    return matrix / matrix.sum()


def forecast_team_rows(
    rows: pd.DataFrame,
    raw_goal_rates: Sequence[float],
    *,
    rho: float,
    max_goals: int,
    minimum_goal_rate: float,
    maximum_goal_rate: float,
    include_scorelines: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pair team rates and return team, fixture, and optional scoreline forecasts."""
    forecast_rows = rows.reset_index(drop=True).copy()
    if len(forecast_rows) != len(raw_goal_rates):
        raise ValueError("A goal rate is required for every team row")
    _validate_fixture_pairs(forecast_rows)
    forecast_rows["raw_goal_rate"] = np.clip(
        np.asarray(raw_goal_rates, dtype=float),
        minimum_goal_rate,
        maximum_goal_rate,
    )

    team_records: list[dict[str, object]] = []
    fixture_records: list[dict[str, object]] = []
    scoreline_records: list[dict[str, object]] = []
    fixture_keys = ["season", "fixture_id"]

    for (season, fixture_id), fixture in forecast_rows.groupby(
        fixture_keys, sort=False, dropna=False
    ):
        home = fixture.loc[fixture["is_home"].eq(1)].iloc[0]
        away = fixture.loc[fixture["is_home"].eq(0)].iloc[0]
        matrix = score_probability_matrix(
            float(home["raw_goal_rate"]),
            float(away["raw_goal_rate"]),
            rho=rho,
            max_goals=max_goals,
        )
        goal_values = np.arange(max_goals + 1)
        home_expected_goals = float((matrix * goal_values[:, None]).sum())
        away_expected_goals = float((matrix * goal_values[None, :]).sum())
        home_clean_sheet = float(matrix[:, 0].sum())
        away_clean_sheet = float(matrix[0, :].sum())
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        most_likely = np.unravel_index(int(matrix.argmax()), matrix.shape)

        fixture_record: dict[str, object] = {
            "season": season,
            "fixture_id": fixture_id,
            "gameweek": home["gameweek"],
            "kickoff_time": home["kickoff_time"],
            "home_team": home["team_name"],
            "away_team": away["team_name"],
            "raw_home_goal_rate": float(home["raw_goal_rate"]),
            "raw_away_goal_rate": float(away["raw_goal_rate"]),
            "home_expected_goals": home_expected_goals,
            "away_expected_goals": away_expected_goals,
            "home_win_probability": home_win,
            "draw_probability": draw,
            "away_win_probability": away_win,
            "home_clean_sheet_probability": home_clean_sheet,
            "away_clean_sheet_probability": away_clean_sheet,
            "most_likely_home_goals": int(most_likely[0]),
            "most_likely_away_goals": int(most_likely[1]),
            "most_likely_score_probability": float(matrix[most_likely]),
        }
        if "target_team_goals" in fixture:
            actual_home_goals = int(home["target_team_goals"])
            actual_away_goals = int(away["target_team_goals"])
            fixture_record.update(
                {
                    "actual_home_goals": actual_home_goals,
                    "actual_away_goals": actual_away_goals,
                    "actual_score_probability": float(
                        matrix[actual_home_goals, actual_away_goals]
                        if actual_home_goals <= max_goals
                        and actual_away_goals <= max_goals
                        else 1e-15
                    ),
                    "actual_result_probability": (
                        home_win
                        if actual_home_goals > actual_away_goals
                        else away_win
                        if actual_home_goals < actual_away_goals
                        else draw
                    ),
                }
            )
        fixture_records.append(fixture_record)

        common = {
            "season": season,
            "fixture_id": fixture_id,
            "gameweek": home["gameweek"],
            "kickoff_time": home["kickoff_time"],
        }
        perspectives = [
            (
                home,
                away,
                home_expected_goals,
                away_expected_goals,
                home_clean_sheet,
                home_win,
                away_win,
            ),
            (
                away,
                home,
                away_expected_goals,
                home_expected_goals,
                away_clean_sheet,
                away_win,
                home_win,
            ),
        ]
        for (
            team,
            opponent,
            expected_for,
            expected_against,
            clean_sheet,
            win_probability,
            loss_probability,
        ) in perspectives:
            record: dict[str, object] = {
                **common,
                "team_key": team["team_key"],
                "team_name": team["team_name"],
                "opponent_team_key": opponent["team_key"],
                "opponent_team_name": opponent["team_name"],
                "is_home": int(team["is_home"]),
                "raw_goal_rate": float(team["raw_goal_rate"]),
                "expected_goals_for": expected_for,
                "expected_goals_against": expected_against,
                "clean_sheet_probability": clean_sheet,
                "win_probability": win_probability,
                "draw_probability": draw,
                "loss_probability": loss_probability,
            }
            if "target_team_goals" in fixture:
                record.update(
                    {
                        "actual_goals_for": int(team["target_team_goals"]),
                        "actual_goals_against": int(team["target_opponent_goals"]),
                        "actual_clean_sheet": int(team["target_team_clean_sheet"]),
                    }
                )
            team_records.append(record)

        if include_scorelines:
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    scoreline_records.append(
                        {
                            "season": season,
                            "fixture_id": fixture_id,
                            "home_team": home["team_name"],
                            "away_team": away["team_name"],
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "probability": float(matrix[home_goals, away_goals]),
                        }
                    )

    return (
        pd.DataFrame(team_records),
        pd.DataFrame(fixture_records),
        pd.DataFrame(scoreline_records),
    )


def predict_with_artifact(
    artifact: Mapping[str, object],
    fixture_rows: pd.DataFrame,
    *,
    include_scorelines: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict a set of fixtures using a saved team-score artifact."""
    feature_columns = list(artifact["feature_columns"])
    required = set(IDENTIFIER_COLUMNS).union(feature_columns)
    missing = sorted(required.difference(fixture_rows.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")

    pipeline = artifact["pipeline"]
    raw_rates = pipeline.predict(fixture_rows[feature_columns])
    return forecast_team_rows(
        fixture_rows,
        raw_rates,
        rho=float(artifact["dixon_coles_rho"]),
        max_goals=int(artifact["max_goals"]),
        minimum_goal_rate=float(artifact["minimum_goal_rate"]),
        maximum_goal_rate=float(artifact["maximum_goal_rate"]),
        include_scorelines=include_scorelines,
    )


def load_artifact(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, object]:
    """Load a trained team-score model artifact."""
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
    parser.add_argument("--input", required=True, help="Fixture-row JSON input.")
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
        payload.get("fixture_rows") if isinstance(payload, dict) else payload
    )
    if not isinstance(fixture_records, list):
        raise ValueError("Input JSON must be a list or contain a fixture_rows list")

    artifact = load_artifact(args.model_dir)
    teams, fixtures, scorelines = predict_with_artifact(
        artifact, pd.DataFrame(fixture_records), include_scorelines=True
    )
    output = {
        "teams": records_for_json(teams),
        "fixtures": records_for_json(fixtures),
        "scorelines": records_for_json(scorelines),
    }
    rendered = json.dumps(output, indent=2, allow_nan=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote team-score predictions to {output_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
