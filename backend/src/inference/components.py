"""Run trained component models in their required dependency order."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from training.attacking.model import predict_with_artifact as predict_attacking
from training.bonus.model import build_bps_predictions
from training.defensive.model import predict_with_artifact as predict_defensive
from training.minutes_state.model import APPEARED_STATES, MINUTES_STATES
from training.minutes_state.model import predict_with_artifact as predict_minutes
from training.misc_events.model import predict_with_artifact as predict_miscellaneous
from training.team_score.model import predict_with_artifact as predict_team_score


PLAYER_KEYS = ["season", "fixture_id", "player_key"]
TEAM_KEYS = ["season", "fixture_id", "team_key"]


def load_artifacts(model_dirs: Mapping[str, object]) -> dict[str, dict[str, object]]:
    required = ["team_score", "minutes_state", "attacking", "defensive", "misc_events", "bonus"]
    missing = sorted(set(required).difference(model_dirs))
    if missing:
        raise ValueError(f"Model directories are missing: {', '.join(missing)}")
    artifacts = {}
    for name in required:
        path = Path(str(model_dirs[name])) / "model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Trained model does not exist: {path}")
        artifacts[name] = joblib.load(path)
    return artifacts


def _team_feature_rows(features: pd.DataFrame, artifact: Mapping[str, object]) -> pd.DataFrame:
    identifiers = [
        "season", "season_start", "fixture_id", "gameweek", "kickoff_time",
        "team_key", "team_name", "opponent_team_key", "opponent_team_name", "is_home",
    ]
    columns = list(dict.fromkeys(identifiers + list(artifact["feature_columns"])))
    selected = features[columns]
    varying = selected.groupby(TEAM_KEYS, dropna=False)[list(artifact["feature_columns"])].nunique(dropna=False)
    if varying.gt(1).any(axis=None):
        raise ValueError("Team-model features vary within a team fixture")
    return selected.groupby(TEAM_KEYS, as_index=False, dropna=False).first()


def apply_live_availability(feature_rows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply the official next-round availability flag to the next Gameweek only."""
    output = predictions.copy()
    gameweeks = pd.to_numeric(feature_rows["gameweek"], errors="coerce")
    next_gameweek = gameweeks.min()
    next_round = gameweeks.eq(next_gameweek)
    stated = feature_rows["live_availability_probability"].fillna(1).to_numpy(float)
    applied = np.where(next_round, stated, 1.0)
    state_columns = [f"probability_{state}" for state in MINUTES_STATES]
    appeared_columns = [f"probability_{state}" for state in APPEARED_STATES]
    output["base_appearance_probability"] = output[appeared_columns].sum(axis=1)
    output[appeared_columns] = output[appeared_columns].mul(applied, axis=0)
    output["probability_did_not_play"] = 1 - output[appeared_columns].sum(axis=1)
    if not np.allclose(output[state_columns].sum(axis=1), 1):
        raise ValueError("Availability-adjusted minutes probabilities do not sum to one")
    output["appearance_probability"] = output[appeared_columns].sum(axis=1)
    output["start_probability"] = output[["probability_start_under_60", "probability_start_60_plus"]].sum(axis=1)
    output["sixty_plus_probability"] = output[["probability_sub_60_plus", "probability_start_60_plus"]].sum(axis=1)
    output["expected_minutes"] = sum(
        output[f"probability_{state}"] * output[f"minutes_if_{state}"]
        for state in APPEARED_STATES
    )
    output["expected_minutes_if_appears"] = np.divide(
        output["expected_minutes"], output["appearance_probability"],
        out=np.zeros(len(output), dtype=float), where=output["appearance_probability"].gt(0),
    )
    output["availability_probability_applied"] = applied
    output["most_likely_state"] = np.asarray(MINUTES_STATES)[output[state_columns].to_numpy().argmax(axis=1)]
    return output


def forecast_components(
    feature_rows: pd.DataFrame,
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, pd.DataFrame]:
    """Return all component forecasts needed by the xPts simulator."""
    features = feature_rows.copy()
    features["kickoff_time"] = pd.to_datetime(features["kickoff_time"], utc=True)
    if features.duplicated(PLAYER_KEYS).any():
        raise ValueError("Future features contain duplicate player-fixture rows")

    team_inputs = _team_feature_rows(features, artifacts["team_score"])
    team, fixtures, scorelines = predict_team_score(
        artifacts["team_score"], team_inputs, include_scorelines=True
    )
    minutes = apply_live_availability(
        features, predict_minutes(artifacts["minutes_state"], features)
    )
    team_upstream = team[TEAM_KEYS + ["expected_goals_for", "expected_goals_against", "clean_sheet_probability"]]
    minutes_upstream = minutes[PLAYER_KEYS + ["appearance_probability", "start_probability", "sixty_plus_probability", "expected_minutes"]]
    attacking_inputs = features.merge(
        minutes_upstream[PLAYER_KEYS + ["expected_minutes"]],
        on=PLAYER_KEYS, how="left", validate="one_to_one",
    ).merge(
        team_upstream[TEAM_KEYS + ["expected_goals_for"]].rename(columns={"expected_goals_for": "team_expected_goals"}),
        on=TEAM_KEYS, how="left", validate="many_to_one",
    )
    attacking = predict_attacking(artifacts["attacking"], attacking_inputs)

    defensive_inputs = features.merge(
        minutes_upstream[PLAYER_KEYS + ["expected_minutes", "sixty_plus_probability"]],
        on=PLAYER_KEYS, how="left", validate="one_to_one",
    ).merge(
        team_upstream.rename(columns={
            "expected_goals_against": "team_expected_goals_against",
            "clean_sheet_probability": "team_clean_sheet_probability",
        })[TEAM_KEYS + ["team_expected_goals_against", "team_clean_sheet_probability"]],
        on=TEAM_KEYS, how="left", validate="many_to_one",
    )
    defensive = predict_defensive(artifacts["defensive"], defensive_inputs)

    miscellaneous_inputs = features.merge(
        minutes_upstream[PLAYER_KEYS + ["expected_minutes"]],
        on=PLAYER_KEYS, how="left", validate="one_to_one",
    )
    miscellaneous = predict_miscellaneous(artifacts["misc_events"], miscellaneous_inputs)

    bonus_inputs = features.merge(minutes_upstream, on=PLAYER_KEYS, how="left", validate="one_to_one")
    bonus_inputs = bonus_inputs.merge(
        team_upstream.rename(columns={
            "expected_goals_for": "team_expected_goals_for",
            "expected_goals_against": "team_expected_goals_against",
            "clean_sheet_probability": "team_clean_sheet_probability",
        }),
        on=TEAM_KEYS, how="left", validate="many_to_one",
    )
    bonus_inputs = bonus_inputs.merge(
        attacking[PLAYER_KEYS + ["expected_goals", "expected_fpl_assists"]],
        on=PLAYER_KEYS, how="left", validate="one_to_one",
    ).merge(
        defensive[PLAYER_KEYS + [
            "player_clean_sheet_probability", "expected_goals_conceded",
            "expected_saves", "expected_penalty_saves", "expected_defensive_contributions",
        ]],
        on=PLAYER_KEYS, how="left", validate="one_to_one",
    ).merge(
        miscellaneous[PLAYER_KEYS + [
            "expected_yellow_cards", "expected_red_cards", "expected_own_goals",
            "expected_penalties_missed",
        ]],
        on=PLAYER_KEYS, how="left", validate="one_to_one",
    )
    bonus_artifact = artifacts["bonus"]
    bonus_features = list(bonus_artifact["feature_columns"])
    conditional_bps = bonus_artifact["bps_model"].predict(bonus_inputs[bonus_features])
    bps = build_bps_predictions(
        bonus_inputs, conditional_bps, bonus_artifact["residual_std_by_position"]
    )
    return {
        "team": team,
        "fixtures": fixtures,
        "scorelines": scorelines,
        "minutes": minutes,
        "attacking": attacking,
        "defensive": defensive,
        "misc_events": miscellaneous,
        "bps": bps,
    }
