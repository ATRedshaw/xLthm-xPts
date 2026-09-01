"""Joint fixture simulation and FPL points aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd


PLAYER_KEYS = ["season", "fixture_id", "player_key"]
FIXTURE_KEYS = ["season", "fixture_id"]
MINUTES_STATES = [
    "did_not_play", "sub_under_60", "sub_60_plus",
    "start_under_60", "start_60_plus",
]
def merge_component_frames(
    features: pd.DataFrame,
    components: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    base_columns = [
        "season", "fixture_id", "gameweek", "kickoff_time", "player_id",
        "player_code", "player_key", "player_name", "position", "team_id",
        "team_key", "team_name", "opponent_team_id", "opponent_team_key",
        "opponent_team_name", "is_home", "live_status",
        "live_availability_probability", "feature_player_history_minutes",
    ]
    output = features[base_columns].copy()
    selections = {
        "minutes": [
            *(f"probability_{state}" for state in MINUTES_STATES),
            "minutes_if_sub_under_60", "minutes_if_sub_60_plus",
            "minutes_if_start_under_60", "minutes_if_start_60_plus",
            "appearance_probability", "start_probability", "sixty_plus_probability",
            "expected_minutes", "availability_probability_applied",
        ],
        "attacking": [
            "expected_goals", "goal_probability", "expected_fpl_assists",
            "assist_probability", "history_weight",
        ],
        "defensive": [
            "team_clean_sheet_probability", "player_clean_sheet_probability",
            "expected_goals_conceded", "save_rate_per90", "expected_saves",
            "expected_penalty_saves", "defensive_contribution_rate_per90",
            "expected_defensive_contributions",
            "defensive_contribution_points_probability",
        ],
        "misc_events": [
            "yellow_cards_rate_per90", "expected_yellow_cards", "yellow_card_probability",
            "red_cards_rate_per90", "expected_red_cards", "red_card_probability",
            "own_goals_rate_per90", "expected_own_goals", "own_goal_probability",
            "penalties_missed_rate_per90", "expected_penalties_missed",
            "penalty_miss_probability",
        ],
        "bps": [
            "conditional_expected_bps", "expected_bps", "bps_residual_std",
            "bps_residual_scale",
        ],
    }
    for name, columns in selections.items():
        frame = components[name][PLAYER_KEYS + columns]
        output = output.merge(frame, on=PLAYER_KEYS, how="left", validate="one_to_one")
    team_columns = [
        "expected_goals_for", "expected_goals_against", "clean_sheet_probability",
        "win_probability", "draw_probability", "loss_probability",
    ]
    output = output.merge(
        components["team"][["season", "fixture_id", "team_key"] + team_columns],
        on=["season", "fixture_id", "team_key"], how="left", validate="many_to_one",
    )
    if output.isna().all(axis=1).any():
        raise ValueError("A component row is completely empty after merging")
    return output.sort_values(["kickoff_time", "fixture_id", "player_id"]).reset_index(drop=True)


def _sample_minutes(fixture: pd.DataFrame, simulations: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    probabilities = fixture[[f"probability_{state}" for state in MINUTES_STATES]].to_numpy(float)
    cumulative = np.cumsum(probabilities, axis=1)
    state = np.minimum(
        (rng.random((simulations, len(fixture), 1)) > cumulative[None, :, :]).sum(axis=2),
        len(MINUTES_STATES) - 1,
    )
    durations = fixture[[
        "minutes_if_sub_under_60", "minutes_if_sub_60_plus",
        "minutes_if_start_under_60", "minutes_if_start_60_plus",
    ]].to_numpy(float)
    minutes = np.zeros((simulations, len(fixture)), dtype=np.int16)
    for state_index in range(1, 5):
        minutes = np.where(
            state == state_index,
            np.rint(durations[:, state_index - 1])[None, :].astype(np.int16),
            minutes,
        )
    return state, minutes


def _sample_indices(weights: np.ndarray, fallback: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    totals = weights.sum(axis=1)
    missing = totals <= 0
    if missing.any():
        weights = weights.copy()
        weights[missing] = fallback[missing]
        totals = weights.sum(axis=1)
    still_missing = totals <= 0
    if still_missing.any():
        weights[still_missing] = 1
        totals = weights.sum(axis=1)
    cumulative = np.cumsum(weights / totals[:, None], axis=1)
    draws = rng.random(len(weights))
    return np.minimum((draws[:, None] > cumulative).sum(axis=1), weights.shape[1] - 1)


def _sample_team_goal_actions(
    team_indices: np.ndarray,
    opponent_indices: np.ndarray,
    team_goals: np.ndarray,
    minutes: np.ndarray,
    fixture: pd.DataFrame,
    goals: np.ndarray,
    assists: np.ndarray,
    own_goals: np.ndarray,
    rng: np.random.Generator,
) -> None:
    scoring_minutes = minutes[:, team_indices]
    opponent_minutes = minutes[:, opponent_indices]
    goal_rates = np.divide(
        fixture.iloc[team_indices]["expected_goals"].to_numpy(float) * 90,
        fixture.iloc[team_indices]["expected_minutes"].to_numpy(float),
        out=np.zeros(len(team_indices)),
        where=fixture.iloc[team_indices]["expected_minutes"].to_numpy(float) > 0,
    )
    assist_rates = np.divide(
        fixture.iloc[team_indices]["expected_fpl_assists"].to_numpy(float) * 90,
        fixture.iloc[team_indices]["expected_minutes"].to_numpy(float),
        out=np.zeros(len(team_indices)),
        where=fixture.iloc[team_indices]["expected_minutes"].to_numpy(float) > 0,
    )
    own_goal_rates = fixture.iloc[opponent_indices]["own_goals_rate_per90"].to_numpy(float)
    expected_team_goals = float(fixture.iloc[team_indices]["expected_goals_for"].iloc[0])
    expected_opponent_own_goals = float(fixture.iloc[opponent_indices]["expected_own_goals"].sum())
    own_goal_per_goal = min(expected_opponent_own_goals / max(expected_team_goals, 1e-8), 1.0)
    assist_per_goal = min(float(fixture.iloc[team_indices]["expected_fpl_assists"].sum()) / max(expected_team_goals, 1e-8), 1.0)
    goal_weights = scoring_minutes * goal_rates[None, :]
    assist_weights = scoring_minutes * assist_rates[None, :]
    opponent_own_goal_weights = opponent_minutes * own_goal_rates[None, :]
    appeared_fallback = scoring_minutes > 0
    opponent_fallback = opponent_minutes > 0
    max_goals = int(team_goals.max())
    simulation_rows = np.arange(len(team_goals))
    for goal_number in range(max_goals):
        active = team_goals > goal_number
        if not active.any():
            continue
        active_rows = simulation_rows[active]
        is_own_goal = rng.random(active.sum()) < own_goal_per_goal
        if is_own_goal.any():
            own_rows = active_rows[is_own_goal]
            selected = _sample_indices(
                opponent_own_goal_weights[own_rows], opponent_fallback[own_rows], rng
            )
            own_goals[own_rows, opponent_indices[selected]] += 1
        normal_rows = active_rows[~is_own_goal]
        selected_scorers = np.array([], dtype=int)
        if len(normal_rows):
            selected_scorers = _sample_indices(
                goal_weights[normal_rows], appeared_fallback[normal_rows], rng
            )
            goals[normal_rows, team_indices[selected_scorers]] += 1
        assist_rows = active_rows[rng.random(active.sum()) < assist_per_goal]
        if not len(assist_rows):
            continue
        weights = assist_weights[assist_rows].copy()
        fallback = appeared_fallback[assist_rows].copy()
        normal_lookup = {row: scorer for row, scorer in zip(normal_rows, selected_scorers)}
        for row_index, simulation_row in enumerate(assist_rows):
            scorer = normal_lookup.get(simulation_row)
            if scorer is not None:
                weights[row_index, scorer] = 0
                fallback[row_index, scorer] = False
        selected_assists = _sample_indices(weights, fallback, rng)
        assists[assist_rows, team_indices[selected_assists]] += 1


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
    bonus[second & (first_count == 2)[:, None]] = 1
    one_first = first_count == 1
    bonus[second & one_first[:, None]] = 2
    second_count = second.sum(axis=1)
    remaining = np.where(second, -np.inf, remaining)
    third_value = np.max(remaining, axis=1)
    third = (remaining == third_value[:, None]) & np.isfinite(remaining)
    bonus[third & (one_first & (second_count == 1))[:, None]] = 1
    return bonus


def _point_summary(samples: np.ndarray) -> dict[str, object]:
    values, counts = np.unique(samples, return_counts=True)
    simulations = len(samples)
    return {
        "negative_points": round(float(np.mean(samples < 0)), 6),
        "zero_points": round(float(np.mean(samples == 0)), 6),
        "two_plus_points": round(float(np.mean(samples >= 2)), 6),
        "five_plus_points": round(float(np.mean(samples >= 5)), 6),
        "ten_plus_points": round(float(np.mean(samples >= 10)), 6),
        "percentiles": {
            "p10": float(np.percentile(samples, 10)),
            "p25": float(np.percentile(samples, 25)),
            "p50": float(np.percentile(samples, 50)),
            "p75": float(np.percentile(samples, 75)),
            "p90": float(np.percentile(samples, 90)),
        },
        "points_distribution": {
            str(int(value)): round(float(count / simulations), 6)
            for value, count in zip(values, counts)
        },
    }


def _simulate_fixture(
    fixture: pd.DataFrame,
    scorelines: pd.DataFrame,
    rules: Mapping[str, object],
    simulations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    _, minutes = _sample_minutes(fixture, simulations, rng)
    appeared = minutes > 0
    sixty_plus = minutes >= 60
    score_probabilities = scorelines["probability"].to_numpy(float)
    score_probabilities /= score_probabilities.sum()
    score_indices = rng.choice(len(scorelines), size=simulations, p=score_probabilities)
    home_goals = scorelines.iloc[score_indices]["home_goals"].to_numpy(int)
    away_goals = scorelines.iloc[score_indices]["away_goals"].to_numpy(int)
    home_indices = np.flatnonzero(fixture["is_home"].eq(1).to_numpy())
    away_indices = np.flatnonzero(fixture["is_home"].eq(0).to_numpy())

    goals = np.zeros_like(minutes, dtype=np.int8)
    assists = np.zeros_like(minutes, dtype=np.int8)
    own_goals = np.zeros_like(minutes, dtype=np.int8)
    _sample_team_goal_actions(home_indices, away_indices, home_goals, minutes, fixture, goals, assists, own_goals, rng)
    _sample_team_goal_actions(away_indices, home_indices, away_goals, minutes, fixture, goals, assists, own_goals, rng)

    exposure = minutes / 90
    saves = rng.poisson(fixture["save_rate_per90"].to_numpy(float)[None, :] * exposure)
    penalty_rate = np.divide(
        fixture["expected_penalty_saves"].to_numpy(float) * 90,
        fixture["expected_minutes"].to_numpy(float),
        out=np.zeros(len(fixture)), where=fixture["expected_minutes"].to_numpy(float) > 0,
    )
    penalty_saves = rng.poisson(penalty_rate[None, :] * exposure)
    defcons = rng.poisson(
        fixture["defensive_contribution_rate_per90"].to_numpy(float)[None, :] * exposure
    )
    yellow = rng.random(minutes.shape) < (1 - np.exp(-fixture["yellow_cards_rate_per90"].to_numpy(float)[None, :] * exposure))
    red = rng.random(minutes.shape) < (1 - np.exp(-fixture["red_cards_rate_per90"].to_numpy(float)[None, :] * exposure))
    penalty_misses = rng.random(minutes.shape) < (1 - np.exp(-fixture["penalties_missed_rate_per90"].to_numpy(float)[None, :] * exposure))

    goals_against = np.zeros_like(minutes, dtype=np.int8)
    goals_against[:, home_indices] = rng.binomial(
        away_goals[:, None], np.clip(exposure[:, home_indices], 0, 1)
    )
    goals_against[:, away_indices] = rng.binomial(
        home_goals[:, None], np.clip(exposure[:, away_indices], 0, 1)
    )
    team_clean_sheet = np.zeros_like(minutes, dtype=bool)
    team_clean_sheet[:, home_indices] = away_goals[:, None] == 0
    team_clean_sheet[:, away_indices] = home_goals[:, None] == 0

    positions = fixture["position"].to_numpy(str)
    goal_values = np.array([rules["goal"][position] for position in positions], dtype=int)
    clean_values = np.array([rules["clean_sheet"][position] for position in positions], dtype=int)
    appearance_points = np.where(sixty_plus, int(rules["appearance_60_plus"]), np.where(appeared, int(rules["appearance_under_60"]), 0))
    point_samples: dict[str, np.ndarray] = {
        "appearance": appearance_points,
        "goals": goals * goal_values[None, :],
        "assists": assists * int(rules["assist"]),
        "clean_sheet": (team_clean_sheet & sixty_plus) * clean_values[None, :],
        "saves": (saves // int(rules["saves_per_point"])),
        "penalty_saves": penalty_saves * int(rules["penalty_save"]),
        "defensive_contributions": np.zeros_like(minutes, dtype=np.int16),
        "goals_conceded": np.zeros_like(minutes, dtype=np.int16),
        "yellow_cards": yellow * int(rules["yellow_card"]),
        "red_cards": red * int(rules["red_card"]),
        "own_goals": own_goals * int(rules["own_goal"]),
        "penalty_misses": penalty_misses * int(rules["penalty_miss"]),
    }
    for index, position in enumerate(positions):
        threshold = rules["defensive_contribution_threshold"].get(position)
        if threshold is not None:
            point_samples["defensive_contributions"][:, index] = (
                defcons[:, index] >= int(threshold)
            ) * int(rules["defensive_contribution_points"])
        if position in {"GK", "DEF"}:
            point_samples["goals_conceded"][:, index] = -(
                goals_against[:, index] // int(rules["goals_conceded_per_deduction"])
            )
    conditional_bps = fixture["conditional_expected_bps"].to_numpy(float)
    bps_std = fixture["bps_residual_std"].to_numpy(float) * fixture["bps_residual_scale"].to_numpy(float)
    sampled_bps = np.rint(rng.normal(conditional_bps, bps_std, minutes.shape))
    bonus = _allocate_bonus(sampled_bps, appeared)
    point_samples["bonus"] = bonus
    total_points = sum(point_samples.values())
    action_samples = {
        "goals": goals,
        "assists": assists,
        "saves": saves,
        "penalty_saves": penalty_saves,
        "defensive_contributions": defcons,
        "goals_conceded": goals_against,
        "yellow_cards": yellow,
        "red_cards": red,
        "own_goals": own_goals,
        "penalty_misses": penalty_misses,
        "clean_sheet": team_clean_sheet & sixty_plus,
        "bonus": bonus,
    }
    return total_points, minutes, point_samples, action_samples


def simulate_all_fixtures(
    player_components: pd.DataFrame,
    fixture_forecasts: pd.DataFrame,
    scorelines: pd.DataFrame,
    *,
    rules: Mapping[str, object],
    simulations: int,
    random_state: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[tuple[str, int], dict[str, object]], dict[str, float]]:
    if simulations < 1:
        raise ValueError("simulation_count must be positive")
    rng = np.random.default_rng(random_state)
    fixture_projections: dict[str, list[dict[str, object]]] = defaultdict(list)
    gameweek_samples: dict[tuple[str, int], dict[str, object]] = {}
    maximum_team_goal_error = 0.0
    maximum_minutes_error = 0.0
    for fixture_key, fixture in player_components.groupby(FIXTURE_KEYS, sort=False):
        season, fixture_id = fixture_key
        fixture = fixture.reset_index(drop=True)
        fixture_scorelines = scorelines.loc[
            scorelines["season"].eq(season) & scorelines["fixture_id"].eq(fixture_id)
        ]
        total, sampled_minutes, point_samples, actions = _simulate_fixture(
            fixture, fixture_scorelines, rules, simulations, rng
        )
        team_forecast = fixture_forecasts.loc[
            fixture_forecasts["season"].eq(season) & fixture_forecasts["fixture_id"].eq(fixture_id)
        ].iloc[0]
        sampled_home_goals = actions["goals"][:, fixture["is_home"].eq(1).to_numpy()].sum(axis=1) + actions["own_goals"][:, fixture["is_home"].eq(0).to_numpy()].sum(axis=1)
        sampled_away_goals = actions["goals"][:, fixture["is_home"].eq(0).to_numpy()].sum(axis=1) + actions["own_goals"][:, fixture["is_home"].eq(1).to_numpy()].sum(axis=1)
        maximum_team_goal_error = max(
            maximum_team_goal_error,
            abs(float(sampled_home_goals.mean()) - float(team_forecast["home_expected_goals"])),
            abs(float(sampled_away_goals.mean()) - float(team_forecast["away_expected_goals"])),
        )
        maximum_minutes_error = max(
            maximum_minutes_error,
            float(np.max(np.abs(sampled_minutes.mean(axis=0) - fixture["expected_minutes"].to_numpy(float)))),
        )
        for index, player in fixture.iterrows():
            player_id = str(int(player["player_id"]))
            contribution = {
                name: round(float(values[:, index].mean()), 4)
                for name, values in point_samples.items()
            }
            probabilities = {
                "appearance": round(float(player["appearance_probability"]), 6),
                "start": round(float(player["start_probability"]), 6),
                "sixty_plus": round(float(player["sixty_plus_probability"]), 6),
                "goal": round(float(np.mean(actions["goals"][:, index] > 0)), 6),
                "assist": round(float(np.mean(actions["assists"][:, index] > 0)), 6),
                "clean_sheet": round(float(np.mean(actions["clean_sheet"][:, index])), 6),
                "save_points": round(float(np.mean(actions["saves"][:, index] >= int(rules["saves_per_point"]))), 6),
                "penalty_save": round(float(np.mean(actions["penalty_saves"][:, index] > 0)), 6),
                "defensive_contribution_points": round(float(np.mean(point_samples["defensive_contributions"][:, index] > 0)), 6),
                "yellow_card": round(float(np.mean(actions["yellow_cards"][:, index] > 0)), 6),
                "red_card": round(float(np.mean(actions["red_cards"][:, index] > 0)), 6),
                "own_goal": round(float(np.mean(actions["own_goals"][:, index] > 0)), 6),
                "penalty_miss": round(float(np.mean(actions["penalty_misses"][:, index] > 0)), 6),
                "any_bonus": round(float(np.mean(actions["bonus"][:, index] > 0)), 6),
                "one_bonus": round(float(np.mean(actions["bonus"][:, index] == 1)), 6),
                "two_bonus": round(float(np.mean(actions["bonus"][:, index] == 2)), 6),
                "three_bonus": round(float(np.mean(actions["bonus"][:, index] == 3)), 6),
            }
            expected_actions = {
                "minutes": round(float(sampled_minutes[:, index].mean()), 4),
                "goals": round(float(actions["goals"][:, index].mean()), 4),
                "assists": round(float(actions["assists"][:, index].mean()), 4),
                "saves": round(float(actions["saves"][:, index].mean()), 4),
                "penalty_saves": round(float(actions["penalty_saves"][:, index].mean()), 4),
                "defensive_contributions": round(float(actions["defensive_contributions"][:, index].mean()), 4),
                "goals_conceded": round(float(actions["goals_conceded"][:, index].mean()), 4),
                "yellow_cards": round(float(actions["yellow_cards"][:, index].mean()), 4),
                "red_cards": round(float(actions["red_cards"][:, index].mean()), 4),
                "own_goals": round(float(actions["own_goals"][:, index].mean()), 4),
                "penalty_misses": round(float(actions["penalty_misses"][:, index].mean()), 4),
                "bps": round(float(player["expected_bps"]), 4),
                "bonus": round(float(actions["bonus"][:, index].mean()), 4),
            }
            projection = {
                "fixture_id": int(fixture_id),
                "kickoff_time": str(player["kickoff_time"]),
                "opponent": {
                    "id": int(player["opponent_team_id"]),
                    "name": str(player["opponent_team_name"]),
                },
                "is_home": bool(player["is_home"]),
                "xpts": round(float(total[:, index].mean()), 4),
                "xmins": round(float(player["expected_minutes"]), 4),
                "action_probabilities": probabilities,
                "expected_actions": expected_actions,
                "xpts_breakdown": contribution,
                "outcome_probabilities": _point_summary(total[:, index]),
                "team_forecast": {
                    "expected_goals_for": round(float(player["expected_goals_for"]), 4),
                    "expected_goals_against": round(float(player["expected_goals_against"]), 4),
                    "clean_sheet_probability": round(float(player["clean_sheet_probability"]), 6),
                    "win_probability": round(float(player["win_probability"]), 6),
                    "draw_probability": round(float(player["draw_probability"]), 6),
                    "loss_probability": round(float(player["loss_probability"]), 6),
                },
                "model_context": {
                    "history_minutes": round(float(player["feature_player_history_minutes"]), 1) if pd.notna(player["feature_player_history_minutes"]) else None,
                    "attacking_history_weight": round(float(player["history_weight"]), 6),
                    "availability_probability_applied": round(float(player["availability_probability_applied"]), 6),
                },
            }
            fixture_projections[player_id].append(projection)
            gameweek = int(player["gameweek"])
            key = (player_id, gameweek)
            if key not in gameweek_samples:
                gameweek_samples[key] = {
                    "points": np.zeros(simulations, dtype=np.int16),
                    "minutes": np.zeros(simulations, dtype=np.int16),
                    "fixture_ids": [],
                }
            gameweek_samples[key]["points"] += total[:, index]
            gameweek_samples[key]["minutes"] += sampled_minutes[:, index]
            gameweek_samples[key]["fixture_ids"].append(int(fixture_id))
    quality = {
        "maximum_sampled_team_goal_mean_error": round(maximum_team_goal_error, 6),
        "maximum_sampled_player_minutes_mean_error": round(maximum_minutes_error, 6),
    }
    return dict(fixture_projections), gameweek_samples, quality


def gameweek_projection(sample: Mapping[str, object]) -> dict[str, object]:
    points = sample["points"]
    minutes = sample["minutes"]
    return {
        "fixture_ids": list(sample["fixture_ids"]),
        "xpts": round(float(points.mean()), 4),
        "xmins": round(float(minutes.mean()), 4),
        "outcome_probabilities": _point_summary(points),
    }
