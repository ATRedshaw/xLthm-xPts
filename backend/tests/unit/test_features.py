from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from preprocessing.features import build_training_dataset
from preprocessing.vaastav import OUTCOME_COLUMNS


def row(
    *,
    season: str,
    fixture: int,
    kickoff: str,
    player: int,
    team: int,
    opponent: int,
    is_home: int,
    minutes: int,
    goals: int,
    expected_goals: float,
    defensive_contribution: float | None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "season": season,
        "season_start": int(season[:4]),
        "fixture_id": fixture,
        "gameweek": fixture,
        "kickoff_time": kickoff,
        "player_id": player,
        "player_code": player * 100,
        "player_key": f"code:{player * 100}",
        "player_name": f"Player {player}",
        "position": "MID" if player == 1 else "DEF",
        "team_id": team,
        "team_code": team * 1000,
        "team_key": f"code:{team * 1000}",
        "team_name": f"Team {team}",
        "opponent_team_id": opponent,
        "opponent_team_code": opponent * 1000,
        "opponent_team_key": f"code:{opponent * 1000}",
        "opponent_team_name": f"Team {opponent}",
        "is_home": is_home,
        "home_goals": 2,
        "away_goals": 1,
        "fpl_expected_points": 3.0,
        "price_tenths": 70 + fixture,
        "selected": 1000,
        "transfers_balance": 10,
        "transfers_in": 20,
        "transfers_out": 10,
    }
    values.update({column: 0 for column in OUTCOME_COLUMNS})
    values.update(
        {
            "minutes": minutes,
            "starts": int(minutes > 0),
            "goals_scored": goals,
            "expected_goals": expected_goals,
            "expected_assists": 0.1,
            "expected_goal_involvements": expected_goals + 0.1,
            "defensive_contribution": defensive_contribution,
            "clearances_blocks_interceptions": defensive_contribution,
            "recoveries": defensive_contribution,
            "tackles": defensive_contribution,
            "total_points": goals * 5 + int(minutes > 0),
        }
    )
    return values


def source_rows() -> pd.DataFrame:
    rows = []
    for season, fixture, kickoff in (
        ("2024-25", 1, "2024-08-10T14:00:00Z"),
        ("2025-26", 2, "2025-08-10T14:00:00Z"),
        ("2025-26", 3, "2025-08-17T14:00:00Z"),
    ):
        rows.extend(
            [
                row(
                    season=season,
                    fixture=fixture,
                    kickoff=kickoff,
                    player=1,
                    team=1,
                    opponent=2,
                    is_home=1,
                    minutes=90,
                    goals=fixture - 1,
                    expected_goals=0.2 * fixture,
                    defensive_contribution=None if fixture == 1 else 8,
                ),
                row(
                    season=season,
                    fixture=fixture,
                    kickoff=kickoff,
                    player=2,
                    team=2,
                    opponent=1,
                    is_home=0,
                    minutes=90,
                    goals=0,
                    expected_goals=0.1,
                    defensive_contribution=None if fixture == 1 else 10,
                ),
            ]
        )
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_current_outcome_cannot_change_current_features(self) -> None:
        source = source_rows()
        changed = source.copy()
        current = (changed["fixture_id"] == 2) & (changed["player_id"] == 1)
        changed.loc[current, ["goals_scored", "expected_goals", "total_points"]] = [
            9,
            5.0,
            50,
        ]

        first = build_training_dataset(source)
        second = build_training_dataset(changed)
        row_key = (first["fixture_id"] == 2) & (first["player_id"] == 1)
        feature_columns = [column for column in first if column.startswith("feature_")]

        pd.testing.assert_series_equal(
            first.loc[row_key, feature_columns].iloc[0],
            second.loc[row_key, feature_columns].iloc[0],
        )
        self.assertNotEqual(
            first.loc[row_key, "target_expected_goals"].iloc[0],
            second.loc[row_key, "target_expected_goals"].iloc[0],
        )

    def test_carries_previous_season_and_preserves_missing_defensive_data(self) -> None:
        dataset = build_training_dataset(source_rows())
        current = dataset[(dataset["season"] == "2025-26") & (dataset["player_id"] == 1)].iloc[0]

        self.assertEqual(current["feature_player_previous_season_minutes"], 90)
        self.assertEqual(current["feature_player_minutes_sum_3"], 90)
        self.assertTrue(pd.isna(current["feature_price_change_previous_snapshot"]))
        self.assertTrue(
            pd.isna(current["feature_player_defensive_contribution_sum_3"])
        )
        self.assertEqual(current["target_defensive_contribution_available"], 1)

    def test_uses_each_gameweeks_price_snapshot(self) -> None:
        dataset = build_training_dataset(source_rows())
        later = dataset[
            (dataset["fixture_id"] == 3) & (dataset["player_id"] == 1)
        ].iloc[0]

        self.assertEqual(later["feature_price_tenths"], 73)
        self.assertEqual(later["feature_price_millions"], 7.3)
        self.assertEqual(later["feature_price_change_previous_snapshot"], 1)
        self.assertEqual(later["feature_price_change_season"], 1)

    def test_all_outcomes_are_explicit_targets(self) -> None:
        dataset = build_training_dataset(source_rows())

        self.assertTrue(all(f"target_{column}" in dataset for column in OUTCOME_COLUMNS))
        self.assertFalse(any(column in dataset for column in OUTCOME_COLUMNS))


if __name__ == "__main__":
    unittest.main()
