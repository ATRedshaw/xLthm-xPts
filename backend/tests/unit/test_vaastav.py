from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from preprocessing.vaastav import SeasonData, normalise_season


def season_data(player_rows: list[dict[str, object]]) -> SeasonData:
    return SeasonData(
        season="2025-26",
        player_fixtures=pd.DataFrame(player_rows),
        players=pd.DataFrame(
            [{"id": 7, "code": 7007}, {"id": 8, "code": 8008}]
        ),
        teams=pd.DataFrame(
            [
                {"id": 1, "code": 101, "name": "Home"},
                {"id": 2, "code": 202, "name": "Away"},
            ]
        ),
        fixtures=pd.DataFrame(
            [
                {
                    "id": 11,
                    "event": 3,
                    "kickoff_time": "2025-09-01T14:00:00Z",
                    "team_h": 1,
                    "team_a": 2,
                    "team_h_score": 2,
                    "team_a_score": 1,
                }
            ]
        ),
    )


def player_row(**changes: object) -> dict[str, object]:
    row = {
        "element": 7,
        "fixture": 11,
        "GW": 3,
        "kickoff_time": "2025-09-01T14:00:00Z",
        "minutes": 90,
        "opponent_team": 2,
        "position": "MID",
        "was_home": True,
        "team_h_score": 99,
        "team_a_score": 99,
        "name": "Test Player",
        "value": 75,
        "selected": 1000,
        "transfers_in": 50,
        "transfers_out": 20,
        "expected_goals": 0.4,
    }
    row.update(changes)
    return row


class NormaliseSeasonTests(unittest.TestCase):
    def test_uses_fixture_context_and_stable_codes(self) -> None:
        rows = normalise_season(season_data([player_row()]))

        self.assertEqual(rows.loc[0, "player_key"], "code:7007")
        self.assertEqual(rows.loc[0, "team_key"], "code:101")
        self.assertEqual(rows.loc[0, "opponent_team_key"], "code:202")
        self.assertEqual(rows.loc[0, "gameweek"], 3)
        self.assertEqual(rows.loc[0, "price_tenths"], 75)
        self.assertEqual(rows.loc[0, "home_goals"], 2)
        self.assertEqual(rows.loc[0, "away_goals"], 1)
        self.assertTrue(pd.isna(rows.loc[0, "defensive_contribution"]))

    def test_removes_exact_source_duplicates(self) -> None:
        row = player_row()
        rows = normalise_season(season_data([row, row.copy()]))

        self.assertEqual(len(rows), 1)

    def test_excludes_assistant_manager_chip_elements(self) -> None:
        rows = normalise_season(
            season_data([player_row(), player_row(element=8, position="AM")])
        )

        self.assertEqual(rows["player_id"].tolist(), [7])

    def test_rejects_conflicting_player_fixture_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting rows"):
            normalise_season(
                season_data([player_row(), player_row(minutes=45)])
            )


if __name__ == "__main__":
    unittest.main()
