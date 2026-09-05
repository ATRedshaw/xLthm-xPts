import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from api import create_app
from inference.storage import SCHEMA


class ProjectionWindowTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "projections.sqlite"
        with sqlite3.connect(database_path) as connection:
            connection.executescript(SCHEMA)
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("season", json.dumps("2026-27")),
                    ("generated_at", json.dumps("2026-09-05T12:00:00+00:00")),
                ],
            )
            connection.executemany(
                "INSERT INTO teams (id, code, name, short_name) VALUES (?, ?, ?, ?)",
                [(1, 1, "Home", "HOM"), (2, 2, "Away", "AWY")],
            )
            connection.execute(
                """
                INSERT INTO players (
                    id, code, name, position, team_id, price, selected_by,
                    status, availability_probability, news
                ) VALUES (1, 1, 'Player', 'MID', 1, 5.0, 1.0, 'a', 1.0, '')
                """
            )
            connection.execute(
                """
                INSERT INTO fixtures (
                    id, gameweek, kickoff_time, home_team_id, away_team_id,
                    home_expected_goals, away_expected_goals,
                    home_win_probability, draw_probability, away_win_probability,
                    home_clean_sheet_probability, away_clean_sheet_probability
                ) VALUES (100, 38, '2027-05-23T15:00:00+00:00', 1, 2, 1.5, 1.0, 0.5, 0.3, 0.2, 0.4, 0.3)
                """
            )
            connection.executemany(
                """
                INSERT INTO player_gameweek_projections (
                    player_id, gameweek, xpts, xmins, fixture_ids,
                    outcome_probabilities
                ) VALUES (1, ?, 0, 0, '[]', '{}')
                """,
                [(36,), (37,), (38,)],
            )
            connection.execute(
                """
                INSERT INTO player_fixture_projections (
                    player_id, fixture_id, gameweek, opponent_team_id, is_home,
                    xpts, xmins, action_probabilities, expected_actions,
                    xpts_breakdown, outcome_probabilities, model_context
                ) VALUES (1, 100, 38, 2, 1, 5.0, 90.0, '{}', '{}', '{}', '{}', '{}')
                """
            )
        self.client = create_app({
            "TESTING": True,
            "PREDICTION_DATABASE": str(database_path),
        }).test_client()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_window_is_shortened_to_remaining_gameweeks(self):
        response = self.client.get(
            "/api/v1/fixtures?start_gameweek=38&gameweeks=5"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["meta"]["start_gameweek"], 38)
        self.assertEqual(payload["meta"]["end_gameweek"], 38)
        self.assertEqual([fixture["fixture"] for fixture in payload["fixtures"]], [100])

    def test_fixtures_include_full_and_short_team_names(self):
        response = self.client.get(
            "/api/v1/fixtures?start_gameweek=38&gameweeks=1"
        )

        self.assertEqual(response.status_code, 200)
        fixture = response.get_json()["fixtures"][0]
        self.assertEqual(fixture["home_team"], "HOM")
        self.assertEqual(fixture["away_team"], "AWY")
        self.assertEqual(fixture["home_team_name"], "Home")
        self.assertEqual(fixture["away_team_name"], "Away")

    def test_players_include_full_team_and_opponent_names(self):
        response = self.client.get(
            "/api/v1/players?start_gameweek=38&gameweeks=1"
        )

        self.assertEqual(response.status_code, 200)
        player = response.get_json()["players"][0]
        self.assertEqual(player["team"], "HOM")
        self.assertEqual(player["team_name"], "Home")
        fixture = player["future_points"][0]["fixture_projections"][0]
        self.assertEqual(fixture["opponent"], "AWY")
        self.assertEqual(fixture["opponent_name"], "Away")

    def test_start_gameweek_outside_projection_range_is_rejected(self):
        response = self.client.get(
            "/api/v1/fixtures?start_gameweek=39&gameweeks=5"
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "start_gameweek must be between 36 and 38"},
        )

    def test_gameweek_count_must_still_be_positive(self):
        response = self.client.get(
            "/api/v1/fixtures?start_gameweek=38&gameweeks=0"
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "gameweeks must be at least 1"},
        )


if __name__ == "__main__":
    unittest.main()
