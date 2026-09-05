from __future__ import annotations

import unittest
from datetime import datetime, timezone

from inference.features import normalise_live_future_rows


def bootstrap(*, current_gameweek: int | None = None) -> dict[str, object]:
    return {
        "teams": [
            {"id": 1, "code": 10, "name": "Home", "short_name": "HOM"},
            {"id": 2, "code": 20, "name": "Away", "short_name": "AWY"},
        ],
        "element_types": [{"id": 1, "singular_name_short": "GKP"}],
        "events": [
            {"id": gameweek, "is_current": gameweek == current_gameweek}
            for gameweek in (4, 5, 6)
        ],
        "elements": [
            {
                "id": 1,
                "code": 100,
                "web_name": "Home Player",
                "first_name": "Home",
                "second_name": "Player",
                "element_type": 1,
                "team": 1,
                "status": "a",
                "chance_of_playing_next_round": None,
                "news": "",
                "news_added": None,
                "now_cost": 50,
                "selected_by_percent": "10.0",
                "transfers_in_event": 0,
                "transfers_out_event": 0,
                "ep_next": "4.0",
            },
            {
                "id": 2,
                "code": 200,
                "web_name": "Away Player",
                "first_name": "Away",
                "second_name": "Player",
                "element_type": 1,
                "team": 2,
                "status": "a",
                "chance_of_playing_next_round": None,
                "news": "",
                "news_added": None,
                "now_cost": 45,
                "selected_by_percent": "5.0",
                "transfers_in_event": 0,
                "transfers_out_event": 0,
                "ep_next": "3.0",
            },
        ],
        "total_players": 1_000_000,
    }


def fixture(
    fixture_id: int,
    gameweek: int | None,
    *,
    started: bool = False,
    finished: bool = False,
    kickoff_time: str | None = "2026-09-20T14:00:00Z",
) -> dict[str, object]:
    return {
        "id": fixture_id,
        "event": gameweek,
        "team_h": 1,
        "team_a": 2,
        "started": started,
        "finished": finished,
        "kickoff_time": kickoff_time,
    }


class NormaliseLiveFutureRowsTests(unittest.TestCase):
    retrieved_at = datetime(2026, 9, 5, tzinfo=timezone.utc)

    def normalise(
        self,
        fixtures: list[dict[str, object]],
        *,
        current_gameweek: int | None = None,
    ):
        return normalise_live_future_rows(
            bootstrap(current_gameweek=current_gameweek),
            fixtures,
            season="2026-27",
            retrieved_at=self.retrieved_at,
        )

    def test_excludes_every_fixture_in_the_official_current_gameweek(self):
        future, context = self.normalise(
            [fixture(1, 4), fixture(2, 4), fixture(3, 5)],
            current_gameweek=4,
        )

        self.assertEqual(set(future["fixture_id"]), {3})
        self.assertEqual([item["id"] for item in context["fixtures"]], [3])

    def test_started_fixture_excludes_its_whole_gameweek_as_a_fallback(self):
        future, context = self.normalise(
            [fixture(1, 4, started=True), fixture(2, 4), fixture(3, 5)]
        )

        self.assertEqual(set(future["fixture_id"]), {3})
        self.assertEqual([item["id"] for item in context["fixtures"]], [3])

    def test_keeps_all_fixtures_when_no_gameweek_is_active(self):
        future, context = self.normalise([fixture(1, 5), fixture(2, 6)])

        self.assertEqual(set(future["fixture_id"]), {1, 2})
        self.assertEqual([item["id"] for item in context["fixtures"]], [1, 2])

    def test_still_reports_unassigned_kickoff_times(self):
        future, context = self.normalise(
            [fixture(1, 5, kickoff_time=None), fixture(2, 6)]
        )

        self.assertEqual(set(future["fixture_id"]), {2})
        self.assertEqual(
            context["skipped_fixtures"],
            [{"fixture_id": 1, "reason": "kickoff_time is not assigned"}],
        )


if __name__ == "__main__":
    unittest.main()
