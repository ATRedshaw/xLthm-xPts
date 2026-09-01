"""Projection queries and response shaping."""

from __future__ import annotations

from collections.abc import Iterable

from .database import get_database, load_json


def rounded(value: object, digits: int = 4) -> object:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


def player_identity(row: object, *, detail: str = "summary") -> dict[str, object]:
    player = {
        "id": row["id"],
        "name": row["name"],
        "position": row["position"],
        "team": row["team"],
        "price": row["price"],
        "selected_by": row["selected_by"],
    }
    if detail in {"standard", "full"}:
        player.update({
            "status": row["status"],
            "availability_probability": row["availability_probability"],
        })
    if detail == "full":
        player.update({
            "code": row["code"],
            "team_id": row["team_id"],
            "news": row["news"],
        })
    return player


def outcome_probabilities(value: str, *, include_distribution: bool) -> dict[str, object]:
    probabilities = load_json(value)
    if not include_distribution:
        probabilities.pop("points_distribution", None)
    return rounded(probabilities)


def load_gameweeks(
    player_ids: Iterable[int],
    *,
    start_gameweek: int,
    end_gameweek: int,
    detail: str,
    include_distribution: bool,
) -> dict[int, list[dict[str, object]]]:
    ids = list(player_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = get_database().execute(
        f"""
        SELECT player_id, gameweek, xpts, xmins, fixture_ids, outcome_probabilities
        FROM player_gameweek_projections
        WHERE player_id IN ({placeholders}) AND gameweek BETWEEN ? AND ?
        ORDER BY player_id, gameweek
        """,
        [*ids, start_gameweek, end_gameweek],
    ).fetchall()
    projections = {player_id: [] for player_id in ids}
    for row in rows:
        projection = {
            "gameweek": row["gameweek"],
            "xpts": round(row["xpts"], 3),
            "xmins": round(row["xmins"], 1),
        }
        if detail in {"standard", "full"}:
            projection["fixtures"] = load_json(row["fixture_ids"])
        if detail == "full" or include_distribution:
            projection["outcome_probabilities"] = outcome_probabilities(
                row["outcome_probabilities"],
                include_distribution=include_distribution,
            )
        projections[row["player_id"]].append(projection)
    return projections


def load_fixture_projections(
    player_ids: Iterable[int],
    *,
    start_gameweek: int,
    end_gameweek: int,
    detail: str,
    include_distribution: bool,
) -> dict[int, dict[int, list[dict[str, object]]]]:
    ids = list(player_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = get_database().execute(
        f"""
        SELECT projection.*, opponent.short_name AS opponent
        FROM player_fixture_projections AS projection
        JOIN teams AS opponent ON opponent.id = projection.opponent_team_id
        WHERE projection.player_id IN ({placeholders})
          AND projection.gameweek BETWEEN ? AND ?
        ORDER BY projection.player_id, projection.gameweek, projection.fixture_id
        """,
        [*ids, start_gameweek, end_gameweek],
    ).fetchall()
    projections: dict[int, dict[int, list[dict[str, object]]]] = {}
    for row in rows:
        projection = {
            "fixture": row["fixture_id"],
            "opponent": row["opponent"],
            "is_home": bool(row["is_home"]),
            "xpts": round(row["xpts"], 3),
            "xmins": round(row["xmins"], 1),
        }
        if detail in {"standard", "full"}:
            projection.update({
                "action_probabilities": rounded(load_json(row["action_probabilities"])),
                "xpts_breakdown": rounded(load_json(row["xpts_breakdown"]), 3),
            })
        if detail == "full":
            projection.update({
                "expected_actions": rounded(load_json(row["expected_actions"]), 3),
                "model_context": rounded(load_json(row["model_context"])),
            })
        if detail == "full" or include_distribution:
            projection["outcome_probabilities"] = outcome_probabilities(
                row["outcome_probabilities"],
                include_distribution=include_distribution,
            )
        projections.setdefault(row["player_id"], {}).setdefault(
            row["gameweek"], []
        ).append(projection)
    return projections


def attach_projections(
    rows: list[object],
    *,
    start_gameweek: int,
    end_gameweek: int,
    detail: str,
    include_distribution: bool,
) -> list[dict[str, object]]:
    player_ids = [row["id"] for row in rows]
    gameweeks = load_gameweeks(
        player_ids,
        start_gameweek=start_gameweek,
        end_gameweek=end_gameweek,
        detail=detail,
        include_distribution=include_distribution,
    )
    fixtures = load_fixture_projections(
        player_ids,
        start_gameweek=start_gameweek,
        end_gameweek=end_gameweek,
        detail=detail,
        include_distribution=include_distribution,
    )
    players = []
    for row in rows:
        player = player_identity(row, detail=detail)
        future_points = gameweeks.get(row["id"], [])
        for projection in future_points:
            fixture_details = fixtures.get(row["id"], {}).get(projection["gameweek"])
            if fixture_details:
                projection.pop("fixtures", None)
                projection["fixture_projections"] = fixture_details
        player["future_points"] = future_points
        player["total_xpts"] = round(sum(item["xpts"] for item in future_points), 3)
        players.append(player)
    return players
