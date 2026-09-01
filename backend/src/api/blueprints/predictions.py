"""Player and fixture prediction endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest, NotFound

from ..database import get_database, load_metadata
from ..parameters import (
    boolean_argument,
    detail_argument,
    integer_argument,
    projection_window,
)
from ..projections import attach_projections, load_fixture_projections, player_identity


blueprint = Blueprint("predictions", __name__, url_prefix="/api/v1")


PLAYER_COLUMNS = """
    player.id, player.code, player.name, player.position, player.team_id,
    player.price, player.selected_by, player.status,
    player.availability_probability, player.news, team.short_name AS team
"""


def response_meta(
    *,
    start_gameweek: int,
    end_gameweek: int,
    detail: str,
    count: int,
) -> dict[str, object]:
    metadata = load_metadata()
    return {
        "season": metadata["season"],
        "generated_at": metadata["generated_at"],
        "start_gameweek": start_gameweek,
        "end_gameweek": end_gameweek,
        "detail": detail,
        "count": count,
    }


@blueprint.get("/players")
def players():
    start_gameweek, end_gameweek, _ = projection_window()
    detail = detail_argument()
    include_distribution = boolean_argument("include_distribution")
    limit = integer_argument("limit", default=9999, minimum=1, maximum=9999)
    offset = integer_argument("offset", default=0, minimum=0, maximum=100000)
    position = request.args.get("position")
    if position:
        position = position.upper()
        if position not in {"GK", "DEF", "MID", "FWD"}:
            raise BadRequest("position must be GK, DEF, MID or FWD")
    filters = []
    values: list[object] = []
    if position:
        filters.append("player.position = ?")
        values.append(position)
    if team := request.args.get("team"):
        filters.append("LOWER(team.short_name) = LOWER(?)")
        values.append(team)
    if search := request.args.get("search"):
        filters.append("LOWER(player.name) LIKE ?")
        values.append(f"%{search.lower()}%")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = get_database().execute(
        f"""
        SELECT {PLAYER_COLUMNS}
        FROM players AS player
        JOIN teams AS team ON team.id = player.team_id
        {where}
        ORDER BY player.id
        LIMIT ? OFFSET ?
        """,
        [*values, limit, offset],
    ).fetchall()
    result = attach_projections(
        rows,
        start_gameweek=start_gameweek,
        end_gameweek=end_gameweek,
        detail=detail,
        include_distribution=include_distribution,
    )
    metadata = response_meta(
        start_gameweek=start_gameweek,
        end_gameweek=end_gameweek,
        detail=detail,
        count=len(result),
    )
    metadata.update({"limit": limit, "offset": offset})
    return jsonify({
        "meta": metadata,
        "players": result,
    })


@blueprint.get("/players/<int:player_id>")
def player(player_id: int):
    start_gameweek, end_gameweek, _ = projection_window()
    detail = detail_argument()
    include_distribution = boolean_argument("include_distribution")
    row = get_database().execute(
        f"""
        SELECT {PLAYER_COLUMNS}
        FROM players AS player
        JOIN teams AS team ON team.id = player.team_id
        WHERE player.id = ?
        """,
        (player_id,),
    ).fetchone()
    if row is None:
        raise NotFound(f"Player {player_id} does not exist")
    result = attach_projections(
        [row],
        start_gameweek=start_gameweek,
        end_gameweek=end_gameweek,
        detail=detail,
        include_distribution=include_distribution,
    )[0]
    return jsonify(result)


def fixture_response(row: object) -> dict[str, object]:
    return {
        "fixture": row["id"],
        "gameweek": row["gameweek"],
        "kickoff_time": row["kickoff_time"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "forecast": {
            "expected_goals": {
                "home": round(row["home_expected_goals"], 3),
                "away": round(row["away_expected_goals"], 3),
            },
            "result_probabilities": {
                "home_win": round(row["home_win_probability"], 4),
                "draw": round(row["draw_probability"], 4),
                "away_win": round(row["away_win_probability"], 4),
            },
            "clean_sheet_probabilities": {
                "home": round(row["home_clean_sheet_probability"], 4),
                "away": round(row["away_clean_sheet_probability"], 4),
            },
        },
    }


FIXTURE_QUERY = """
    SELECT fixture.*, home.short_name AS home_team, away.short_name AS away_team
    FROM fixtures AS fixture
    JOIN teams AS home ON home.id = fixture.home_team_id
    JOIN teams AS away ON away.id = fixture.away_team_id
"""


@blueprint.get("/fixtures")
def fixtures():
    start_gameweek, end_gameweek, _ = projection_window()
    rows = get_database().execute(
        f"{FIXTURE_QUERY} WHERE fixture.gameweek BETWEEN ? AND ? "
        "ORDER BY fixture.gameweek, fixture.kickoff_time, fixture.id",
        (start_gameweek, end_gameweek),
    ).fetchall()
    result = [fixture_response(row) for row in rows]
    return jsonify({
        "meta": response_meta(
            start_gameweek=start_gameweek,
            end_gameweek=end_gameweek,
            detail="fixture",
            count=len(result),
        ),
        "fixtures": result,
    })


@blueprint.get("/fixtures/<int:fixture_id>")
def fixture(fixture_id: int):
    detail = detail_argument()
    include_distribution = boolean_argument("include_distribution")
    row = get_database().execute(
        f"{FIXTURE_QUERY} WHERE fixture.id = ?",
        (fixture_id,),
    ).fetchone()
    if row is None:
        raise NotFound(f"Fixture {fixture_id} does not exist")
    result = fixture_response(row)
    player_rows = get_database().execute(
        f"""
        SELECT {PLAYER_COLUMNS}
        FROM players AS player
        JOIN teams AS team ON team.id = player.team_id
        JOIN player_fixture_projections AS projection
          ON projection.player_id = player.id
        WHERE projection.fixture_id = ?
        ORDER BY player.id
        """,
        (fixture_id,),
    ).fetchall()
    projection_rows = load_fixture_projections(
        [player["id"] for player in player_rows],
        start_gameweek=row["gameweek"],
        end_gameweek=row["gameweek"],
        detail=detail,
        include_distribution=include_distribution,
    )
    players_result = []
    for player_row in player_rows:
        player_result = player_identity(player_row, detail=detail)
        projections = projection_rows[player_row["id"]][row["gameweek"]]
        player_result["projection"] = next(
            item for item in projections if item["fixture"] == fixture_id
        )
        players_result.append(player_result)
    result["players"] = players_result
    return jsonify(result)
