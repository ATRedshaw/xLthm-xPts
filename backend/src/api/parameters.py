"""Shared query-parameter parsing for projection endpoints."""

from __future__ import annotations

from flask import request
from werkzeug.exceptions import BadRequest

from .database import get_database


DETAIL_LEVELS = {"summary", "standard", "full"}


def integer_argument(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = request.args.get(name)
    try:
        value = default if raw_value is None else int(raw_value)
    except ValueError as error:
        raise BadRequest(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise BadRequest(f"{name} must be between {minimum} and {maximum}")
    return value


def boolean_argument(name: str, *, default: bool = False) -> bool:
    raw_value = request.args.get(name)
    if raw_value is None:
        return default
    normalised = raw_value.lower()
    if normalised in {"1", "true", "yes"}:
        return True
    if normalised in {"0", "false", "no"}:
        return False
    raise BadRequest(f"{name} must be true or false")


def detail_argument() -> str:
    detail = request.args.get("detail", "summary").lower()
    if detail not in DETAIL_LEVELS:
        raise BadRequest("detail must be summary, standard or full")
    return detail


def projection_window() -> tuple[int, int, int]:
    bounds = get_database().execute(
        "SELECT MIN(gameweek) AS first_gameweek, MAX(gameweek) AS last_gameweek "
        "FROM player_gameweek_projections"
    ).fetchone()
    first_gameweek = int(bounds["first_gameweek"])
    last_gameweek = int(bounds["last_gameweek"])
    start_gameweek = integer_argument(
        "start_gameweek",
        default=first_gameweek,
        minimum=first_gameweek,
        maximum=last_gameweek,
    )
    remaining_gameweeks = last_gameweek - start_gameweek + 1
    gameweeks = integer_argument(
        "gameweeks",
        default=min(5, remaining_gameweeks),
        minimum=1,
        maximum=remaining_gameweeks,
    )
    return start_gameweek, start_gameweek + gameweeks - 1, gameweeks
