"""Service status and inference-batch metadata endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify

from ..database import DatabaseUnavailable, get_database, load_metadata


blueprint = Blueprint("system", __name__)


@blueprint.get("/health")
def health():
    try:
        get_database().execute("SELECT 1").fetchone()
    except (DatabaseUnavailable, OSError):
        return jsonify({"status": "unavailable"}), 503
    return jsonify({"status": "ok"})


@blueprint.get("/api/v1/meta")
def metadata():
    return jsonify(load_metadata())


@blueprint.get("/api/v1")
def index():
    return jsonify({
        "name": "xPts projections",
        "version": "v1",
        "endpoints": {
            "players": "/api/v1/players",
            "player": "/api/v1/players/<id>",
            "fixtures": "/api/v1/fixtures",
            "fixture": "/api/v1/fixtures/<id>",
            "metadata": "/api/v1/meta",
        },
        "projection_parameters": {
            "start_gameweek": "First gameweek to return",
            "gameweeks": "Number of gameweeks to return; defaults to 5",
            "detail": "summary, standard or full",
            "include_distribution": "Include simulated point distributions; defaults to false",
        },
        "player_filters": {
            "position": "GK, DEF, MID or FWD",
            "team": "FPL team short name",
            "search": "Partial player name",
            "limit": "Page size; defaults to 100",
            "offset": "Number of players to skip",
        },
    })
