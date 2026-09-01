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
        "defaults": {
            "players": "all players",
            "gameweeks": "all available gameweeks from start_gameweek",
            "detail": "summary",
            "include_distribution": False,
            "player_order": "FPL player ID ascending",
        },
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "Check whether the projection database is available.",
                "parameters": [],
            },
            "/api/v1/meta": {
                "method": "GET",
                "description": "Return inference batch, model, coverage and quality metadata.",
                "parameters": [],
            },
            "/api/v1/players": {
                "method": "GET",
                "description": "Return player projections ordered by FPL player ID.",
                "parameters": [
                    "start_gameweek", "gameweeks", "detail", "include_distribution",
                    "position", "team", "search", "limit", "offset",
                ],
            },
            "/api/v1/players/<id>": {
                "method": "GET",
                "description": "Return projections for one FPL player ID.",
                "parameters": [
                    "start_gameweek", "gameweeks", "detail", "include_distribution",
                ],
            },
            "/api/v1/fixtures": {
                "method": "GET",
                "description": "Return team forecasts for all requested fixtures.",
                "parameters": ["start_gameweek", "gameweeks"],
            },
            "/api/v1/fixtures/<id>": {
                "method": "GET",
                "description": "Return one FPL fixture and its player projections.",
                "parameters": ["detail", "include_distribution"],
            },
        },
        "parameters": {
            "start_gameweek": {
                "type": "integer",
                "default": "first available gameweek",
                "description": "First gameweek in the requested projection window.",
            },
            "gameweeks": {
                "type": "integer",
                "default": "all available gameweeks from start_gameweek",
                "description": "Number of consecutive gameweeks to return.",
            },
            "detail": {
                "type": "string",
                "default": "summary",
                "allowed": ["summary", "standard", "full"],
                "options": {
                    "summary": "Player identity, price, ownership, xPts, xMins and compact fixtures.",
                    "standard": "Summary plus availability, action probabilities and xPts breakdowns.",
                    "full": "Standard plus news, expected actions and outcome probabilities.",
                },
            },
            "include_distribution": {
                "type": "boolean",
                "default": False,
                "accepted": ["true", "false", "1", "0", "yes", "no"],
                "description": "Add outcome probabilities, percentiles and the full points distribution.",
            },
            "position": {
                "type": "string",
                "allowed": ["GK", "DEF", "MID", "FWD"],
                "description": "Filter the player list by position.",
            },
            "team": {
                "type": "string",
                "description": "Filter by case-insensitive FPL team short name, such as MCI.",
            },
            "search": {
                "type": "string",
                "description": "Filter by a case-insensitive partial player name.",
            },
            "limit": {
                "type": "integer",
                "default": 9999,
                "minimum": 1,
                "maximum": 9999,
                "description": "Maximum players returned; the default covers the full player list.",
            },
            "offset": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Number of matching players to skip after filtering and FPL-ID ordering; this is a row count, not a player ID.",
            },
        },
        "outcome_probabilities": {
            "thresholds": {
                "negative_points": "Probability of scoring fewer than zero points.",
                "zero_points": "Probability of scoring exactly zero points.",
                "two_plus_points": "Probability of scoring at least two points.",
                "five_plus_points": "Probability of scoring at least five points.",
                "ten_plus_points": "Probability of scoring at least ten points.",
            },
            "percentiles": {
                "description": "Point cut-offs from the simulated outcomes; they are not probabilities.",
                "p10": "10th-percentile downside: about 10% of simulations scored at or below this value.",
                "p25": "Lower quartile: about 25% of simulations scored at or below this value.",
                "p50": "Median: half of simulations scored at or below and half at or above this value.",
                "p75": "Upper quartile: about 75% of simulations scored at or below this value.",
                "p90": "90th-percentile upside: about 90% of simulations scored at or below this value.",
            },
            "points_distribution": "Probability attached to every simulated integer FPL points total.",
        },
        "examples": [
            "/api/v1/players",
            "/api/v1/players?gameweeks=5&position=FWD",
            "/api/v1/players?limit=100&offset=100",
            "/api/v1/players/411?detail=standard",
            "/api/v1/players/411?gameweeks=1&include_distribution=true",
            "/api/v1/fixtures",
        ],
    })
