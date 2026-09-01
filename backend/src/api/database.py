"""Read-only access to the latest projection database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from flask import Flask, current_app, g


class DatabaseUnavailable(RuntimeError):
    """Raised when no completed inference batch is available."""


def get_database() -> sqlite3.Connection:
    if "database" not in g:
        path = Path(current_app.config["PREDICTION_DATABASE"])
        if not path.exists():
            raise DatabaseUnavailable(f"Projection database does not exist: {path}")
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        g.database = connection
    return g.database


def close_database(_: BaseException | None = None) -> None:
    connection = g.pop("database", None)
    if connection is not None:
        connection.close()


def load_json(value: str) -> object:
    return json.loads(value)


def load_metadata() -> dict[str, object]:
    rows = get_database().execute("SELECT key, value FROM metadata").fetchall()
    return {row["key"]: load_json(row["value"]) for row in rows}


def init_app(app: Flask) -> None:
    app.teardown_appcontext(close_database)
