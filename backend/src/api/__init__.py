"""Flask application factory for xPts projections."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from . import database
from .blueprints.predictions import blueprint as predictions_blueprint
from .blueprints.system import blueprint as system_blueprint


DEFAULT_DATABASE = Path(__file__).resolve().parents[3] / "data" / "inference" / "xpts.sqlite"


def create_app(config: dict[str, object] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        PREDICTION_DATABASE=os.environ.get("XPTS_DATABASE_PATH", str(DEFAULT_DATABASE)),
    )
    app.json.sort_keys = False
    if config:
        app.config.update(config)

    database.init_app(app)
    app.register_blueprint(system_blueprint)
    app.register_blueprint(predictions_blueprint)

    @app.errorhandler(database.DatabaseUnavailable)
    def database_unavailable(error: database.DatabaseUnavailable):
        return jsonify({"error": str(error)}), 503

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        return jsonify({"error": error.description}), error.code

    return app
