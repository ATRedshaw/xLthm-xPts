"""Run the development API server."""

from __future__ import annotations

import os

from . import create_app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("XPTS_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("XPTS_API_PORT", "5000")),
    )
