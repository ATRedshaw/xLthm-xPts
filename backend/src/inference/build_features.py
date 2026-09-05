"""Fetch live inputs and save future player-fixture feature rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from preprocessing.fpl import DEFAULT_BASE_URL as FPL_BASE_URL
from preprocessing.fpl import fetch_live_fpl, load_current_season
from preprocessing.vaastav import DEFAULT_BASE_URL as VAASTAV_BASE_URL
from preprocessing.vaastav import load_seasons

from .features import build_future_feature_rows, normalise_live_future_rows


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "inference.yaml"


def load_settings(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = config.get("inference") if isinstance(config, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must contain an inference mapping: {path}")
    return settings


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--context-output", default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    timeout = float(settings.get("timeout_seconds", 30))
    fpl_base_url = str(settings.get("fpl_base_url", FPL_BASE_URL))
    fpl_attempts = int(settings.get("fpl_request_attempts", 3))
    fpl_workers = int(settings.get("fpl_request_workers", 8))
    live = fetch_live_fpl(
        base_url=fpl_base_url,
        timeout=timeout,
        attempts=fpl_attempts,
    )
    season = str(settings["current_season"])
    future, context = normalise_live_future_rows(live.bootstrap, live.fixtures, season=season)
    seasons = [str(value) for value in settings["historical_seasons"]]
    archived_seasons = [value for value in seasons if value != season]
    historical_frames = []
    if archived_seasons:
        historical_frames.append(
            load_seasons(
                archived_seasons,
                base_url=str(settings.get("vaastav_base_url", VAASTAV_BASE_URL)),
                timeout=timeout,
                progress=print,
            )
        )
    current_history = load_current_season(
        live,
        season=season,
        base_url=fpl_base_url,
        timeout=timeout,
        attempts=fpl_attempts,
        workers=fpl_workers,
        progress=print,
    )
    historical_frames.append(current_history)
    historical = pd.concat(historical_frames, ignore_index=True).sort_values(
        ["kickoff_time", "season", "fixture_id", "player_id"],
        kind="stable",
    )
    features = build_future_feature_rows(historical, future)
    output_path = Path(args.output or str(settings["features_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    features.to_csv(temporary, index=False, date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.10g")
    temporary.replace(output_path)
    context_path = Path(args.context_output or str(settings["live_context_path"]))
    context["historical_seasons"] = seasons
    context["historical_sources"] = {
        value: "official_fpl" if value == season else "vaastav" for value in seasons
    }
    context["current_season_historical_rows"] = int(len(current_history))
    context["future_player_fixture_rows"] = int(len(features))
    context["future_fixtures"] = int(features[["season", "fixture_id"]].drop_duplicates().shape[0])
    write_json(context_path, context)
    print(f"Wrote {len(features):,} future player-fixture feature rows to {output_path}")
    print(f"Wrote normalised live context to {context_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
