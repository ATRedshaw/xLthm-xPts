"""Run joint fixture simulations and atomically replace the projection database."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .build_features import DEFAULT_CONFIG_PATH, load_settings
from .simulation import merge_component_frames, simulate_all_fixtures
from .storage import write_projection_database


COMPONENT_NAMES = [
    "team", "fixtures", "scorelines", "minutes", "attacking",
    "defensive", "misc_events", "bps",
]


def _load_frames(component_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in COMPONENT_NAMES:
        path = component_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Component forecast does not exist: {path}")
        frames[name] = pd.read_csv(path, low_memory=False)
    return frames


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--features", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--component-dir", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--simulations", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    features = pd.read_csv(Path(args.features or str(settings["features_path"])), low_memory=False)
    context_path = Path(args.context or str(settings["live_context_path"]))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    component_dir = Path(args.component_dir or str(settings["component_dir"]))
    components = _load_frames(component_dir)
    manifest = json.loads((component_dir / "manifest.json").read_text(encoding="utf-8"))
    player_components = merge_component_frames(features, components)
    simulation_count = int(args.simulations or settings["simulation_count"])
    fixture_projections, gameweek_samples, quality = simulate_all_fixtures(
        player_components,
        components["fixtures"],
        components["scorelines"],
        rules=settings["rules"],
        simulations=simulation_count,
        random_state=int(settings["random_state"]),
    )
    gameweeks = sorted(
        int(value) for value in pd.to_numeric(features["gameweek"], errors="coerce").dropna().unique()
    )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_retrieved_at": context["retrieved_at"],
        "season": context["season"],
        "ruleset": settings["rules"]["version"],
        "simulation_count": simulation_count,
        "random_state": int(settings["random_state"]),
        "models": manifest["models"],
        "coverage": {
            "players": len(context["players"]),
            "fixtures": len(context["fixtures"]),
            "gameweeks": gameweeks,
            "skipped_fixtures": context["skipped_fixtures"],
        },
        "quality_checks": quality,
        "methodology": {
            "team_scores": "Sampled from the team-score probability matrix.",
            "minutes": "Sampled from coherent player minutes states after applying official next-round availability.",
            "goals_and_assists": "Team goals are allocated to appearing players using reconciled attacking rates; assists are sampled and allocated separately.",
            "other_events": "Saves, penalty saves, defensive contributions, cards and penalty misses are sampled from exposure-adjusted component rates.",
            "bonus": "Conditional BPS and position residuals are sampled for the same appearance outcomes, then fixture 3/2/1 tie rules are applied.",
        },
        "limitations": [
            "Event timing is not modelled, so clean sheets and goals conceded use full-match scorelines with minutes exposure approximations.",
            "Player appearances are sampled independently and do not enforce an exact eleven-player starting lineup.",
            "BPS residuals are not yet conditioned on the specific goals, assists and defensive actions sampled in the same run.",
        ],
    }
    database_path = write_projection_database(
        args.database or str(settings["database_path"]),
        context=context,
        metadata=metadata,
        fixture_forecasts=components["fixtures"],
        fixture_projections=fixture_projections,
        gameweek_samples=gameweek_samples,
        gameweeks=gameweeks,
    )
    print(
        f"Wrote xPts projections for {len(context['players']):,} players "
        f"to {database_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
