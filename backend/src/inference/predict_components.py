"""Run all trained component models and save their future forecasts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .build_features import DEFAULT_CONFIG_PATH, load_settings, write_json
from .components import forecast_components, load_artifacts


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.10g")
    temporary.replace(path)


def _artifact_manifest(artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        name: {
            "model_type": artifact.get("model_type", name),
            "artifact_version": artifact.get("artifact_version"),
            "trained_at": artifact.get("trained_at"),
            "feature_profile": artifact.get("feature_profile"),
        }
        for name, artifact in artifacts.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    features_path = Path(args.features or str(settings["features_path"]))
    features = pd.read_csv(features_path, low_memory=False)
    artifacts = load_artifacts(settings["model_dirs"])
    predictions = forecast_components(features, artifacts)
    output_dir = Path(args.output_dir or str(settings["component_dir"]))
    for name, frame in predictions.items():
        _write_frame(output_dir / f"{name}.csv", frame)
        print(f"Wrote {len(frame):,} {name} rows")
    write_json(output_dir / "manifest.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "future_player_fixture_rows": int(len(features)),
        "future_fixtures": int(features[["season", "fixture_id"]].drop_duplicates().shape[0]),
        "models": _artifact_manifest(artifacts),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
