"""Build the shared component-model training CSV from Vaastav FPL data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import yaml

from .features import build_training_dataset
from .vaastav import DEFAULT_BASE_URL, load_seasons


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "preprocessing.yaml"
DEFAULT_OUTPUT_PATH = Path("data/processed/model_training.csv")
DEFAULT_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26", "2026-27")


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    rows: int
    columns: int
    features: int
    targets: int
    players: int
    fixtures: int
    xg_target_rows: int
    defensive_target_rows: int


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must contain a YAML mapping: {config_path}")
    return config


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    section = config.get(name, {})
    return section if isinstance(section, dict) else {}


def summarise_dataset(dataset: pd.DataFrame) -> DatasetSummary:
    return DatasetSummary(
        rows=len(dataset),
        columns=len(dataset.columns),
        features=sum(column.startswith("feature_") for column in dataset),
        targets=sum(column.startswith("target_") for column in dataset),
        players=dataset["player_key"].nunique(),
        fixtures=dataset[["season", "fixture_id"]].drop_duplicates().shape[0],
        xg_target_rows=int(dataset["target_expected_goals_available"].sum()),
        defensive_target_rows=int(
            dataset["target_defensive_contribution_available"].sum()
        ),
    )


def write_dataset(dataset: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    dataset.to_csv(
        temporary_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
        float_format="%.10g",
    )
    temporary_path.replace(path)
    return path


def run_preprocessing(
    seasons: Sequence[str],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    progress: bool = True,
) -> tuple[Path, DatasetSummary]:
    reporter = print if progress else None
    raw_rows = load_seasons(
        seasons,
        base_url=base_url,
        timeout=timeout,
        progress=reporter,
    )
    if reporter:
        reporter(f"Building pre-match features for {len(raw_rows):,} player fixtures")
    dataset = build_training_dataset(raw_rows)
    path = write_dataset(dataset, output_path)
    return path, summarise_dataset(dataset)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML configuration file used for defaults.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Vaastav season folders, for example 2022-23 2023-24.",
    )
    parser.add_argument("--output", default=None, help="Processed CSV output path.")
    parser.add_argument("--base-url", default=None, help="Vaastav data base URL.")
    parser.add_argument(
        "--timeout", type=float, default=None, help="HTTP timeout per source file."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    source_config = _section(config, "source")
    preprocessing_config = _section(config, "preprocessing")

    configured_seasons = preprocessing_config.get("seasons", list(DEFAULT_SEASONS))
    if not isinstance(configured_seasons, list) or not configured_seasons:
        raise ValueError("preprocessing.seasons must be a non-empty list")
    seasons = args.seasons or [str(season) for season in configured_seasons]
    output_path = args.output or str(
        preprocessing_config.get("output_path", DEFAULT_OUTPUT_PATH)
    )
    base_url = args.base_url or str(source_config.get("base_url", DEFAULT_BASE_URL))
    timeout = args.timeout
    if timeout is None:
        timeout = float(source_config.get("timeout_seconds", 30.0))

    try:
        path, summary = run_preprocessing(
            seasons,
            output_path,
            base_url=base_url,
            timeout=timeout,
        )
    except HTTPError as error:
        raise SystemExit(
            f"Could not retrieve {error.url} (HTTP {error.code}). "
            "Check the configured Vaastav seasons."
        ) from error

    print(
        f"Wrote {summary.rows:,} rows and {summary.columns:,} columns to {path} "
        f"({summary.features:,} features, {summary.targets:,} targets)"
    )
    print(
        f"Coverage: {summary.players:,} players, {summary.fixtures:,} fixtures, "
        f"{summary.xg_target_rows:,} xG rows, "
        f"{summary.defensive_target_rows:,} defensive-contribution rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
