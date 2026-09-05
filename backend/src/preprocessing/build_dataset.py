"""Build the shared component-model training CSV from historical FPL data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import yaml

from .features import build_training_dataset
from .fpl import DEFAULT_BASE_URL as FPL_BASE_URL
from .fpl import fetch_live_fpl, load_current_season
from .vaastav import DEFAULT_BASE_URL as VAASTAV_BASE_URL
from .vaastav import load_seasons


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
    current_season: str,
    vaastav_base_url: str = VAASTAV_BASE_URL,
    fpl_base_url: str = FPL_BASE_URL,
    timeout: float = 30.0,
    fpl_attempts: int = 3,
    fpl_workers: int = 8,
    progress: bool = True,
) -> tuple[Path, DatasetSummary]:
    reporter = print if progress else None
    archived_seasons = [season for season in seasons if season != current_season]
    frames = []
    if archived_seasons:
        frames.append(
            load_seasons(
                archived_seasons,
                base_url=vaastav_base_url,
                timeout=timeout,
                progress=reporter,
            )
        )
    if current_season in seasons:
        live = fetch_live_fpl(
            base_url=fpl_base_url,
            timeout=timeout,
            attempts=fpl_attempts,
        )
        frames.append(
            load_current_season(
                live,
                season=current_season,
                base_url=fpl_base_url,
                timeout=timeout,
                attempts=fpl_attempts,
                workers=fpl_workers,
                progress=reporter,
            )
        )
    raw_rows = pd.concat(frames, ignore_index=True).sort_values(
        ["kickoff_time", "season", "fixture_id", "player_id"],
        kind="stable",
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
        help="Seasons to include, for example 2022-23 2023-24.",
    )
    parser.add_argument("--output", default=None, help="Processed CSV output path.")
    parser.add_argument(
        "--vaastav-base-url", default=None, help="Vaastav archive base URL."
    )
    parser.add_argument("--fpl-base-url", default=None, help="Official FPL API URL.")
    parser.add_argument(
        "--current-season", default=None, help="Season sourced from the FPL API."
    )
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
    current_season = args.current_season or str(
        preprocessing_config.get("current_season", seasons[-1])
    )
    output_path = args.output or str(
        preprocessing_config.get("output_path", DEFAULT_OUTPUT_PATH)
    )
    vaastav_base_url = args.vaastav_base_url or str(
        source_config.get("vaastav_base_url", VAASTAV_BASE_URL)
    )
    fpl_base_url = args.fpl_base_url or str(
        source_config.get("fpl_base_url", FPL_BASE_URL)
    )
    timeout = args.timeout
    if timeout is None:
        timeout = float(source_config.get("timeout_seconds", 30.0))

    try:
        path, summary = run_preprocessing(
            seasons,
            output_path,
            current_season=current_season,
            vaastav_base_url=vaastav_base_url,
            fpl_base_url=fpl_base_url,
            timeout=timeout,
            fpl_attempts=int(source_config.get("fpl_request_attempts", 3)),
            fpl_workers=int(source_config.get("fpl_request_workers", 8)),
        )
    except HTTPError as error:
        raise SystemExit(
            f"Could not retrieve {error.url} (HTTP {error.code}). "
            "Check the configured historical sources and seasons."
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
