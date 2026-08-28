from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from preprocessing.build_dataset import summarise_dataset, write_dataset


class BuildDatasetTests(unittest.TestCase):
    def test_writes_csv_and_reports_target_coverage(self) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "season": "2025-26",
                    "fixture_id": 1,
                    "player_key": "code:1",
                    "feature_price_tenths": 50,
                    "target_expected_goals_available": 1,
                    "target_defensive_contribution_available": 0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset(dataset, Path(directory) / "processed.csv")
            written = pd.read_csv(path)

        summary = summarise_dataset(dataset)
        self.assertEqual(written.loc[0, "feature_price_tenths"], 50)
        self.assertEqual(summary.rows, 1)
        self.assertEqual(summary.features, 1)
        self.assertEqual(summary.targets, 2)
        self.assertEqual(summary.xg_target_rows, 1)
        self.assertEqual(summary.defensive_target_rows, 0)


if __name__ == "__main__":
    unittest.main()
