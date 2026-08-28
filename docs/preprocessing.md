# Preprocessing dataset

The preprocessing job builds one CSV at player-fixture grain for all component
models in `plan.md`. It downloads Vaastav's FPL archive CSVs directly into
memory and does not retain raw files or create an ingestion database.

Run from the repository root:

```powershell
$env:PYTHONPATH = "backend/src"
python -m preprocessing.build_dataset
```

The default output is `data/processed/model_training.csv`. Change seasons or
the output path in `configs/preprocessing.yaml`, or pass `--seasons` and
`--output` on the command line.

## Dataset contract

- The row key is `season`, `player_id`, `fixture_id`.
- Identifier and fixture-context columns are unprefixed.
- Every model input starts with `feature_`.
- Every outcome starts with `target_`, allowing each component trainer to
  select its own target without rebuilding the shared history.
- Current price, ownership, and transfer features come from that Gameweek's
  `merged_gw.csv` snapshot. Price changes are calculated between those
  snapshots.
- Historical player and team features are shifted before rolling or expanding
  calculations. The target fixture never contributes to its own features.
- Player and team codes provide cross-season identity. Fixture membership is
  used for each row so domestic transfers do not inherit the player's
  end-of-season team.
- Assistant Manager chip elements in the 2024/25 archive are excluded; the
  training grain contains only goalkeepers, defenders, midfielders, and
  forwards.
- Blanks and doubles remain fixture rows. Team fixture count and sequence
  features describe congestion within the Gameweek.

Vaastav has xG fields for every configured season from 2022/23. Defensive
contribution, clearances/blocks/interceptions, recoveries, and tackles begin in
2025/26. Earlier targets and rolling history remain blank, with explicit
availability and observation-count columns; they are not changed to zero.

`data/` is ignored by Git because the generated CSV is reproducible and large.
