import { CheckCircle2, Database, Route } from "lucide-react";
import { formatDateTime, humanise } from "../lib/format";
import type { Metadata } from "../types";
import { PageHeading, StatStrip } from "../components/ui";

const methodology = [
  {
    stage: "Data and observation unit",
    description: "Historical player and fixture records come from Vaastav's FPL archive. Current player metadata, availability, prices, ownership, transfers and future fixtures come from the official FPL API. The base table contains one row per player and fixture. Team-score training reduces this to one row for each team perspective in a fixture.",
  },
  {
    stage: "Pre-match features",
    description: "Player and team form is calculated as shifted rolling sums, means and per-90 rates over 3, 5 and 10 fixtures. Other candidates include season-to-date form, previous-season form, five-match venue form, rest, home advantage, fixture order within a gameweek and market data. Numeric gaps are median-imputed with missingness indicators. Categorical gaps are mode-imputed before one-hot encoding.",
  },
  {
    stage: "Walk-forward validation",
    description: "Model selection uses expanding-window splits by season. Each validation season is predicted from earlier seasons only. Most components use 2023-24, 2024-25 and 2025-26 as validation folds. Bonus uses the final two of those seasons. Downstream models are trained with out-of-fold team and minutes predictions rather than fitted upstream values.",
  },
  {
    stage: "Model family and objectives",
    description: "Every supervised estimator uses scikit-learn's histogram gradient-boosted decision trees. This is boosting, not a random forest. Boosting adds trees sequentially to reduce a chosen loss, while a random forest averages independently fitted trees. Regressors use Poisson loss for goal and event rates, squared error for state-conditioned minutes, and squared or absolute error for BPS. Minutes-state classifiers optimise log loss.",
  },
  {
    stage: "Selection statistics",
    description: "Count models are compared by mean Poisson deviance. The team model is selected by exact-score log loss, with clean-sheet Brier score and goal MAE as tie-breakers. Minutes classifiers use multiclass log loss. Minutes duration uses RMSE. The BPS regressor uses appeared-player RMSE, while the bonus simulation uses multiclass log loss. Brier scores, calibration error, MAE and ROC AUC are retained as diagnostics where applicable.",
  },
  {
    stage: "Team score distribution",
    description: "A scikit-learn HistGradientBoostingRegressor with Poisson loss estimates each team's goal rate from match context and rolling score or expected-goal form. Predicted home and away rates create a Poisson score matrix from 0 to 10 goals. A Dixon-Coles parameter adjusts the 0-0, 0-1, 1-0 and 1-1 cells, then the matrix is normalised. Its value is tuned on walk-forward exact-score log loss. The matrix provides expected goals, result probabilities, clean-sheet probabilities and scoreline probabilities.",
  },
  {
    stage: "Minutes distribution",
    description: "Two HistGradientBoostingClassifier models use a log-loss objective. The first estimates appearance. The second splits an appearance into substitute under 60, substitute 60 or more, starter under 60 and starter 60 or more. Multiplying the conditional probabilities gives one coherent five-state distribution including did not play. Temperature scaling is tuned on walk-forward predictions. A squared-error HistGradientBoostingRegressor predicts minutes conditional on state, with an out-of-fold offset correction and state-specific bounds.",
  },
  {
    stage: "Attacking rates",
    description: "Separate Poisson HistGradientBoostingRegressor models estimate player xG and xA rates per 90. Training rows are weighted by minutes. Each rate is shrunk towards a position-level prior according to the player's historical minutes, then multiplied by forecast minutes. Player xG is rescaled within each team to equal the team model's expected goals. Assist shares are rescaled using the historical rate of FPL assists per team goal and a multiplier tuned out of fold. Any-goal and any-assist probabilities use the Poisson expression 1 - exp(-expected count).",
  },
  {
    stage: "Defence and rare events",
    description: "Poisson HistGradientBoostingRegressor models estimate goalkeeper saves per 90 and outfield defensive contributions per 90. Both estimates are exposure-weighted and shrunk towards position priors. Player clean-sheet probability combines the team clean-sheet probability with the chance of reaching 60 minutes. Penalty saves, red cards, own goals and penalty misses use current and previous-season counts shrunk towards position rates. Yellow cards use another Poisson gradient-boosting rate model. Poisson tail probabilities convert count means into save and defensive-contribution thresholds.",
  },
  {
    stage: "BPS and bonus",
    description: "A HistGradientBoostingRegressor predicts BPS conditional on appearance from the upstream team, minutes, attack, defence and card estimates. Candidate models use squared-error or absolute-error loss. Residual standard deviations are estimated by position. Simulation adds a normal residual to conditional BPS, rounds the result and applies the official 3, 2 and 1 bonus allocation with its tie rules. The residual scale is selected by out-of-fold bonus log loss.",
  },
  {
    stage: "Joint fixture simulation",
    description: "Each Monte Carlo run draws a scoreline from the Dixon-Coles matrix and a minutes state for every player. Goals and assists are assigned among players who appear using minutes-adjusted attacking weights. Saves, penalty saves and defensive contributions are Poisson draws. Card and penalty-miss events are Bernoulli draws derived from Poisson event probabilities. Goals conceded are sampled against each player's minutes exposure. The current FPL rules convert the joint outcome into points.",
  },
  {
    stage: "Live inference and API output",
    description: "At inference time, official status and chance-of-playing data adjust appearance probabilities for the next gameweek. Components run in dependency order, starting with team score and minutes and ending with BPS. xPts and xMins are Monte Carlo means. Threshold probabilities and percentiles are empirical summaries of the samples. Fixture samples are added before summarising double gameweeks. The completed batch is written to a temporary SQLite database and replaces the live API database only after a successful write.",
  },
];

const qualityCheckLabels: Record<string, string> = {
  maximum_sampled_team_goal_mean_error: "Worst team-goal simulation drift",
  maximum_sampled_player_minutes_mean_error: "Worst player-minutes simulation drift",
};

export function ModelView({ metadata }: { metadata: Metadata }) {
  const gameweeks = metadata.coverage.gameweeks;
  const firstGameweek = gameweeks[0];
  const lastGameweek = gameweeks[gameweeks.length - 1];

  return (
    <>
      <PageHeading
        eyebrow={`Season ${metadata.season} / Inference record`}
        title="Model & API"
        description="See how historical FPL records become current player projections, with run details and simulation checks."
      />
      <StatStrip items={[
        { label: "Simulation size", value: metadata.simulation_count.toLocaleString("en-GB"), detail: "outcomes / fixture" },
        { label: "Coverage", value: `${metadata.coverage.players}`, detail: "players" },
        { label: "Fixture set", value: `${metadata.coverage.fixtures}`, detail: "fixtures" },
        { label: "Projection span", value: `GW ${firstGameweek}—${lastGameweek}`, detail: `${gameweeks.length} rounds` },
      ]} />

      <div className="grid gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,1fr)_280px] lg:px-8">
        <div className="space-y-5">
          <section className="border border-white/[0.08] bg-ink-900">
            <div className="flex items-center gap-3 border-b border-white/[0.08] px-4 py-3">
              <Database className="h-4 w-4 text-signal-350" />
              <div><h2 className="text-sm font-medium text-stone-100">Inference batch</h2><p className="mt-0.5 text-[10px] text-stone-600">Reproducibility and source-data timestamps</p></div>
            </div>
            <div className="grid sm:grid-cols-2">
              {[
                ["Generated", formatDateTime(metadata.generated_at)],
                ["Source retrieved", formatDateTime(metadata.data_retrieved_at)],
                ["Season", metadata.season],
                ["Ruleset", metadata.ruleset],
                ["Random state", String(metadata.random_state)],
                ["Skipped fixtures", String(metadata.coverage.skipped_fixtures.length)],
              ].map(([label, value]) => (
                <div key={label} className="border-b border-white/[0.06] px-4 py-3 last:border-b-0 sm:border-r sm:[&:nth-last-child(-n+2)]:border-b-0 sm:[&:nth-child(even)]:border-r-0">
                  <p className="section-label">{label}</p><p className="mt-1.5 font-mono text-xs text-stone-300">{value}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="border border-white/[0.08] bg-ink-900">
            <div className="flex items-center gap-3 border-b border-white/[0.08] px-4 py-3">
              <Route className="h-4 w-4 text-signal-350" />
              <div><h2 className="text-sm font-medium text-stone-100">Methodology</h2><p className="mt-0.5 text-[10px] text-stone-600">From source records to published player projections</p></div>
            </div>
            <ol className="divide-y divide-white/[0.06]">
              {methodology.map(({ stage, description }, index) => (
                <li key={stage} className="grid grid-cols-[2rem_1fr] gap-3 px-4 py-4">
                  <span className="font-mono text-xs text-signal-350">{String(index + 1).padStart(2, "0")}</span>
                  <div><h3 className="text-xs font-medium text-stone-200">{stage}</h3><p className="mt-1 text-xs leading-5 text-stone-500">{description}</p></div>
                </li>
              ))}
            </ol>
          </section>
        </div>

        <aside className="space-y-5">
          <section className="border border-white/[0.08] bg-ink-900">
            <div className="flex items-center gap-3 border-b border-white/[0.08] px-4 py-3"><CheckCircle2 className="h-4 w-4 text-emerald-400" /><h2 className="text-sm font-medium text-stone-100">Quality checks</h2></div>
            <div className="divide-y divide-white/[0.06]">
              {Object.entries(metadata.quality_checks).map(([check, value]) => (
                <div key={check} className="px-4 py-4"><p className="text-[10px] leading-4 text-stone-500">{qualityCheckLabels[check] ?? humanise(check)}</p><p className="mt-2 font-mono text-xl font-medium text-stone-200">{value.toFixed(3)}</p></div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </>
  );
}
