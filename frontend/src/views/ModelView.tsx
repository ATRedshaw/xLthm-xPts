import { CheckCircle2, Database, Route } from "lucide-react";
import { formatDateTime, humanise } from "../lib/format";
import type { Metadata } from "../types";
import { PageHeading, StatStrip } from "../components/ui";

const methodology = [
  {
    stage: "Historical data",
    description: "Archived player-fixture records are read from Vaastav. The active season, current player status and remaining schedule come from official FPL endpoints. Records are ordered by kickoff time; rolling form is shifted by one fixture, so a training row cannot see its own result.",
  },
  {
    stage: "Training rows",
    description: "The training set contains one row per player and fixture, with team-level rows derived from the same history. It includes prior minutes and events, team and opponent form, venue, rest, price, ownership and transfers where a model calls for them. Missing history is imputed inside each training fold.",
  },
  {
    stage: "Model selection",
    description: "Training runs chronologically. Earlier seasons fit each candidate and later seasons score it. Feature sets, gradient-boosting settings, shrinkage and calibration values are chosen from these walk-forward results. Models that depend on team scores or minutes are trained against walk-forward upstream predictions rather than fitted values.",
  },
  {
    stage: "Teams and minutes",
    description: "A Poisson gradient-boosting model estimates each team's goal rate. The home and away rates feed a Dixon-Coles score matrix, which supplies scoreline, result and clean-sheet probabilities. Separate gradient-boosting classifiers estimate mutually exclusive minutes states; a duration regressor supplies minutes within each appeared state.",
  },
  {
    stage: "Player events",
    description: "Attacking models estimate player goal and FPL-assist rates for forecast minutes. Goal estimates are rescaled so the players on a team sum to its expected goals; assist shares are calibrated separately. Defensive models use forecast minutes and team strength to estimate clean-sheet exposure, goals conceded, saves, penalty saves and defensive contributions.",
  },
  {
    stage: "Cards and bonus",
    description: "Yellow cards use a fitted event-rate model. Rarer red cards, own goals and penalty misses rely on player history shrunk towards position rates. The BPS regressor takes the upstream component forecasts and estimates points conditional on appearance, with residual spread learned by position.",
  },
  {
    stage: "Live prediction",
    description: "Before inference, the same feature builder joins historical form to the official live player and fixture feed. Player status and chance of playing adjust the next gameweek's minutes distribution. Saved models then run in dependency order: team and minutes first, followed by the player-event and BPS forecasts.",
  },
  {
    stage: "Fixture simulation",
    description: "For each fixture, the simulator draws a scoreline and a minutes state for every player. It assigns sampled goals and assists among players who appear, draws the other events at rates scaled by sampled minutes, and applies the current FPL scoring rules. BPS residuals are sampled for the same appearance outcome before the 3/2/1 bonus tie rules are applied.",
  },
  {
    stage: "Published projections",
    description: "xPts and xMins are the means of the simulated player outcomes. Fixture samples are added at gameweek level, so double gameweeks retain the combined distribution. Threshold probabilities and percentiles come from those samples. A complete batch is written to a temporary SQLite database and replaces the live database after the write succeeds.",
  },
];

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
                <div key={check} className="px-4 py-4"><p className="text-[10px] leading-4 text-stone-500">{humanise(check)}</p><p className="mt-2 font-mono text-xl font-medium text-stone-200">{value.toFixed(3)}</p></div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </>
  );
}
