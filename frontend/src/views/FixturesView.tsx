import { useMemo, useState } from "react";
import { CalendarClock, Search } from "lucide-react";
import { api } from "../lib/api";
import { formatDateTime, formatPercent } from "../lib/format";
import { useRequest } from "../hooks/useRequest";
import type { Fixture, Metadata } from "../types";
import { EmptyState, ErrorState, LoadingState, PageHeading, StatStrip, TeamBadge } from "../components/ui";
import { FixtureDrawer } from "../components/FixtureDrawer";

const windowOptions = [1, 3, 5, 10, 38];

function resultLeader(fixture: Fixture) {
  const probabilities = fixture.forecast.result_probabilities;
  const entries = [
    { label: fixture.home_team, value: probabilities.home_win },
    { label: "Draw", value: probabilities.draw },
    { label: fixture.away_team, value: probabilities.away_win },
  ];
  return entries.sort((a, b) => b.value - a.value)[0];
}

function FixtureRow({ fixture, onOpen }: { fixture: Fixture; onOpen: () => void }) {
  const probabilities = fixture.forecast.result_probabilities;
  const leader = resultLeader(fixture);
  return (
    <button className="group w-full border border-white/[0.08] bg-ink-900 p-4 text-left transition-colors hover:border-signal-350/30 hover:bg-signal-450/[0.025]" onClick={onOpen}>
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.1em] text-stone-600">
          <CalendarClock className="h-3 w-3" />{formatDateTime(fixture.kickoff_time)}
        </span>
        <span className="font-mono text-[9px] text-stone-700">#{fixture.fixture}</span>
      </div>
      <div className="mt-5 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <TeamBadge team={fixture.home_team} />
          <span className="truncate font-mono text-sm font-medium text-stone-200">{fixture.home_team}</span>
        </div>
        <div className="text-center">
          <p className="font-mono text-[9px] uppercase text-stone-600">model xG</p>
          <p className="mt-1 font-mono text-lg font-semibold tabular-nums text-white">
            {fixture.forecast.expected_goals.home.toFixed(2)}<span className="mx-1.5 text-stone-700">:</span>{fixture.forecast.expected_goals.away.toFixed(2)}
          </p>
        </div>
        <div className="flex min-w-0 items-center justify-end gap-2">
          <span className="truncate font-mono text-sm font-medium text-stone-200">{fixture.away_team}</span>
          <TeamBadge team={fixture.away_team} />
        </div>
      </div>
      <div className="mt-5 flex h-1.5 overflow-hidden bg-white/[0.06]" aria-label={`${fixture.home_team} win ${formatPercent(probabilities.home_win)}, draw ${formatPercent(probabilities.draw)}, ${fixture.away_team} win ${formatPercent(probabilities.away_win)}`}>
        <span className="bg-signal-450" style={{ width: `${probabilities.home_win * 100}%` }} />
        <span className="bg-stone-500" style={{ width: `${probabilities.draw * 100}%` }} />
        <span className="bg-signal-350/35" style={{ width: `${probabilities.away_win * 100}%` }} />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[9px] text-stone-600">
        <span>H {formatPercent(probabilities.home_win)}</span>
        <span className="text-stone-500">Lean: {leader.label} {formatPercent(leader.value)}</span>
        <span>A {formatPercent(probabilities.away_win)}</span>
      </div>
    </button>
  );
}

export function FixturesView({ metadata }: { metadata: Metadata }) {
  const availableGameweeks = metadata.coverage.gameweeks;
  const [startGameweek, setStartGameweek] = useState(availableGameweeks[0]);
  const [gameweeks, setGameweeks] = useState(Math.min(5, availableGameweeks.length));
  const [search, setSearch] = useState("");
  const [selectedFixture, setSelectedFixture] = useState<Fixture | null>(null);
  const remainingGameweeks = availableGameweeks.filter((item) => item >= startGameweek).length;
  const effectiveWindow = Math.min(gameweeks, remainingGameweeks);
  const request = useRequest(
    (signal) => api.fixtures({ start_gameweek: startGameweek, gameweeks: effectiveWindow }, signal),
    [startGameweek, effectiveWindow],
  );
  const fixtures = useMemo(() => request.data?.fixtures ?? [], [request.data]);
  const filteredFixtures = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return fixtures.filter((fixture) => !needle || `${fixture.home_team} ${fixture.away_team}`.toLowerCase().includes(needle));
  }, [fixtures, search]);
  const groupedFixtures = useMemo(() => {
    return filteredFixtures.reduce<Record<number, Fixture[]>>((groups, fixture) => {
      (groups[fixture.gameweek] ||= []).push(fixture);
      return groups;
    }, {});
  }, [filteredFixtures]);
  const averageGoals = fixtures.length
    ? fixtures.reduce((total, item) => total + item.forecast.expected_goals.home + item.forecast.expected_goals.away, 0) / fixtures.length
    : 0;
  const strongestFavourite = [...fixtures].sort((a, b) => resultLeader(b).value - resultLeader(a).value)[0];
  const cleanSheets = fixtures.flatMap((fixture) => [
    { team: fixture.home_team, value: fixture.forecast.clean_sheet_probabilities.home },
    { team: fixture.away_team, value: fixture.forecast.clean_sheet_probabilities.away },
  ]);
  const bestCleanSheet = cleanSheets.sort((a, b) => b.value - a.value)[0];

  function changeStartGameweek(value: number) {
    const remaining = availableGameweeks.filter((item) => item >= value).length;
    setStartGameweek(value);
    setGameweeks((current) => Math.min(current, remaining));
  }

  return (
    <>
      <PageHeading
        eyebrow={`Season ${metadata.season} / Fixture model`}
        title="Fixture forecasts"
        description="Read the score model across the schedule, compare result probabilities, and drill into each match’s projected player output."
        actions={
          <div className="flex items-center gap-2">
            <select className="control pr-8" value={startGameweek} onChange={(event) => changeStartGameweek(Number(event.target.value))} aria-label="Start gameweek">
              {availableGameweeks.map((item) => <option key={item} value={item}>From GW {item}</option>)}
            </select>
            <select className="control pr-8" value={effectiveWindow} onChange={(event) => setGameweeks(Number(event.target.value))} aria-label="Projection window">
              {windowOptions.filter((item) => item <= remainingGameweeks).map((item) => <option key={item} value={item}>{item === remainingGameweeks ? `All ${item} GWs` : `${item} GW${item === 1 ? "" : "s"}`}</option>)}
              {!windowOptions.includes(remainingGameweeks) && <option value={remainingGameweeks}>All {remainingGameweeks} GWs</option>}
            </select>
          </div>
        }
      />

      {request.loading && !request.data ? <LoadingState label="Loading fixture forecasts" /> : request.error ? <ErrorState error={request.error} retry={request.retry} /> : (
        <>
          <StatStrip items={[
            { label: "Schedule", value: `${fixtures.length}`, detail: `fixtures / ${effectiveWindow} GWs` },
            { label: "Expected goals", value: averageGoals.toFixed(2), detail: "average total" },
            { label: "Strongest favourite", value: strongestFavourite ? resultLeader(strongestFavourite).label : "—", detail: strongestFavourite ? formatPercent(resultLeader(strongestFavourite).value) : undefined },
            { label: "Best clean-sheet", value: bestCleanSheet?.team ?? "—", detail: bestCleanSheet ? formatPercent(bestCleanSheet.value) : undefined },
          ]} />

          <section className="px-4 py-5 sm:px-6 lg:px-8">
            <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="section-label">Scheduled rounds</p>
                <p className="mt-1 text-xs text-stone-500">Open any fixture for team forecasts and player-level xPts.</p>
              </div>
              <label className="relative w-full sm:w-64">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-stone-600" />
                <input className="field pl-9" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter by team…" />
              </label>
            </div>

            {Object.keys(groupedFixtures).length ? (
              <div className="space-y-8">
                {Object.entries(groupedFixtures).map(([gameweek, items]) => (
                  <section key={gameweek}>
                    <div className="mb-3 flex items-center gap-3">
                      <h2 className="font-mono text-xs font-medium text-stone-200">GW {gameweek}</h2>
                      <span className="h-px flex-1 bg-white/[0.08]" />
                      <span className="font-mono text-[9px] uppercase tracking-[0.1em] text-stone-600">{items.length} fixtures</span>
                    </div>
                    <div className="grid gap-2 xl:grid-cols-2">
                      {items.map((fixture) => <FixtureRow key={fixture.fixture} fixture={fixture} onOpen={() => setSelectedFixture(fixture)} />)}
                    </div>
                  </section>
                ))}
              </div>
            ) : <EmptyState>No fixtures match that team filter.</EmptyState>}
          </section>
        </>
      )}

      {selectedFixture && <FixtureDrawer summary={selectedFixture} onClose={() => setSelectedFixture(null)} />}
    </>
  );
}
