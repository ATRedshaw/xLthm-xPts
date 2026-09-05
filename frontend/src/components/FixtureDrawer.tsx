import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { api } from "../lib/api";
import { classNames, formatDateTime, formatPoints, formatPrice, humanise } from "../lib/format";
import { useRequest } from "../hooks/useRequest";
import type { Fixture, Position } from "../types";
import { Drawer, ErrorState, JsonPanel, LoadingState, PositionBadge, ProbabilityBar, TeamBadge } from "./ui";

export function FixtureDrawer({ summary, onClose }: { summary: Fixture; onClose: () => void }) {
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState<Position | "ALL">("ALL");
  const [expandedPlayer, setExpandedPlayer] = useState<number | null>(null);
  const request = useRequest((signal) => api.fixture(summary.fixture, signal), [summary.fixture]);
  const fixture = request.data;
  const players = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return [...(fixture?.players ?? [])]
      .filter((player) => position === "ALL" || player.position === position)
      .filter((player) => !needle || player.name.toLowerCase().includes(needle))
      .sort((a, b) => b.projection.xpts - a.projection.xpts);
  }, [fixture, position, search]);

  return (
    <Drawer open wide title={`${summary.home_team} — ${summary.away_team}`} subtitle={`GW ${summary.gameweek} · Fixture ${summary.fixture} · ${formatDateTime(summary.kickoff_time)}`} onClose={onClose}>
      {request.loading && !fixture ? <LoadingState label="Loading fixture detail" /> : request.error ? <ErrorState error={request.error} retry={request.retry} /> : fixture ? (
        <div>
          <section className="border-b border-white/[0.08] px-5 py-6 sm:px-6">
            <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4">
              <div className="flex flex-col items-center gap-2 sm:flex-row"><TeamBadge team={fixture.home_team} size="lg" /><span className="font-mono text-base font-semibold text-white">{fixture.home_team}</span></div>
              <div className="text-center">
                <p className="font-mono text-[9px] uppercase tracking-[0.12em] text-stone-600">Expected goals</p>
                <p className="mt-2 font-mono text-3xl font-semibold text-white">{fixture.forecast.expected_goals.home.toFixed(2)}<span className="mx-2 text-stone-700">:</span>{fixture.forecast.expected_goals.away.toFixed(2)}</p>
              </div>
              <div className="flex flex-col-reverse items-center gap-2 sm:flex-row sm:justify-end"><span className="font-mono text-base font-semibold text-white">{fixture.away_team}</span><TeamBadge team={fixture.away_team} size="lg" /></div>
            </div>
          </section>

          <section className="grid gap-5 border-b border-white/[0.08] p-5 sm:grid-cols-2 sm:p-6">
            <div>
              <p className="section-label mb-4">Match result</p>
              <div className="space-y-4">
                <ProbabilityBar label={`${fixture.home_team} win`} value={fixture.forecast.result_probabilities.home_win} />
                <ProbabilityBar label="Draw" value={fixture.forecast.result_probabilities.draw} tone="stone" />
                <ProbabilityBar label={`${fixture.away_team} win`} value={fixture.forecast.result_probabilities.away_win} />
              </div>
            </div>
            <div>
              <p className="section-label mb-4">Clean sheet</p>
              <div className="space-y-4">
                <ProbabilityBar label={fixture.home_team} value={fixture.forecast.clean_sheet_probabilities.home} />
                <ProbabilityBar label={fixture.away_team} value={fixture.forecast.clean_sheet_probabilities.away} />
              </div>
            </div>
          </section>

          <section className="p-5 sm:p-6">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div><p className="section-label">Player projections</p><p className="mt-1 text-xs text-stone-500">Sorted by fixture xPts. Open a row for model components.</p></div>
              <label className="relative sm:w-56"><Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-stone-600" /><input className="field pl-9" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a player…" /></label>
            </div>
            <div className="mt-4 flex gap-1">
              {(["ALL", "GK", "DEF", "MID", "FWD"] as const).map((item) => <button key={item} className={classNames("h-7 border px-2.5 font-mono text-[9px]", position === item ? "border-signal-350/50 bg-signal-450/10 text-signal-300" : "border-white/[0.08] text-stone-600")} onClick={() => setPosition(item)}>{item}</button>)}
            </div>

            <div className="mt-3 border border-white/[0.08]">
              <div className="grid grid-cols-[1fr_4rem_4rem_4rem] border-b border-white/[0.08] bg-black/10 px-3 py-2 font-mono text-[8px] uppercase tracking-[0.11em] text-stone-600 sm:grid-cols-[1fr_5rem_5rem_5rem_5rem]">
                <span>Player</span><span className="hidden sm:block">Price</span><span className="text-right">xMins</span><span className="text-right">xPts</span><span className="text-right">Owned</span>
              </div>
              {players.map((player) => {
                const expanded = expandedPlayer === player.id;
                return (
                  <div key={player.id} className="border-b border-white/[0.06] last:border-0">
                    <button className="grid w-full grid-cols-[1fr_4rem_4rem_4rem] items-center px-3 py-2.5 text-left hover:bg-signal-450/[0.035] sm:grid-cols-[1fr_5rem_5rem_5rem_5rem]" onClick={() => setExpandedPlayer(expanded ? null : player.id)}>
                      <span className="flex min-w-0 items-center gap-2"><PositionBadge position={player.position} /><span className="truncate text-xs text-stone-200">{player.name}</span><span className="font-mono text-[9px] text-stone-600">{player.team}</span></span>
                      <span className="hidden font-mono text-[10px] text-stone-500 sm:block">{formatPrice(player.price)}</span>
                      <span className="text-right font-mono text-[10px] text-stone-400">{player.projection.xmins.toFixed(1)}</span>
                      <span className="text-right font-mono text-xs font-medium text-signal-300">{formatPoints(player.projection.xpts)}</span>
                      <span className="text-right font-mono text-[10px] text-stone-500">{player.selected_by.toFixed(1)}%</span>
                    </button>
                    {expanded && (
                      <div className="grid gap-6 border-t border-white/[0.06] bg-black/10 p-4 sm:grid-cols-2">
                        <div>
                          <p className="section-label mb-3">Action probabilities</p>
                          <div className="space-y-3">{Object.entries(player.projection.action_probabilities ?? {}).map(([key, value]) => <ProbabilityBar key={key} label={humanise(key)} value={value} />)}</div>
                        </div>
                        <div>
                          <p className="section-label mb-3">xPts breakdown</p>
                          <div className="space-y-2">{Object.entries(player.projection.xpts_breakdown ?? {}).map(([key, value]) => <div key={key} className="flex items-center justify-between gap-3 border-b border-white/[0.05] pb-2 text-xs"><span className="text-stone-500">{humanise(key)}</span><span className={classNames("font-mono tabular-nums", value < 0 ? "text-rose-300" : "text-stone-200")}>{value > 0 ? "+" : ""}{value.toFixed(3)}</span></div>)}</div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <p className="mt-2 font-mono text-[9px] text-stone-600">{players.length} player records</p>
            <div className="mt-6"><JsonPanel data={fixture} label="Full fixture API response" /></div>
          </section>
        </div>
      ) : null}
    </Drawer>
  );
}
