import { useEffect, useMemo, useState } from "react";
import { Activity, Database, LineChart, ShieldCheck } from "lucide-react";
import { api } from "../lib/api";
import { classNames, formatPercent, formatPoints, formatPrice, humanise } from "../lib/format";
import { useRequest } from "../hooks/useRequest";
import type { FixtureProjection, Player } from "../types";
import {
  Drawer,
  ErrorState,
  JsonPanel,
  KeyValueGrid,
  LoadingState,
  PlayerBadge,
  PointsChart,
  PositionBadge,
  ProbabilityBar,
  TeamBadge,
} from "./ui";

type Tab = "outlook" | "probability" | "data";

const tabs: Array<{ id: Tab; label: string; icon: typeof Activity }> = [
  { id: "outlook", label: "Outlook", icon: LineChart },
  { id: "probability", label: "Probability", icon: Activity },
  { id: "data", label: "Data", icon: Database },
];

function Breakdown({ values }: { values: Record<string, number> }) {
  const maximum = Math.max(...Object.values(values).map(Math.abs), 1);
  return (
    <div className="space-y-3">
      {Object.entries(values).map(([key, value]) => (
        <div key={key} className="grid grid-cols-[8rem_1fr_3rem] items-center gap-3 text-xs">
          <span className="truncate text-stone-500" title={humanise(key)}>{humanise(key)}</span>
          <div className="h-1 bg-white/[0.07]">
            <div className={classNames("h-full", value < 0 ? "bg-rose-400" : "bg-violet-450")} style={{ width: `${(Math.abs(value) / maximum) * 100}%` }} />
          </div>
          <span className={classNames("text-right font-mono tabular-nums", value < 0 ? "text-rose-300" : "text-stone-200")}>{value > 0 ? "+" : ""}{value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function Distribution({ values }: { values: Record<string, number> }) {
  const entries = Object.entries(values).sort(([a], [b]) => Number(a) - Number(b));
  const maximum = Math.max(...entries.map(([, value]) => value), 0.01);
  return (
    <div className="overflow-x-auto pb-2">
      <div className="flex h-44 min-w-[520px] items-end gap-1 border-b border-white/10 px-1">
        {entries.map(([points, probability]) => (
          <div key={points} className="group flex h-full min-w-3 flex-1 flex-col justify-end" title={`${points} pts: ${formatPercent(probability, 1)}`}>
            <div className="relative bg-violet-450/55 transition-colors group-hover:bg-violet-350" style={{ height: `${Math.max(2, (probability / maximum) * 100)}%` }} />
            <span className="mt-2 block text-center font-mono text-[7px] text-stone-600">{points}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-center font-mono text-[8px] uppercase tracking-[0.15em] text-stone-600">Simulated FPL points</p>
    </div>
  );
}

export function PlayerDrawer({
  summary,
  startGameweek,
  gameweeks,
  onClose,
}: {
  summary: Player;
  startGameweek: number;
  gameweeks: number;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<Tab>("outlook");
  const [selectedGameweek, setSelectedGameweek] = useState(startGameweek);
  const [fixtureIndex, setFixtureIndex] = useState(0);
  const request = useRequest(
    (signal) => api.player(summary.id, { start_gameweek: startGameweek, gameweeks }, signal),
    [summary.id, startGameweek, gameweeks],
  );
  const player = request.data;
  const gameweek = player?.future_points.find((item) => item.gameweek === selectedGameweek);
  const fixtures = gameweek?.fixture_projections ?? [];
  const fixture: FixtureProjection | undefined = fixtures[fixtureIndex] ?? fixtures[0];

  useEffect(() => {
    setSelectedGameweek(startGameweek);
    setFixtureIndex(0);
  }, [startGameweek, summary.id]);

  const averageMinutes = useMemo(() => {
    if (!player?.future_points.length) return 0;
    return player.future_points.reduce((total, item) => total + item.xmins, 0) / player.future_points.length;
  }, [player]);

  return (
    <Drawer open title={summary.name} subtitle={`${summary.team} · ${summary.position} · Player ${summary.id}`} onClose={onClose}>
      {request.loading && !player ? (
        <LoadingState label="Running full-detail request" />
      ) : request.error ? (
        <ErrorState error={request.error} retry={request.retry} />
      ) : player ? (
        <>
          <div className="flex items-center gap-4 border-b border-white/[0.08] px-5 py-5 sm:px-6">
            <PlayerBadge name={player.name} />
            <TeamBadge team={player.team} size="lg" />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2"><PositionBadge position={player.position} /><span className="font-mono text-xs text-stone-400">{formatPrice(player.price)}</span></div>
              <p className="mt-2 text-xs text-stone-500">Selected by {player.selected_by.toFixed(1)}% of managers</p>
            </div>
            <div className="text-right">
              <p className="font-mono text-3xl font-semibold tracking-tight text-white">{formatPoints(player.total_xpts)}</p>
              <p className="font-mono text-[9px] uppercase tracking-[0.12em] text-stone-600">total xPts</p>
            </div>
          </div>

          <div className="grid grid-cols-3 border-b border-white/[0.08]">
            {[
              ["Avg xMins", averageMinutes.toFixed(1)],
              ["Availability", player.availability_probability == null ? "—" : formatPercent(player.availability_probability)],
              ["Status", player.status === "a" ? "Available" : player.status?.toUpperCase() || "—"],
            ].map(([label, value]) => (
              <div key={label} className="border-r border-white/[0.08] px-3 py-3 text-center last:border-0">
                <p className="font-mono text-sm text-stone-200">{value}</p>
                <p className="mt-1 text-[9px] uppercase tracking-[0.11em] text-stone-600">{label}</p>
              </div>
            ))}
          </div>

          {player.news && (
            <div className="flex gap-3 border-b border-amber-300/15 bg-amber-300/[0.035] px-5 py-3 text-xs leading-5 text-amber-100/70 sm:px-6">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-amber-200" />{player.news}
            </div>
          )}

          <div className="sticky top-0 z-10 flex border-b border-white/[0.08] bg-ink-900 px-5 sm:px-6">
            {tabs.map((item) => {
              const Icon = item.icon;
              return (
                <button key={item.id} className={classNames("flex items-center gap-2 border-b-2 px-3 py-3 text-xs", tab === item.id ? "border-violet-350 text-white" : "border-transparent text-stone-500 hover:text-stone-300")} onClick={() => setTab(item.id)}>
                  <Icon className="h-3.5 w-3.5" />{item.label}
                </button>
              );
            })}
          </div>

          <div className="space-y-6 px-5 py-6 sm:px-6">
            <div>
              <div className="mb-3 flex items-center justify-between">
                <p className="section-label">Gameweek</p>
                <p className="font-mono text-[9px] text-stone-600">Select to inspect</p>
              </div>
              <div className="flex gap-1 overflow-x-auto pb-1">
                {player.future_points.map((item) => (
                  <button key={item.gameweek} className={classNames("min-w-[64px] border px-2 py-2 text-center", selectedGameweek === item.gameweek ? "border-violet-350/50 bg-violet-450/10" : "border-white/[0.08] hover:border-white/20")} onClick={() => { setSelectedGameweek(item.gameweek); setFixtureIndex(0); }}>
                    <span className="block font-mono text-[8px] uppercase text-stone-600">GW {item.gameweek}</span>
                    <span className={classNames("mt-1 block font-mono text-sm", selectedGameweek === item.gameweek ? "text-violet-300" : "text-stone-300")}>{formatPoints(item.xpts)}</span>
                  </button>
                ))}
              </div>
            </div>

            {tab === "outlook" && (
              <>
                <section>
                  <div className="mb-3 flex items-center justify-between"><p className="section-label">Projection path</p><span className="font-mono text-[9px] text-stone-600">xPts / gameweek</span></div>
                  <PointsChart values={player.future_points.map((item) => ({ label: `GW${item.gameweek}`, value: item.xpts }))} />
                </section>

                <section>
                  <p className="section-label mb-3">GW {selectedGameweek} fixture model</p>
                  {fixtures.length > 1 && (
                    <div className="mb-3 flex gap-1">
                      {fixtures.map((item, index) => <button key={item.fixture} className={classNames("control", index === fixtureIndex && "border-violet-350/50 text-violet-300")} onClick={() => setFixtureIndex(index)}>{item.opponent} ({item.is_home ? "H" : "A"})</button>)}
                    </div>
                  )}
                  {fixture ? (
                    <div className="border border-white/[0.08]">
                      <div className="grid grid-cols-3 border-b border-white/[0.08] bg-white/[0.018]">
                        <div className="p-3"><p className="text-[9px] uppercase text-stone-600">Opponent</p><p className="mt-1 font-mono text-sm text-stone-200">{fixture.opponent} ({fixture.is_home ? "H" : "A"})</p></div>
                        <div className="border-l border-white/[0.08] p-3"><p className="text-[9px] uppercase text-stone-600">xPts</p><p className="mt-1 font-mono text-sm text-violet-300">{formatPoints(fixture.xpts)}</p></div>
                        <div className="border-l border-white/[0.08] p-3"><p className="text-[9px] uppercase text-stone-600">xMins</p><p className="mt-1 font-mono text-sm text-stone-200">{fixture.xmins.toFixed(1)}</p></div>
                      </div>
                      {fixture.action_probabilities && (
                        <div className="grid gap-4 p-4 sm:grid-cols-2">
                          {Object.entries(fixture.action_probabilities).map(([key, value]) => <ProbabilityBar key={key} label={humanise(key)} value={value} />)}
                        </div>
                      )}
                    </div>
                  ) : <p className="text-xs text-stone-600">No fixture in this gameweek.</p>}
                </section>

                {fixture?.xpts_breakdown && <section><p className="section-label mb-4">xPts components</p><Breakdown values={fixture.xpts_breakdown} /></section>}
              </>
            )}

            {tab === "probability" && gameweek?.outcome_probabilities && (
              <>
                <section>
                  <p className="section-label mb-4">Outcome thresholds · GW {selectedGameweek}</p>
                  <div className="grid gap-4 sm:grid-cols-2">
                    {Object.entries(gameweek.outcome_probabilities)
                      .filter(([key]) => !["percentiles", "points_distribution"].includes(key))
                      .map(([key, value]) => <ProbabilityBar key={key} label={humanise(key)} value={value as number} tone={key === "negative_points" ? "red" : "violet"} />)}
                  </div>
                </section>
                <section>
                  <p className="section-label mb-3">Points range</p>
                  <KeyValueGrid values={gameweek.outcome_probabilities.percentiles} />
                </section>
                {gameweek.outcome_probabilities.points_distribution && (
                  <section><p className="section-label mb-4">Full distribution</p><Distribution values={gameweek.outcome_probabilities.points_distribution} /></section>
                )}
              </>
            )}

            {tab === "probability" && !gameweek?.outcome_probabilities && <p className="text-sm text-stone-500">No probability distribution is available for this gameweek.</p>}

            {tab === "data" && (
              <>
                <section>
                  <p className="section-label mb-3">Expected actions · GW {selectedGameweek}</p>
                  {fixture?.expected_actions ? <KeyValueGrid values={fixture.expected_actions} /> : <p className="text-xs text-stone-600">No fixture actions available.</p>}
                </section>
                {fixture?.outcome_probabilities && <JsonPanel data={fixture.outcome_probabilities} label="Fixture probability payload" />}
                <JsonPanel data={player} label="Full player API response" />
              </>
            )}
          </div>
        </>
      ) : null}
    </Drawer>
  );
}
