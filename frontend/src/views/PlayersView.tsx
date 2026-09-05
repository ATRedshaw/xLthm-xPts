import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Search, SlidersHorizontal } from "lucide-react";
import { api } from "../lib/api";
import { classNames, formatPoints, formatPrice } from "../lib/format";
import { useRequest } from "../hooks/useRequest";
import type { Metadata, Player, Position } from "../types";
import {
  EmptyState,
  ErrorState,
  LoadingState,
  PageHeading,
  PlayerBadge,
  PositionBadge,
  StatStrip,
  TeamBadge,
} from "../components/ui";
import { PlayerDrawer } from "../components/PlayerDrawer";

type SortKey = "total_xpts" | "current_xpts" | "price" | "selected_by" | "name";
type SortDirection = "asc" | "desc";

const positions: Array<Position | "ALL"> = ["ALL", "GK", "DEF", "MID", "FWD"];
const windowOptions = [1, 3, 5, 10, 38];

function firstGameweekPoints(player: Player, gameweek: number) {
  return player.future_points.find((item) => item.gameweek === gameweek)?.xpts ?? 0;
}

function fixtureLabel(player: Player, gameweek: number) {
  const fixtures = player.future_points.find((item) => item.gameweek === gameweek)?.fixture_projections;
  if (!fixtures?.length) return "No fixture";
  return fixtures
    .map((fixture) => `${fixture.opponent} ${fixture.is_home ? "H" : "A"}`)
    .join(" · ");
}

function SortButton({
  label,
  sortKey,
  activeKey,
  direction,
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  direction: SortDirection;
  onSort: (key: SortKey) => void;
}) {
  const active = sortKey === activeKey;
  return (
    <button className="inline-flex items-center gap-1.5 hover:text-stone-200" onClick={() => onSort(sortKey)}>
      {label}
      {active && (direction === "desc" ? <ArrowDown className="h-3 w-3 text-signal-350" /> : <ArrowUp className="h-3 w-3 text-signal-350" />)}
    </button>
  );
}

export function PlayersView({ metadata }: { metadata: Metadata }) {
  const availableGameweeks = metadata.coverage.gameweeks;
  const [startGameweek, setStartGameweek] = useState(availableGameweeks[0]);
  const [gameweeks, setGameweeks] = useState(Math.min(5, availableGameweeks.length));
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState<Position | "ALL">("ALL");
  const [team, setTeam] = useState("ALL");
  const [sortKey, setSortKey] = useState<SortKey>("total_xpts");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [visibleCount, setVisibleCount] = useState(50);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);

  const remainingGameweeks = availableGameweeks.filter((item) => item >= startGameweek).length;
  const effectiveWindow = Math.min(gameweeks, remainingGameweeks);
  const windowEnd = startGameweek + effectiveWindow - 1;
  const request = useRequest(
    (signal) =>
      api.players(
        {
          start_gameweek: startGameweek,
          gameweeks: effectiveWindow,
          detail: "summary",
          limit: 9999,
        },
        signal,
      ),
    [startGameweek, effectiveWindow],
  );

  const players = useMemo(() => request.data?.players ?? [], [request.data]);
  const teams = useMemo(
    () => [...new Set(players.map((player) => player.team))].sort(),
    [players],
  );
  const filteredPlayers = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return players
      .filter((player) => position === "ALL" || player.position === position)
      .filter((player) => team === "ALL" || player.team === team)
      .filter((player) => !needle || `${player.name} ${player.team}`.toLowerCase().includes(needle))
      .sort((left, right) => {
        let comparison = 0;
        if (sortKey === "name") comparison = left.name.localeCompare(right.name);
        if (sortKey === "total_xpts") comparison = left.total_xpts - right.total_xpts;
        if (sortKey === "current_xpts") comparison = firstGameweekPoints(left, startGameweek) - firstGameweekPoints(right, startGameweek);
        if (sortKey === "price") comparison = left.price - right.price;
        if (sortKey === "selected_by") comparison = left.selected_by - right.selected_by;
        return sortDirection === "desc" ? -comparison : comparison;
      });
  }, [players, position, search, sortDirection, sortKey, startGameweek, team]);

  useEffect(() => setVisibleCount(50), [position, search, sortDirection, sortKey, team]);

  const visiblePlayers = filteredPlayers.slice(0, visibleCount);
  const topPlayer = [...players].sort((a, b) => b.total_xpts - a.total_xpts)[0];
  const average = players.length
    ? players.reduce((total, player) => total + player.total_xpts, 0) / players.length
    : 0;

  function changeStartGameweek(value: number) {
    const remaining = availableGameweeks.filter((item) => item >= value).length;
    setStartGameweek(value);
    setGameweeks((current) => Math.min(current, remaining));
  }

  function changeSort(key: SortKey) {
    if (key === sortKey) {
      setSortDirection((current) => (current === "desc" ? "asc" : "desc"));
      return;
    }
    setSortKey(key);
    setSortDirection(key === "name" ? "asc" : "desc");
  }

  return (
    <>
      <PageHeading
        eyebrow={`Season ${metadata.season} / Player model`}
        title="Player projections"
        description="Rank the player pool, inspect each gameweek, then open a player for the simulation output behind the headline number."
        actions={
          <div className="flex items-center gap-2">
            <label>
              <span className="sr-only">Start gameweek</span>
              <select className="control pr-8" value={startGameweek} onChange={(event) => changeStartGameweek(Number(event.target.value))}>
                {availableGameweeks.map((item) => <option key={item} value={item}>From GW {item}</option>)}
              </select>
            </label>
            <label>
              <span className="sr-only">Projection window</span>
              <select className="control pr-8" value={effectiveWindow} onChange={(event) => setGameweeks(Number(event.target.value))}>
                {windowOptions.filter((item) => item <= remainingGameweeks).map((item) => (
                  <option key={item} value={item}>{item === remainingGameweeks ? `All ${item} GWs` : `${item} GW${item === 1 ? "" : "s"}`}</option>
                ))}
                {!windowOptions.includes(remainingGameweeks) && <option value={remainingGameweeks}>All {remainingGameweeks} GWs</option>}
              </select>
            </label>
          </div>
        }
      />

      {request.loading && !request.data ? (
        <LoadingState />
      ) : request.error ? (
        <ErrorState error={request.error} retry={request.retry} />
      ) : (
        <>
          <StatStrip
            items={[
              { label: "Projection window", value: `GW ${startGameweek}—${windowEnd}`, detail: `${effectiveWindow} rounds` },
              { label: "Player pool", value: players.length, detail: `${teams.length} teams` },
              { label: "Highest xPts", value: topPlayer ? formatPoints(topPlayer.total_xpts) : "—", detail: topPlayer?.name },
              { label: "Pool average", value: formatPoints(average), detail: "xPts / player" },
            ]}
          />

          <section className="px-4 py-5 sm:px-6 lg:px-8">
            <div className="border border-white/[0.08] bg-ink-900">
              <div className="flex flex-col gap-3 border-b border-white/[0.08] p-3 xl:flex-row xl:items-center xl:justify-between">
                <div className="flex flex-wrap gap-1.5" aria-label="Filter by position">
                  {positions.map((item) => (
                    <button
                      key={item}
                      className={classNames("h-8 border px-3 font-mono text-[10px] transition-colors", position === item ? "border-signal-350/50 bg-signal-450/15 text-signal-300" : "border-white/[0.08] text-stone-500 hover:border-white/20 hover:text-stone-300")}
                      onClick={() => setPosition(item)}
                    >
                      {item === "ALL" ? "All positions" : item}
                    </button>
                  ))}
                </div>
                <div className="flex min-w-0 flex-col gap-2 sm:flex-row">
                  <label className="relative min-w-0 sm:w-64">
                    <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-stone-600" />
                    <input className="field pl-9" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a player or team…" />
                  </label>
                  <label className="relative sm:w-40">
                    <SlidersHorizontal className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-stone-600" />
                    <select className="field pl-9" value={team} onChange={(event) => setTeam(event.target.value)}>
                      <option value="ALL">All teams</option>
                      {teams.map((item) => <option key={item}>{item}</option>)}
                    </select>
                  </label>
                </div>
              </div>

              {visiblePlayers.length ? (
                <>
                  <div className="hidden overflow-x-auto md:block">
                    <table className="w-full min-w-[880px] border-collapse text-left">
                      <thead>
                        <tr className="border-b border-white/[0.08] bg-black/10 font-mono text-[9px] uppercase tracking-[0.11em] text-stone-600">
                          <th className="w-12 px-3 py-3 text-center">#</th>
                          <th className="px-3 py-3"><SortButton label="Player" sortKey="name" activeKey={sortKey} direction={sortDirection} onSort={changeSort} /></th>
                          <th className="px-3 py-3">Team</th>
                          <th className="px-3 py-3"><SortButton label="Price" sortKey="price" activeKey={sortKey} direction={sortDirection} onSort={changeSort} /></th>
                          <th className="px-3 py-3"><SortButton label="Owned" sortKey="selected_by" activeKey={sortKey} direction={sortDirection} onSort={changeSort} /></th>
                          <th className="bg-signal-450/[0.035] px-3 py-3"><SortButton label={`GW${startGameweek} xPts`} sortKey="current_xpts" activeKey={sortKey} direction={sortDirection} onSort={changeSort} /></th>
                          <th className="px-3 py-3">Next fixture</th>
                          <th className="bg-signal-450/[0.07] px-3 py-3"><SortButton label={`${effectiveWindow}GW xPts`} sortKey="total_xpts" activeKey={sortKey} direction={sortDirection} onSort={changeSort} /></th>
                        </tr>
                      </thead>
                      <tbody>
                        {visiblePlayers.map((player, index) => (
                          <tr
                            key={player.id}
                            className="group cursor-pointer border-b border-white/[0.055] text-xs transition-colors last:border-0 hover:bg-signal-450/[0.045]"
                            onClick={() => setSelectedPlayer(player)}
                            onKeyDown={(event) => { if (event.key === "Enter") setSelectedPlayer(player); }}
                            tabIndex={0}
                          >
                            <td className="px-3 py-2.5 text-center font-mono text-[10px] tabular-nums text-stone-600">{index + 1}</td>
                            <td className="px-3 py-2.5">
                              <div className="flex min-w-[180px] items-center gap-3">
                                <PlayerBadge name={player.name} />
                                <div className="min-w-0">
                                  <p className="truncate font-medium text-stone-200 group-hover:text-white">{player.name}</p>
                                  <div className="mt-1"><PositionBadge position={player.position} /></div>
                                </div>
                              </div>
                            </td>
                            <td className="px-3 py-2.5"><div className="flex items-center gap-2"><TeamBadge team={player.team} size="sm" /><span className="font-mono text-[10px] text-stone-500">{player.team}</span></div></td>
                            <td className="px-3 py-2.5 font-mono tabular-nums text-stone-300">{formatPrice(player.price)}</td>
                            <td className="px-3 py-2.5 font-mono tabular-nums text-stone-400">{player.selected_by.toFixed(1)}%</td>
                            <td className="bg-signal-450/[0.02] px-3 py-2.5 font-mono text-sm font-medium tabular-nums text-signal-300">{formatPoints(firstGameweekPoints(player, startGameweek))}</td>
                            <td className="max-w-40 truncate px-3 py-2.5 font-mono text-[10px] text-stone-500">{fixtureLabel(player, startGameweek)}</td>
                            <td className="bg-signal-450/[0.045] px-3 py-2.5 font-mono text-sm font-semibold tabular-nums text-white">{formatPoints(player.total_xpts)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="divide-y divide-white/[0.06] md:hidden">
                    {visiblePlayers.map((player, index) => (
                      <button key={player.id} className="flex w-full items-center gap-3 p-3 text-left hover:bg-signal-450/[0.045]" onClick={() => setSelectedPlayer(player)}>
                        <span className="w-5 font-mono text-[9px] text-stone-600">{index + 1}</span>
                        <PlayerBadge name={player.name} />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-medium text-stone-200">{player.name}</span>
                          <span className="mt-1 flex items-center gap-2 text-[10px] text-stone-500"><PositionBadge position={player.position} /> {player.team} · {formatPrice(player.price)} · {fixtureLabel(player, startGameweek)}</span>
                        </span>
                        <span className="text-right">
                          <span className="block font-mono text-base font-semibold text-white">{formatPoints(player.total_xpts)}</span>
                          <span className="font-mono text-[9px] uppercase text-stone-600">{effectiveWindow}GW</span>
                        </span>
                      </button>
                    ))}
                  </div>

                  <div className="flex items-center justify-between border-t border-white/[0.08] px-3 py-3">
                    <p className="font-mono text-[9px] uppercase tracking-[0.12em] text-stone-600">
                      Showing {Math.min(visibleCount, filteredPlayers.length)} / {filteredPlayers.length}
                    </p>
                    {visibleCount < filteredPlayers.length && <button className="control" onClick={() => setVisibleCount((current) => current + 50)}>Load 50 more</button>}
                  </div>
                </>
              ) : (
                <div className="p-4"><EmptyState>No players match those filters.</EmptyState></div>
              )}
            </div>
          </section>
        </>
      )}

      {selectedPlayer && (
        <PlayerDrawer
          summary={selectedPlayer}
          startGameweek={startGameweek}
          gameweeks={effectiveWindow}
          onClose={() => setSelectedPlayer(null)}
        />
      )}
    </>
  );
}
