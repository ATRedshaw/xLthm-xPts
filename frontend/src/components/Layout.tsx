import { Activity, CalendarDays, FlaskConical, UsersRound } from "lucide-react";
import { classNames, formatDateTime, formatRelativeTime } from "../lib/format";
import type { Metadata, ViewName } from "../types";
import { BrandMark } from "./ui";

const navigation: Array<{
  id: ViewName;
  label: string;
  shortLabel: string;
  icon: typeof UsersRound;
}> = [
  { id: "players", label: "Player projections", shortLabel: "Players", icon: UsersRound },
  { id: "fixtures", label: "Fixture forecasts", shortLabel: "Fixtures", icon: CalendarDays },
  { id: "model", label: "Model & API", shortLabel: "Model", icon: FlaskConical },
];

export function Layout({
  activeView,
  onNavigate,
  metadata,
  healthy,
  children,
}: {
  activeView: ViewName;
  onNavigate: (view: ViewName) => void;
  metadata: Metadata | null;
  healthy: boolean | null;
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-ink-950 text-stone-200">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col border-r border-white/[0.08] bg-ink-900 lg:flex">
        <div className="flex h-16 items-center border-b border-white/[0.08] px-5">
          <BrandMark />
        </div>
        <nav className="flex-1 px-3 py-5" aria-label="Primary navigation">
          <p className="px-3 font-mono text-[9px] uppercase tracking-[0.18em] text-stone-600">Explore</p>
          <div className="mt-3 space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              const active = activeView === item.id;
              return (
                <button
                  key={item.id}
                  className={classNames(
                    "group flex w-full items-center gap-3 border-l-2 px-3 py-2.5 text-left text-sm transition-colors",
                    active
                      ? "border-signal-350 bg-signal-450/[0.08] text-white"
                      : "border-transparent text-stone-500 hover:bg-white/[0.025] hover:text-stone-200",
                  )}
                  onClick={() => onNavigate(item.id)}
                >
                  <Icon className={classNames("h-4 w-4", active ? "text-signal-350" : "text-stone-600 group-hover:text-stone-400")} />
                  {item.label}
                </button>
              );
            })}
          </div>
        </nav>
        <div className="border-t border-white/[0.08] p-5">
          <div className="flex items-center gap-2">
            <span className={classNames("h-1.5 w-1.5 rounded-full", healthy === true ? "bg-emerald-400" : healthy === false ? "bg-rose-400" : "bg-stone-600")} />
            <span className="font-mono text-[9px] uppercase tracking-[0.13em] text-stone-500">
              {healthy === true ? "API connected" : healthy === false ? "API offline" : "Checking API"}
            </span>
          </div>
          {metadata && (
            <p className="mt-3 text-[10px] leading-4 text-stone-600">
              Season {metadata.season}<br />
              Batch {formatRelativeTime(metadata.generated_at)}
            </p>
          )}
        </div>
      </aside>

      <div className="lg:pl-60">
        <header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-white/[0.08] bg-ink-950/95 px-4 backdrop-blur sm:px-6 lg:h-16 lg:px-8">
          <div className="lg:hidden"><BrandMark /></div>
          <div className="hidden items-center gap-2 text-xs text-stone-500 lg:flex">
            <Activity className="h-3.5 w-3.5 text-signal-350" />
            <span>Projection workbench</span>
            <span className="text-stone-700">/</span>
            <span className="text-stone-300">{navigation.find((item) => item.id === activeView)?.shortLabel}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className={classNames("h-1.5 w-1.5 rounded-full", healthy === true ? "bg-emerald-400" : healthy === false ? "bg-rose-400" : "bg-stone-600")} />
            <div className="text-right">
              <p className="text-[10px] text-stone-400 sm:text-xs">{healthy === false ? "API unavailable" : "Projection batch"}</p>
              {metadata && <p className="hidden font-mono text-[9px] text-stone-600 sm:block">{formatDateTime(metadata.generated_at)}</p>}
            </div>
          </div>
        </header>

        <main className="min-h-[calc(100vh-3.5rem)] pb-20 lg:min-h-[calc(100vh-4rem)] lg:pb-0">{children}</main>
      </div>

      <nav className="fixed inset-x-0 bottom-0 z-40 grid grid-cols-3 border-t border-white/10 bg-ink-900/95 px-2 pb-[max(0.4rem,env(safe-area-inset-bottom))] pt-1.5 backdrop-blur lg:hidden" aria-label="Primary navigation">
        {navigation.map((item) => {
          const Icon = item.icon;
          const active = activeView === item.id;
          return (
            <button key={item.id} className={classNames("flex flex-col items-center gap-1 py-1.5 text-[10px]", active ? "text-signal-350" : "text-stone-600")} onClick={() => onNavigate(item.id)}>
              <Icon className="h-4 w-4" />
              {item.shortLabel}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
