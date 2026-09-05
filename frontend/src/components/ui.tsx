import { useEffect, useId, type ReactNode } from "react";
import { AlertCircle, ChevronRight, RefreshCw, X } from "lucide-react";
import { classNames, formatPoints, humanise, initials } from "../lib/format";
import type { Position } from "../types";

export function BrandMark() {
  return (
    <div className="flex items-center" aria-label="xLthm">
      <img className="h-12 w-auto object-contain" src="/logo.png" alt="xLthm" />
    </div>
  );
}

export function TeamBadge({ team, size = "md" }: { team: string; size?: "sm" | "md" | "lg" }) {
  const sizes = {
    sm: "h-7 w-7 text-[9px]",
    md: "h-9 w-9 text-[10px]",
    lg: "h-14 w-14 text-xs",
  };
  return (
    <span
      className={classNames(
        "grid shrink-0 place-items-center border border-signal-350/25 bg-signal-450/[0.08] font-mono font-semibold tracking-tight text-signal-350",
        sizes[size],
      )}
      aria-label={team}
    >
      {team}
    </span>
  );
}

export function PlayerBadge({ name }: { name: string }) {
  return (
    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-white/10 bg-white/[0.045] font-mono text-[10px] font-medium text-stone-300">
      {initials(name)}
    </span>
  );
}

export function PositionBadge({ position }: { position: Position }) {
  const colours: Record<Position, string> = {
    GK: "border-amber-300/20 bg-amber-300/[0.07] text-amber-200",
    DEF: "border-sky-300/20 bg-sky-300/[0.07] text-sky-200",
    MID: "border-signal-350/20 bg-signal-350/[0.07] text-signal-300",
    FWD: "border-rose-300/20 bg-rose-300/[0.07] text-rose-200",
  };
  return (
    <span className={classNames("border px-1.5 py-0.5 font-mono text-[10px] font-medium", colours[position])}>
      {position}
    </span>
  );
}

export function PageHeading({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="flex flex-col gap-5 border-b border-white/[0.08] px-4 py-6 sm:px-6 lg:flex-row lg:items-end lg:justify-between lg:px-8 lg:py-7">
      <div>
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-signal-350">{eyebrow}</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-[-0.035em] text-white sm:text-3xl">{title}</h1>
        <p className="mt-1.5 max-w-2xl text-sm leading-6 text-stone-400">{description}</p>
      </div>
      {actions}
    </header>
  );
}

export function StatStrip({ items }: { items: Array<{ label: string; value: ReactNode; detail?: string }> }) {
  return (
    <div className="grid border-b border-white/[0.08] bg-white/[0.012] sm:grid-cols-2 lg:grid-cols-4">
      {items.map((item) => (
        <div key={item.label} className="border-b border-white/[0.08] px-4 py-4 last:border-0 sm:px-6 sm:[&:nth-last-child(-n+2)]:border-b-0 lg:border-b-0 lg:border-r lg:last:border-r-0">
          <p className="font-mono text-[9px] uppercase tracking-[0.17em] text-stone-500">{item.label}</p>
          <div className="mt-1.5 flex items-baseline gap-2">
            <span className="font-mono text-lg font-medium tabular-nums text-stone-100">{item.value}</span>
            {item.detail && <span className="text-[11px] text-stone-500">{item.detail}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <span className={classNames("block bg-white/[0.07]", className)} aria-hidden="true" />;
}

export function InlineLoadingSkeleton({ label = "Updating" }: { label?: string }) {
  return (
    <span className="hidden items-center gap-1.5 font-mono text-[9px] uppercase tracking-[0.1em] text-stone-600 sm:flex" role="status" aria-live="polite">
      <Skeleton className="h-2 w-7 animate-pulse motion-reduce:animate-none" />
      {label}
    </span>
  );
}

function SkeletonStats() {
  return (
    <div className="grid border-b border-white/[0.08] bg-white/[0.012] sm:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 4 }, (_, index) => (
        <div key={index} className="border-b border-white/[0.08] px-4 py-4 last:border-0 sm:px-6 sm:[&:nth-last-child(-n+2)]:border-b-0 lg:border-b-0 lg:border-r lg:last:border-r-0">
          <Skeleton className="h-2 w-20" />
          <div className="mt-2 flex items-center gap-2">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-2.5 w-20" />
          </div>
        </div>
      ))}
    </div>
  );
}

function TableSkeleton() {
  return (
    <section className="px-4 py-5 sm:px-6 lg:px-8">
      <div className="border border-white/[0.08] bg-ink-900">
        <div className="flex flex-col gap-3 border-b border-white/[0.08] p-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex gap-1.5">
            {["w-16", "w-10", "w-11", "w-11"].map((width, index) => <Skeleton key={index} className={classNames("h-8", width)} />)}
          </div>
          <div className="flex gap-2"><Skeleton className="h-9 w-full sm:w-64" /><Skeleton className="h-9 w-36" /></div>
        </div>
        <div className="divide-y divide-white/[0.06]">
          {Array.from({ length: 7 }, (_, index) => (
            <div key={index} className="flex items-center gap-3 px-3 py-2.5">
              <Skeleton className="h-9 w-9 shrink-0 rounded-full" />
              <div className="min-w-0 flex-1"><Skeleton className="h-3 w-32 max-w-full" /><Skeleton className="mt-2 h-2 w-16" /></div>
              <Skeleton className="hidden h-7 w-7 sm:block" />
              <Skeleton className="hidden h-3 w-14 sm:block" />
              <Skeleton className="h-4 w-12" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CardSkeleton() {
  return (
    <section className="px-4 py-5 sm:px-6 lg:px-8">
      <div className="mb-5 flex items-end justify-between gap-4">
        <div><Skeleton className="h-2 w-28" /><Skeleton className="mt-2 h-3 w-64 max-w-full" /></div>
        <Skeleton className="hidden h-9 w-64 sm:block" />
      </div>
      <div className="mb-3 flex items-center gap-3"><Skeleton className="h-3 w-12" /><Skeleton className="h-px flex-1" /><Skeleton className="h-2 w-16" /></div>
      <div className="grid gap-2 xl:grid-cols-2">
        {Array.from({ length: 4 }, (_, index) => (
          <div key={index} className="border border-white/[0.08] bg-ink-900 p-4">
            <div className="flex justify-between"><Skeleton className="h-2 w-28" /><Skeleton className="h-2 w-8" /></div>
            <div className="mt-5 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2"><Skeleton className="h-9 w-9" /><Skeleton className="h-3 w-16" /></div>
              <Skeleton className="h-6 w-14" />
              <div className="flex items-center gap-2"><Skeleton className="h-3 w-16" /><Skeleton className="h-9 w-9" /></div>
            </div>
            <Skeleton className="mt-5 h-1.5 w-full" />
            <div className="mt-2 flex justify-between"><Skeleton className="h-2 w-10" /><Skeleton className="h-2 w-20" /><Skeleton className="h-2 w-10" /></div>
          </div>
        ))}
      </div>
    </section>
  );
}

function DrawerSkeleton() {
  return (
    <div>
      <div className="flex items-center gap-4 border-b border-white/[0.08] px-5 py-5 sm:px-6">
        <Skeleton className="h-9 w-9 rounded-full" />
        <Skeleton className="h-14 w-14" />
        <div className="flex-1"><Skeleton className="h-3 w-24" /><Skeleton className="mt-2 h-2.5 w-40 max-w-full" /></div>
        <div><Skeleton className="ml-auto h-7 w-16" /><Skeleton className="mt-2 h-2 w-12" /></div>
      </div>
      <div className="grid grid-cols-3 border-b border-white/[0.08]">
        {Array.from({ length: 3 }, (_, index) => <div key={index} className="border-r border-white/[0.08] px-3 py-3 last:border-0"><Skeleton className="mx-auto h-4 w-14" /><Skeleton className="mx-auto mt-2 h-2 w-16" /></div>)}
      </div>
      <div className="flex gap-6 border-b border-white/[0.08] px-5 py-3 sm:px-6">
        <Skeleton className="h-3 w-16" /><Skeleton className="h-3 w-20" /><Skeleton className="h-3 w-12" />
      </div>
      <div className="space-y-6 px-5 py-6 sm:px-6">
        <div><Skeleton className="h-2 w-20" /><div className="mt-3 flex gap-1">{Array.from({ length: 5 }, (_, index) => <Skeleton key={index} className="h-12 w-16" />)}</div></div>
        <div><Skeleton className="h-2 w-28" /><Skeleton className="mt-4 h-40 w-full" /></div>
        <div><Skeleton className="h-2 w-36" /><Skeleton className="mt-4 h-28 w-full" /></div>
      </div>
    </div>
  );
}

export function LoadingSkeleton({
  label = "Loading projection data",
  variant = "table",
  withHeading = false,
}: {
  label?: string;
  variant?: "table" | "cards" | "drawer";
  withHeading?: boolean;
}) {
  return (
    <div className="animate-pulse motion-reduce:animate-none" role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      {withHeading && (
        <header className="flex flex-col gap-5 border-b border-white/[0.08] px-4 py-6 sm:px-6 lg:flex-row lg:items-end lg:justify-between lg:px-8 lg:py-7">
          <div className="w-full"><Skeleton className="h-2 w-28" /><Skeleton className="mt-3 h-8 w-64 max-w-full" /><Skeleton className="mt-3 h-3 w-[34rem] max-w-full" /></div>
          <div className="flex gap-2"><Skeleton className="h-9 w-28" /><Skeleton className="h-9 w-24" /></div>
        </header>
      )}
      {variant === "drawer" ? <DrawerSkeleton /> : <><SkeletonStats />{variant === "cards" ? <CardSkeleton /> : <TableSkeleton />}</>}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: Error; retry: () => void }) {
  return (
    <div className="grid min-h-[360px] place-items-center px-6">
      <div className="max-w-md border border-rose-300/20 bg-rose-300/[0.035] p-6 text-center">
        <AlertCircle className="mx-auto h-5 w-5 text-rose-300" />
        <p className="mt-3 text-sm font-medium text-stone-100">The projection API did not respond</p>
        <p className="mt-1.5 text-xs leading-5 text-stone-500">{error.message}</p>
        <button className="control mt-5 inline-flex items-center gap-2" onClick={retry}>
          <RefreshCw className="h-3.5 w-3.5" /> Try again
        </button>
      </div>
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="grid min-h-[280px] place-items-center border border-dashed border-white/10 px-6 text-center text-sm text-stone-500">
      {children}
    </div>
  );
}

export function Drawer({
  open,
  title,
  subtitle,
  onClose,
  children,
  wide = false,
}: {
  open: boolean;
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose, open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <button className="absolute inset-0 cursor-default bg-[rgba(0,0,0,0.65)] backdrop-blur-[2px]" onClick={onClose} aria-label="Close panel" />
      <section className={classNames("relative flex h-full w-full flex-col border-l border-white/10 bg-ink-900 shadow-drawer", wide ? "max-w-3xl" : "max-w-xl")}>
        <header className="flex shrink-0 items-start justify-between gap-4 border-b border-white/10 px-5 py-4 sm:px-6">
          <div className="min-w-0">
            <h2 id={titleId} className="truncate text-lg font-semibold tracking-tight text-white">{title}</h2>
            {subtitle && <p className="mt-0.5 truncate text-xs text-stone-500">{subtitle}</p>}
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close panel">
            <X className="h-4 w-4" />
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
      </section>
    </div>
  );
}

export function ProbabilityBar({
  label,
  value,
  displayValue,
  tone = "signal",
}: {
  label: string;
  value: number;
  displayValue?: string;
  tone?: "signal" | "red" | "stone";
}) {
  const colours = {
    signal: "bg-signal-450",
    red: "bg-rose-400",
    stone: "bg-stone-500",
  };
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-4 text-xs">
        <span className="truncate text-stone-400">{label}</span>
        <span className="font-mono tabular-nums text-stone-200">{displayValue ?? `${Math.round(value * 100)}%`}</span>
      </div>
      <div className="h-1 overflow-hidden bg-white/[0.07]">
        <div className={classNames("h-full", colours[tone])} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
      </div>
    </div>
  );
}

export function PointsChart({ values }: { values: Array<{ label: string; value: number }> }) {
  if (!values.length) return null;
  const width = 500;
  const height = 180;
  const padding = { top: 15, right: 10, bottom: 28, left: 26 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const minimum = Math.min(0, ...values.map((item) => item.value));
  const maximum = Math.max(1, ...values.map((item) => item.value));
  const range = maximum - minimum || 1;
  const points = values.map((item, index) => ({
    ...item,
    x: padding.left + (values.length === 1 ? chartWidth / 2 : (index / (values.length - 1)) * chartWidth),
    y: padding.top + chartHeight - ((item.value - minimum) / range) * chartHeight,
  }));
  const line = points.map((point) => `${point.x},${point.y}`).join(" ");
  const area = `${padding.left},${padding.top + chartHeight} ${line} ${padding.left + chartWidth},${padding.top + chartHeight}`;

  return (
    <svg className="h-auto w-full overflow-visible" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Expected points by gameweek">
      {[0, 0.5, 1].map((position) => {
        const y = padding.top + chartHeight * position;
        const value = maximum - range * position;
        return (
          <g key={position}>
            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} stroke="rgba(255,255,255,.07)" />
            <text x={0} y={y + 3} fill="#78716c" fontSize="9" fontFamily="JetBrains Mono Variable">{formatPoints(value)}</text>
          </g>
        );
      })}
      <polygon points={area} fill="rgba(77, 204, 136, .08)" />
      <polyline points={line} fill="none" stroke="#7ce3ad" strokeWidth="2" strokeLinejoin="round" />
      {points.map((point) => (
        <g key={point.label}>
          <circle cx={point.x} cy={point.y} r="3.5" fill="#0e0c13" stroke="#7ce3ad" strokeWidth="2" />
          <text x={point.x} y={height - 7} fill="#78716c" fontSize="9" textAnchor="middle" fontFamily="JetBrains Mono Variable">{point.label}</text>
        </g>
      ))}
    </svg>
  );
}

export function JsonPanel({ data, label = "Raw API response" }: { data: unknown; label?: string }) {
  return (
    <details className="group border border-white/[0.08] bg-black/10">
      <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-xs font-medium text-stone-300 hover:text-white">
        {label}
        <ChevronRight className="h-4 w-4 text-stone-600 transition-transform group-open:rotate-90" />
      </summary>
      <pre className="max-h-[440px] overflow-auto border-t border-white/[0.08] p-4 font-mono text-[10px] leading-5 text-stone-400">
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  );
}

export function KeyValueGrid({ values }: { values: Record<string, number | string | null | undefined> }) {
  return (
    <div className="grid grid-cols-2 border-l border-t border-white/[0.08] sm:grid-cols-3">
      {Object.entries(values).map(([key, value]) => (
        <div key={key} className="border-b border-r border-white/[0.08] px-3 py-3">
          <p className="truncate text-[10px] text-stone-500" title={humanise(key)}>{humanise(key)}</p>
          <p className="mt-1 font-mono text-sm tabular-nums text-stone-200">
            {typeof value === "number" ? value.toFixed(3).replace(/\.0+$/, "") : value ?? "—"}
          </p>
        </div>
      ))}
    </div>
  );
}
