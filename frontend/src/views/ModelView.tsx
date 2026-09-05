import { AlertTriangle, CheckCircle2, Copy, Database, FlaskConical, Route } from "lucide-react";
import { api } from "../lib/api";
import { formatDate, formatDateTime, humanise } from "../lib/format";
import { useRequest } from "../hooks/useRequest";
import type { Metadata } from "../types";
import { ErrorState, JsonPanel, LoadingState, PageHeading, StatStrip } from "../components/ui";

function copyText(value: string) {
  void navigator.clipboard?.writeText(value);
}

export function ModelView({ metadata }: { metadata: Metadata }) {
  const request = useRequest((signal) => api.directory(signal), []);
  const gameweeks = metadata.coverage.gameweeks;
  const firstGameweek = gameweeks[0];
  const lastGameweek = gameweeks[gameweeks.length - 1];

  return (
    <>
      <PageHeading
        eyebrow={`Season ${metadata.season} / Inference record`}
        title="Model & API"
        description="Audit the current inference batch, its component models and quality checks, then inspect the API contract that powers this workbench."
      />
      <StatStrip items={[
        { label: "Simulation size", value: metadata.simulation_count.toLocaleString("en-GB"), detail: "outcomes / fixture" },
        { label: "Coverage", value: `${metadata.coverage.players}`, detail: "players" },
        { label: "Fixture set", value: `${metadata.coverage.fixtures}`, detail: "fixtures" },
        { label: "Projection span", value: `GW ${firstGameweek}—${lastGameweek}`, detail: `${gameweeks.length} rounds` },
      ]} />

      <div className="grid gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(280px,.7fr)] lg:px-8">
        <div className="space-y-5">
          <section className="border border-white/[0.08] bg-ink-900">
            <div className="flex items-center gap-3 border-b border-white/[0.08] px-4 py-3">
              <Database className="h-4 w-4 text-violet-350" />
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
              <FlaskConical className="h-4 w-4 text-violet-350" />
              <div><h2 className="text-sm font-medium text-stone-100">Component registry</h2><p className="mt-0.5 text-[10px] text-stone-600">Artifacts used in the current projection run</p></div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[620px] text-left">
                <thead><tr className="border-b border-white/[0.08] bg-black/10 font-mono text-[8px] uppercase tracking-[0.11em] text-stone-600"><th className="px-4 py-2.5">Model</th><th className="px-4 py-2.5">Version</th><th className="px-4 py-2.5">Feature profile</th><th className="px-4 py-2.5">Trained</th></tr></thead>
                <tbody>
                  {Object.entries(metadata.models).map(([name, model]) => (
                    <tr key={name} className="border-b border-white/[0.055] last:border-0">
                      <td className="px-4 py-3"><span className="font-mono text-xs text-stone-200">{humanise(name)}</span><span className="ml-2 text-[9px] text-stone-700">{model.model_type}</span></td>
                      <td className="px-4 py-3 font-mono text-[10px] text-violet-300">v{model.artifact_version}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-stone-500">{model.feature_profile || "—"}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-stone-500">{formatDate(model.trained_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="border border-white/[0.08] bg-ink-900">
            <div className="flex items-center gap-3 border-b border-white/[0.08] px-4 py-3">
              <Route className="h-4 w-4 text-violet-350" />
              <div><h2 className="text-sm font-medium text-stone-100">Methodology</h2><p className="mt-0.5 text-[10px] text-stone-600">How one simulated match becomes player points</p></div>
            </div>
            <ol className="divide-y divide-white/[0.06]">
              {Object.entries(metadata.methodology).map(([stage, description], index) => (
                <li key={stage} className="grid grid-cols-[2rem_1fr] gap-3 px-4 py-4">
                  <span className="font-mono text-xs text-violet-350">{String(index + 1).padStart(2, "0")}</span>
                  <div><h3 className="text-xs font-medium text-stone-200">{humanise(stage)}</h3><p className="mt-1 text-xs leading-5 text-stone-500">{description}</p></div>
                </li>
              ))}
            </ol>
          </section>

          {request.loading && !request.data ? <LoadingState label="Loading API contract" /> : request.error ? <ErrorState error={request.error} retry={request.retry} /> : request.data && (
            <section className="border border-white/[0.08] bg-ink-900">
              <div className="flex items-center gap-3 border-b border-white/[0.08] px-4 py-3">
                <Route className="h-4 w-4 text-violet-350" />
                <div><h2 className="text-sm font-medium text-stone-100">Endpoint index</h2><p className="mt-0.5 text-[10px] text-stone-600">{request.data.name} / {request.data.version}</p></div>
              </div>
              <div className="divide-y divide-white/[0.06]">
                {Object.entries(request.data.endpoints).map(([path, endpoint]) => (
                  <div key={path} className="grid gap-2 px-4 py-4 sm:grid-cols-[minmax(15rem,.8fr)_1.2fr]">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="border border-violet-350/25 bg-violet-450/[0.07] px-1.5 py-0.5 font-mono text-[8px] text-violet-300">{endpoint.method}</span>
                      <code className="truncate font-mono text-[10px] text-stone-300">{path}</code>
                      <button className="text-stone-700 hover:text-stone-300" onClick={() => copyText(path)} aria-label={`Copy ${path}`}><Copy className="h-3 w-3" /></button>
                    </div>
                    <div><p className="text-xs leading-5 text-stone-500">{endpoint.description}</p>{endpoint.parameters.length > 0 && <p className="mt-1.5 font-mono text-[8px] leading-4 text-stone-700">{endpoint.parameters.join(" · ")}</p>}</div>
                  </div>
                ))}
              </div>
              <div className="p-4"><JsonPanel data={request.data} label="Complete API contract" /></div>
            </section>
          )}
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

          <section className="border border-amber-300/15 bg-amber-300/[0.025]">
            <div className="flex items-center gap-3 border-b border-amber-300/10 px-4 py-3"><AlertTriangle className="h-4 w-4 text-amber-300" /><h2 className="text-sm font-medium text-stone-100">Known limitations</h2></div>
            <ul className="divide-y divide-amber-300/[0.07]">
              {metadata.limitations.map((limitation, index) => <li key={limitation} className="grid grid-cols-[1.5rem_1fr] gap-2 px-4 py-4 text-xs leading-5 text-stone-500"><span className="font-mono text-[9px] text-amber-300/60">0{index + 1}</span>{limitation}</li>)}
            </ul>
          </section>

          <JsonPanel data={metadata} label="Raw batch metadata" />
        </aside>
      </div>
    </>
  );
}
