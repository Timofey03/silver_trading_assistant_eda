/**
 * /methodology — эволюция модели E1 → Ensemble + объяснение каждого шага.
 */
import { api, type EvolutionResponse } from "@/lib/api";
import { TrendingUp, AlertTriangle, Zap, Award } from "lucide-react";

export const revalidate = 300;

async function safeApi<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try { return await fn(); } catch { return fallback; }
}

const STAGE_META: Record<string, { color: string; icon: any; label: string }> = {
  baseline:    { color: "#71717a", icon: TrendingUp,    label: "BASELINE" },
  feature_eng: { color: "#a78bfa", icon: Zap,           label: "FEATURE ENGINEERING" },
  macro:       { color: "#60a5fa", icon: Zap,           label: "MACRO FACTORS" },
  artifact:    { color: "#f59e0b", icon: AlertTriangle, label: "MEASUREMENT ARTIFACT" },
  advanced:    { color: "#71717a", icon: Zap,           label: "ADVANCED ML" },
  final:       { color: "#10b981", icon: Award,         label: "FINAL ENSEMBLE" },
};


export default async function MethodologyPage() {
  const data = await safeApi(api.evolution, {
    experiments: [], best_sharpe: "", best_return: "",
  } as EvolutionResponse);

  const avail = data.experiments.filter((e) => e.available);

  return (
    <div className="space-y-12">
      <div className="space-y-2">
        <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          Methodology · от E1 baseline до finальный ensemble
        </div>
        <h1 className="text-3xl font-medium tracking-tight">
          Эволюция модели
        </h1>
        <p className="text-sm text-[var(--text-muted)] max-w-2xl">
          Каждая итерация — отдельный walk-forward backtest на 11 годах данных
          (2015-2026). Метрики получены при идентичных costs (0.1% round-trip)
          и параметрах симулятора (trail 20%, max hold 60 дней).
        </p>
      </div>

      {/* Final highlight */}
      <FinalCard experiments={data.experiments} bestSharpe={data.best_sharpe} />

      {/* Evolution chart */}
      <EvolutionTimeline experiments={data.experiments} />

      {/* Detailed table */}
      <ExperimentsTable experiments={data.experiments} bestSharpe={data.best_sharpe} />
    </div>
  );
}

function FinalCard({
  experiments, bestSharpe,
}: { experiments: EvolutionResponse["experiments"]; bestSharpe: string }) {
  const final = experiments.find((e) => e.id === bestSharpe) || experiments[experiments.length - 1];
  if (!final) return null;

  return (
    <section className="rounded-3xl border border-emerald-500/20 bg-gradient-to-b from-emerald-500/[0.06] via-transparent to-transparent bg-[var(--bg-elevated)] p-10">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-emerald-400/80">
        <Award className="h-3 w-3" /> Финальная модель
      </div>
      <h2 className="mt-2 text-2xl font-medium tracking-tight">{final.name}</h2>
      <p className="mt-2 text-sm text-[var(--text-muted)] max-w-2xl">
        {final.description}
      </p>
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-6">
        <BigStat label="Sharpe Ratio" value={final.sharpe.toFixed(2)} color="#10b981" />
        <BigStat label="Total Return" value={`${final.total_return >= 0 ? "+" : ""}${(final.total_return * 100).toFixed(0)}%`}
                 color={final.total_return >= 0 ? "#10b981" : "#f43f5e"} />
        <BigStat label="Win Rate" value={`${(final.win_rate * 100).toFixed(0)}%`} color="#fafafa" />
        <BigStat label="Max DD" value={`${(final.max_dd * 100).toFixed(1)}%`} color="#f43f5e" />
      </div>
    </section>
  );
}

function BigStat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">{label}</div>
      <div
        className="mt-2 font-[family-name:var(--font-mono)] text-3xl font-medium tabular-nums"
        style={{ color }}
      >
        {value}
      </div>
    </div>
  );
}

function EvolutionTimeline({ experiments }: { experiments: EvolutionResponse["experiments"] }) {
  const avail = experiments.filter((e) => e.available);
  if (!avail.length) return null;

  const maxSharpe = Math.max(...avail.map((e) => Math.abs(e.sharpe)));
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-medium tracking-tight">Sharpe Ratio через эксперименты</h2>
      <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 space-y-3">
        {avail.map((e) => {
          const meta = STAGE_META[e.stage] || STAGE_META.advanced;
          const width = Math.abs(e.sharpe) / maxSharpe * 100;
          const isNeg = e.sharpe < 0;
          return (
            <div key={e.id} className="space-y-1">
              <div className="flex items-center justify-between text-xs font-[family-name:var(--font-mono)]">
                <span className="text-[var(--text-secondary)]">{e.name}</span>
                <span className="tabular-nums" style={{ color: meta.color }}>
                  Sharpe {e.sharpe.toFixed(3)}
                </span>
              </div>
              <div className="h-2 w-full rounded bg-[var(--bg-subtle)] overflow-hidden relative">
                <div
                  className="h-full transition-all duration-700"
                  style={{
                    width: `${width}%`,
                    backgroundColor: isNeg ? "#f43f5e" : meta.color,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ExperimentsTable({
  experiments, bestSharpe,
}: { experiments: EvolutionResponse["experiments"]; bestSharpe: string }) {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-medium tracking-tight">Подробно по каждой итерации</h2>
      <div className="space-y-3">
        {experiments.map((e) => {
          const meta = STAGE_META[e.stage] || STAGE_META.advanced;
          const Icon = meta.icon;
          const isWinner = e.id === bestSharpe;
          const isArtifact = e.stage === "artifact";

          return (
            <div
              key={e.id}
              className={`rounded-2xl border bg-[var(--bg-elevated)] px-6 py-5 transition-colors ${
                isWinner
                  ? "border-emerald-500/30 bg-gradient-to-br from-emerald-500/[0.03] to-transparent"
                  : isArtifact
                  ? "border-amber-500/20"
                  : "border-[var(--border)]"
              }`}
            >
              <div className="flex items-start gap-4">
                <div
                  className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
                  style={{ backgroundColor: `${meta.color}15`, color: meta.color }}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest" style={{ color: meta.color }}>
                    {meta.label}
                    {isWinner && <span className="text-emerald-400">· best Sharpe</span>}
                    {isArtifact && <span className="text-amber-400">· honest disclosure</span>}
                  </div>
                  <div className="mt-1 flex items-baseline justify-between gap-4 flex-wrap">
                    <h3 className="text-base font-medium tracking-tight text-[var(--text-primary)]">
                      {e.name}
                    </h3>
                    {e.available && (
                      <div className="flex items-center gap-4 text-xs font-[family-name:var(--font-mono)] tabular-nums">
                        <span style={{ color: e.sharpe >= 0.5 ? "#10b981" : e.sharpe >= 0 ? "#fafafa" : "#f43f5e" }}>
                          Sharpe {e.sharpe.toFixed(3)}
                        </span>
                        <span style={{ color: e.total_return > 0 ? "#10b981" : "#f43f5e" }}>
                          {e.total_return >= 0 ? "+" : ""}{(e.total_return * 100).toFixed(0)}%
                        </span>
                        <span className="text-[var(--text-muted)]">
                          Win {(e.win_rate * 100).toFixed(0)}% · DD {(e.max_dd * 100).toFixed(1)}% · N {e.n_trades}
                        </span>
                      </div>
                    )}
                  </div>
                  <p className="mt-2 text-sm text-[var(--text-muted)] leading-relaxed">
                    {e.description}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
