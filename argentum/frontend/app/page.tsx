/**
 * Главная "/" — Hero (BUY/HOLD/SELL) + цена + explanation.
 */
import {
  api,
  type SignalResponse,
  type PriceResponse,
  type ExplainResponse,
  type FxRates,
} from "@/lib/api";
import { formatPct, formatUsd, formatRub } from "@/lib/utils";
import { Sparkles } from "lucide-react";
import HeroSignal from "@/components/HeroSignal";
import PositionSummaryCard from "@/components/PositionSummaryCard";
import PositionCalculator from "@/components/PositionCalculator";
import FeatureContribution from "@/components/FeatureContribution";

export const revalidate = 60;

async function safeApi<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try { return await fn(); } catch { return fallback; }
}

export default async function HomePage() {
  const [signal, price, explain, fx] = await Promise.all([
    safeApi(api.signal, {
      signal: "HOLD", date: "—", close: 0, p_up: 0,
      entry_threshold: 0.48, exit_threshold: 0.35,
      trail_pct: 0.12, max_hold_days: 30, cooldown_days: 25,
      source: "offline",
    } as SignalResponse),
    safeApi(api.price, {
      current: 0, previous: 0, change_pct: 0,
      currency: "USD", ticker: "SI=F",
      sparkline: [], last_update: "—",
    } as PriceResponse),
    safeApi(api.explain, {
      insights: [], model_version: "E3b", last_updated: "—",
    } as ExplainResponse),
    safeApi(api.fx, {
      usd_silver: 0, usdrub: 0, rub_silver: 0,
      usdrub_change_5d_pct: 0, fx_volatility_flag: false,
      source: "offline", last_update: "—",
    } as FxRates),
  ]);

  return (
    <div className="space-y-10">
      <HeroSignal signal={signal} />
      <PositionSummaryCard />
      <PriceCard price={price} fx={fx} />
      <FeatureContribution />
      <PositionCalculator />
      <ExplainSection explain={explain} signal={signal} />
    </div>
  );
}

// ============================================================================
// PRICE CARD — текущая цена + SVG sparkline
// ============================================================================
function PriceCard({ price, fx }: { price: PriceResponse; fx: FxRates }) {
  const isUp = price.change_pct > 0;
  const spark = sparkPathFromPoints(price.sparkline);
  const color = isUp ? "#10b981" : "#f43f5e";

  return (
    <section className="grid gap-8 md:grid-cols-[1fr_2fr] items-center rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] px-8 py-7 hover:border-[var(--text-faint)]/30 transition-colors">
      <div className="space-y-1.5">
        <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          {price.ticker} · Silver Futures
        </div>
        <div className="font-[family-name:var(--font-mono)] text-4xl font-medium tracking-tight">
          {formatUsd(price.current)}
        </div>
        <div
          className="inline-flex items-center gap-1 text-sm font-medium font-[family-name:var(--font-mono)]"
          style={{ color }}
        >
          <span>{isUp ? "▲" : "▼"}</span>
          {formatPct(price.change_pct)}
        </div>
        {fx.usdrub > 0 && (
          <div className="pt-2 mt-2 border-t border-[var(--border-soft)] space-y-0.5">
            <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
              эквивалент в ₽ (для Tinkoff)
            </div>
            <div className="font-[family-name:var(--font-mono)] text-base text-[var(--text-secondary)] tabular-nums">
              {formatRub(fx.rub_silver)} / oz
            </div>
            <div className="text-[11px] text-[var(--text-faint)] font-[family-name:var(--font-mono)]">
              USDRUB ₽{fx.usdrub.toFixed(2)}{" "}
              <span className={fx.usdrub_change_5d_pct >= 0 ? "text-emerald-400/70" : "text-rose-400/70"}>
                {fx.usdrub_change_5d_pct >= 0 ? "+" : ""}{fx.usdrub_change_5d_pct.toFixed(2)}% 5d
              </span>
              {fx.fx_volatility_flag && (
                <span className="ml-2 text-amber-400/80">⚠ FX-шум</span>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="h-24 relative">
        {price.sparkline.length > 1 && (
          <svg viewBox="0 0 200 50" preserveAspectRatio="none" className="h-full w-full">
            <defs>
              <linearGradient id="sparkGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor={color} stopOpacity="0.25" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={spark.area} fill="url(#sparkGrad)" />
            <path
              d={spark.line}
              fill="none"
              stroke={color}
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        )}
      </div>
    </section>
  );
}

function sparkPathFromPoints(points: { close: number }[]) {
  if (points.length < 2) return { line: "", area: "" };
  const values = points.map((p) => p.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const xStep = 200 / (points.length - 1);
  const coords = points.map((p, i) => {
    const x = i * xStep;
    const y = 50 - ((p.close - min) / range) * 46 - 2;
    return [x, y];
  });
  const linePath = "M " + coords.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" L ");
  const areaPath = linePath + ` L 200,50 L 0,50 Z`;
  return { line: linePath, area: areaPath };
}

// ============================================================================
// EXPLAIN — feature importance "почему BUY?"
// ============================================================================
function ExplainSection({ explain, signal }: { explain: ExplainResponse; signal: SignalResponse }) {
  if (!explain.insights.length) return null;

  const sigLabel = signal.signal === "BUY" ? "BUY" : signal.signal === "SELL" ? "SELL" : "HOLD";

  return (
    <section className="space-y-6">
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          <Sparkles className="h-3 w-3" />
          <span>Почему {sigLabel}</span>
        </div>
        <h2 className="text-xl font-medium tracking-tight">
          Что видит модель в данных
        </h2>
        <p className="text-sm text-[var(--text-muted)]">
          Из {signal.n_features_used} признаков модель выбрала эти как самые информативные
        </p>
      </div>

      <ul className="divide-y divide-[var(--border-soft)] rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] overflow-hidden">
        {explain.insights.map((insight, idx) => (
          <li
            key={insight.feature}
            className="group px-6 py-4 flex items-center gap-5 hover:bg-[var(--bg-subtle)] transition-colors"
            style={{
              animation: `fadeInUp 0.5s ease-out ${idx * 0.05}s both`,
            }}
          >
            <div className="font-[family-name:var(--font-mono)] text-[11px] text-[var(--text-faint)] w-6 tabular-nums">
              {String(idx + 1).padStart(2, "0")}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-[var(--text-primary)]">
                {insight.human_name}
              </div>
              <div className="text-xs text-[var(--text-muted)] mt-0.5">
                {insight.interpretation}
              </div>
            </div>
            <div className="font-[family-name:var(--font-mono)] text-[10px] text-[var(--text-faint)] hidden md:block">
              {insight.feature}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
