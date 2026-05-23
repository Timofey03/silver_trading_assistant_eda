/**
 * Главная страница "Сейчас" — Hero (BUY/HOLD/SELL) + цена + explanation.
 *
 * Server Component — fetches signal/price/explain on the server для скорости.
 * Если backend недоступен — показывает fallback с placeholder данными.
 */
import { api, type SignalResponse, type PriceResponse, type ExplainResponse } from "@/lib/api";
import { formatPct, formatUsd } from "@/lib/utils";
import { ArrowRight, Sparkles, TrendingUp, TrendingDown, Minus } from "lucide-react";

export const revalidate = 60; // refresh every 60s

async function safeApi<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await fn();
  } catch {
    return fallback;
  }
}

export default async function HomePage() {
  const [signal, price, explain] = await Promise.all([
    safeApi(api.signal, {
      signal: "HOLD",
      date: "—",
      close: 0,
      p_up: 0,
      entry_threshold: 0.48,
      exit_threshold: 0.35,
      trail_pct: 0.12,
      max_hold_days: 30,
      cooldown_days: 25,
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
  ]);

  return (
    <div className="space-y-16">
      <HeroSignal signal={signal} />
      <PriceCard price={price} />
      <ExplainSection explain={explain} signal={signal} />
    </div>
  );
}

// ============================================================================
// Hero — большой блок BUY/HOLD/SELL
// ============================================================================
function HeroSignal({ signal }: { signal: SignalResponse }) {
  const variants = {
    BUY: {
      color: "text-emerald-400",
      bg: "from-emerald-500/10 via-emerald-500/5 to-transparent",
      border: "border-emerald-500/30",
      label: "ПОКУПАТЬ",
      sublabel: "Помощник видит сильный сигнал на 2-4 недели вверх",
      Icon: TrendingUp,
    },
    SELL: {
      color: "text-rose-400",
      bg: "from-rose-500/10 via-rose-500/5 to-transparent",
      border: "border-rose-500/30",
      label: "ПРОДАВАТЬ",
      sublabel: "Помощник рекомендует закрыть позицию",
      Icon: TrendingDown,
    },
    HOLD: {
      color: "text-neutral-400",
      bg: "from-neutral-500/10 via-neutral-500/5 to-transparent",
      border: "border-neutral-500/30",
      label: "ОЖИДАТЬ",
      sublabel: "Нет достаточно сильного сигнала — ждём лучшего момента",
      Icon: Minus,
    },
  } as const;

  const v = variants[signal.signal];
  const confidence = Math.round(signal.p_up * 100);

  return (
    <section
      className={`relative overflow-hidden rounded-3xl border ${v.border} bg-gradient-to-br ${v.bg} px-8 py-16 md:px-16 md:py-24`}
    >
      <div className="flex flex-col items-center text-center space-y-8">
        {/* Иконка */}
        <div className={`inline-flex h-20 w-20 items-center justify-center rounded-2xl ${v.color} bg-neutral-950/50`}>
          <v.Icon className="h-10 w-10" strokeWidth={1.5} />
        </div>

        {/* Главное слово */}
        <h1
          className={`font-[family-name:var(--font-mono)] ${v.color} text-7xl md:text-9xl font-medium tracking-tighter`}
        >
          {v.label}
        </h1>

        {/* Подзаголовок */}
        <p className="max-w-md text-lg md:text-xl text-neutral-300 leading-relaxed">
          {v.sublabel}
        </p>

        {/* Уверенность модели */}
        <div className="flex flex-col items-center gap-3 pt-2">
          <div className="flex items-center gap-3 text-sm text-neutral-500">
            <Sparkles className="h-4 w-4" />
            <span>Уверенность модели</span>
          </div>
          <div className="flex items-baseline gap-2 font-[family-name:var(--font-mono)]">
            <span className={`text-4xl font-medium ${v.color}`}>{confidence}%</span>
            <span className="text-sm text-neutral-500">p_up</span>
          </div>
          {/* Progress bar */}
          <div className="w-64 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
            <div
              className={`h-full rounded-full ${
                signal.signal === "BUY" ? "bg-emerald-400"
                : signal.signal === "SELL" ? "bg-rose-400"
                : "bg-neutral-400"
              }`}
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>

        {/* Repeat indicator */}
        {signal.is_repeat && (
          <p className="text-xs text-neutral-500 mt-4 px-4 py-2 rounded-lg bg-neutral-900/50 border border-neutral-800">
            ℹ Сигнал не изменился с прошлого обновления — если уже отреагировал, ничего делать не нужно
          </p>
        )}

        {/* Дата */}
        <p className="text-xs text-neutral-600 font-[family-name:var(--font-mono)]">
          обновлено · {signal.date}
        </p>
      </div>
    </section>
  );
}

// ============================================================================
// Price Card — цена силера + sparkline
// ============================================================================
function PriceCard({ price }: { price: PriceResponse }) {
  const isUp = price.change_pct > 0;
  const sparkPath = sparkPathFromPoints(price.sparkline);

  return (
    <section className="grid gap-6 md:grid-cols-3 items-center rounded-2xl border border-neutral-800 bg-neutral-900/30 px-8 py-8">
      <div>
        <div className="text-xs uppercase tracking-widest text-neutral-500 mb-2">
          {price.ticker} · Silver Futures
        </div>
        <div className="font-[family-name:var(--font-mono)] text-4xl font-medium">
          {formatUsd(price.current)}
        </div>
        <div
          className={`mt-1 inline-flex items-center gap-1 text-sm font-medium font-[family-name:var(--font-mono)] ${
            isUp ? "text-emerald-400" : "text-rose-400"
          }`}
        >
          {isUp ? "▲" : "▼"} {formatPct(price.change_pct)}
        </div>
      </div>

      <div className="md:col-span-2 h-20 relative">
        {price.sparkline.length > 1 && (
          <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="h-full w-full">
            <defs>
              <linearGradient id="sparkGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor={isUp ? "#34D399" : "#FB7185"} stopOpacity="0.4" />
                <stop offset="100%" stopColor={isUp ? "#34D399" : "#FB7185"} stopOpacity="0" />
              </linearGradient>
            </defs>
            <path
              d={sparkPath.area}
              fill="url(#sparkGrad)"
            />
            <path
              d={sparkPath.line}
              fill="none"
              stroke={isUp ? "#34D399" : "#FB7185"}
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

// helper: build SVG path from sparkline
function sparkPathFromPoints(points: { close: number }[]) {
  if (points.length < 2) return { line: "", area: "" };
  const values = points.map((p) => p.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const xStep = 100 / (points.length - 1);
  const coords = points.map((p, i) => {
    const x = i * xStep;
    const y = 30 - ((p.close - min) / range) * 28 - 1;
    return [x, y];
  });
  const linePath = "M " + coords.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" L ");
  const areaPath = linePath + ` L 100,30 L 0,30 Z`;
  return { line: linePath, area: areaPath };
}

// ============================================================================
// Explain Section — feature importance "почему BUY?"
// ============================================================================
function ExplainSection({ explain, signal }: { explain: ExplainResponse; signal: SignalResponse }) {
  if (!explain.insights.length) return null;

  return (
    <section className="space-y-6">
      <div>
        <div className="flex items-center gap-2 text-sm text-neutral-500 mb-2">
          <Sparkles className="h-4 w-4" />
          <span>Почему {signal.signal === "BUY" ? "BUY" : signal.signal === "SELL" ? "SELL" : "HOLD"}?</span>
        </div>
        <h2 className="text-2xl font-medium tracking-tight">
          Что видит модель в данных
        </h2>
        <p className="text-sm text-neutral-500 mt-2">
          Из {signal.n_features_used} признаков модель выбрала эти как самые информативные
        </p>
      </div>

      <ul className="divide-y divide-neutral-800 rounded-xl border border-neutral-800 bg-neutral-900/30">
        {explain.insights.map((insight, idx) => (
          <li key={insight.feature} className="px-6 py-4 flex items-start gap-4">
            <div className="font-[family-name:var(--font-mono)] text-xs text-neutral-600 w-6 pt-0.5">
              {String(idx + 1).padStart(2, "0")}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-neutral-200">
                {insight.human_name}
              </div>
              <div className="text-xs text-neutral-500 mt-0.5">
                {insight.interpretation}
              </div>
              <div className="text-xs font-[family-name:var(--font-mono)] text-neutral-700 mt-1">
                {insight.feature}
              </div>
            </div>
            <ArrowRight className="h-4 w-4 text-neutral-700 flex-shrink-0 mt-1" />
          </li>
        ))}
      </ul>
    </section>
  );
}
