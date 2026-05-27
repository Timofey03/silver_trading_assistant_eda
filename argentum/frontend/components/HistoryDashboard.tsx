"use client";

/**
 * HistoryDashboard — клиентский компонент с shared period state.
 * Метрики и chart синхронизированы: смена периода в chart → пересчёт метрик.
 */
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api, type MetricsResponse } from "@/lib/api";
import { formatPct } from "@/lib/utils";
import CandleChart from "./CandleChart";
import EquityCurve from "./EquityCurve";
import MonthlyHeatmap from "./MonthlyHeatmap";

type Period = "1m" | "3m" | "6m" | "1y" | "3y" | "all";

const PERIODS: { id: Period; label: string }[] = [
  { id: "1m", label: "1М" },
  { id: "3m", label: "3М" },
  { id: "6m", label: "6М" },
  { id: "1y", label: "1Г" },
  { id: "3y", label: "3Г" },
  { id: "all", label: "Всё" },
];

interface Props {
  initialMetrics: MetricsResponse;
}

export default function HistoryDashboard({ initialMetrics }: Props) {
  const [period, setPeriod] = useState<Period>("all");
  const [metrics, setMetrics] = useState<MetricsResponse>(initialMetrics);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(
      `${process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"}/api/metrics?period=${period}`,
      { cache: "no-store" },
    )
      .then((r) => r.json() as Promise<MetricsResponse>)
      .then((m) => {
        if (!cancelled) {
          setMetrics(m);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [period]);

  return (
    <div className="space-y-12">
      {/* Period selector */}
      <div className="flex items-center justify-between">
        <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          Период анализа
        </div>
        <div className="flex gap-1 text-xs font-[family-name:var(--font-mono)]">
          {PERIODS.map((p) => (
            <button
              key={p.id}
              onClick={() => setPeriod(p.id)}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                period === p.id
                  ? "bg-[var(--bg-subtle)] text-[var(--text-primary)] border border-[var(--border)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]/50 border border-transparent"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Metrics — react to period */}
      <MainMetricsAnimated metrics={metrics} period={period} loading={loading} />

      {/* Chart — sync period */}
      <CandleChart period={period} onPeriodChange={setPeriod} />

      {/* Equity curve */}
      <EquityCurve period={period} />

      {/* Monthly heatmap (всегда полный период) */}
      <MonthlyHeatmap />
    </div>
  );
}

function MainMetricsAnimated({
  metrics, period, loading,
}: {
  metrics: MetricsResponse;
  period: Period;
  loading: boolean;
}) {
  const periodLabelMap: Record<Period, string> = {
    "1m": "за месяц", "3m": "за 3 месяца", "6m": "за полгода",
    "1y": "за год",   "3y": "за 3 года",   "all": `за ${metrics.period_years.toFixed(1)} лет`,
  };
  const subLabel = periodLabelMap[period];

  const cards = [
    {
      label: "Накопленная доходность",
      value: metrics.n_trades > 0
        ? formatPct(metrics.total_return * 100, 1)
        : "—",
      sub: metrics.n_trades > 0
        ? `${subLabel} · ${metrics.period_start} → ${metrics.period_end}`
        : "нет сделок в периоде",
      color: metrics.total_return > 0 ? "#10b981" : "#f43f5e",
    },
    {
      label: "Угадывает направление",
      value: metrics.n_trades > 0
        ? `${Math.round(metrics.win_rate * 100)}%`
        : "—",
      sub: `${metrics.n_trades} сдел${
        metrics.n_trades === 1 ? "ка" : metrics.n_trades < 5 ? "ки" : "ок"
      }`,
      color: "#fafafa",
    },
    {
      label: "Максимальная просадка",
      value: metrics.n_trades > 0
        ? formatPct(metrics.max_drawdown * 100, 1)
        : "—",
      sub: `Sharpe ${metrics.sharpe.toFixed(2)}`,
      color: "#f43f5e",
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-3 relative">
      {loading && (
        <div className="absolute -top-6 right-0 text-[10px] text-[var(--text-faint)] font-[family-name:var(--font-mono)] uppercase tracking-widest">
          обновление…
        </div>
      )}
      {cards.map((c, idx) => (
        <motion.div
          key={c.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: idx * 0.05 }}
          className="lift-on-hover rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] px-6 py-7"
        >
          <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            {c.label}
          </div>
          <AnimatePresence mode="wait">
            <motion.div
              key={`${period}-${c.value}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.3 }}
              className="mt-4 font-[family-name:var(--font-mono)] text-4xl font-medium tracking-tight tabular-nums"
              style={{ color: c.color }}
            >
              {c.value}
            </motion.div>
          </AnimatePresence>
          <div className="mt-2 text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
            {c.sub}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
