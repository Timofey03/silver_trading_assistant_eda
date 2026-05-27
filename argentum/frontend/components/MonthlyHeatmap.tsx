"use client";

/**
 * MonthlyHeatmap — SVG heatmap returns × years (Quantopian-style).
 */
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, type MonthlyResponse } from "@/lib/api";

const MONTH_LABELS = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"];

function colorFor(ret: number, maxAbs: number): string {
  if (ret === 0) return "transparent";
  const intensity = Math.min(1, Math.abs(ret) / maxAbs);
  if (ret > 0) {
    // Emerald gradient
    const alpha = 0.15 + intensity * 0.75;
    return `rgba(16, 185, 129, ${alpha.toFixed(2)})`;
  } else {
    const alpha = 0.15 + intensity * 0.75;
    return `rgba(244, 63, 94, ${alpha.toFixed(2)})`;
  }
}

export default function MonthlyHeatmap() {
  const [data, setData] = useState<MonthlyResponse | null>(null);
  const [hover, setHover] = useState<{ year: number; month: number; ret: number; n: number } | null>(null);

  useEffect(() => {
    api.monthly().then(setData).catch(() => {});
  }, []);

  if (!data || data.cells.length === 0) {
    return null;
  }

  // Build lookup table: cellsByKey[year][month] = cell
  const cellMap: Record<number, Record<number, { ret: number; n: number }>> = {};
  for (const c of data.cells) {
    if (!cellMap[c.year]) cellMap[c.year] = {};
    cellMap[c.year][c.month] = { ret: c.return_pct, n: c.n_trades };
  }

  const years = data.years;
  const maxAbs = Math.max(
    Math.abs(data.best_month), Math.abs(data.worst_month),
  );

  // Compute yearly totals
  const yearlyTotals = years.map((y) => {
    const months = cellMap[y] || {};
    const ret = Object.values(months).reduce((acc, c) => acc * (1 + c.ret), 1) - 1;
    return { year: y, ret };
  });

  return (
    <section className="space-y-3">
      <div>
        <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          Monthly returns heatmap · compound by exit-month
        </div>
        <h2 className="text-xl font-medium tracking-tight mt-1">
          Доходность по месяцам за 11 лет
        </h2>
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-4 overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Header: months */}
          <div className="flex">
            <div className="w-14 shrink-0" />
            {MONTH_LABELS.map((m) => (
              <div
                key={m}
                className="w-12 text-center text-[10px] uppercase tracking-widest text-[var(--text-faint)] font-[family-name:var(--font-mono)]"
              >
                {m}
              </div>
            ))}
            <div className="w-16 text-center text-[10px] uppercase tracking-widest text-[var(--text-faint)] font-[family-name:var(--font-mono)]">
              Год
            </div>
          </div>

          {/* Rows: years */}
          {years.map((year, ri) => (
            <motion.div
              key={year}
              className="flex items-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: ri * 0.02 }}
            >
              <div className="w-14 shrink-0 text-right pr-3 text-[11px] text-[var(--text-secondary)] font-[family-name:var(--font-mono)] tabular-nums">
                {year}
              </div>
              {Array.from({ length: 12 }, (_, m) => m + 1).map((m) => {
                const cell = cellMap[year]?.[m];
                const ret = cell?.ret ?? 0;
                const n = cell?.n ?? 0;
                return (
                  <div
                    key={`${year}-${m}`}
                    className="w-12 h-9 m-px rounded-sm flex items-center justify-center font-[family-name:var(--font-mono)] text-[10px] tabular-nums cursor-pointer transition-transform hover:scale-110"
                    style={{
                      backgroundColor: colorFor(ret, maxAbs),
                      color: Math.abs(ret) > maxAbs * 0.5 ? "#fafafa" : "var(--text-secondary)",
                    }}
                    onMouseEnter={() => setHover({ year, month: m, ret, n })}
                    onMouseLeave={() => setHover(null)}
                  >
                    {cell ? `${ret > 0 ? "+" : ""}${(ret * 100).toFixed(1)}` : "·"}
                  </div>
                );
              })}
              {/* Year total */}
              <div
                className="w-16 h-9 mx-1 my-px rounded-sm flex items-center justify-center font-[family-name:var(--font-mono)] text-[11px] font-medium tabular-nums border"
                style={{
                  borderColor: yearlyTotals[ri].ret > 0 ? "rgba(16,185,129,0.4)" : "rgba(244,63,94,0.4)",
                  color: yearlyTotals[ri].ret > 0 ? "#10b981" : "#f43f5e",
                }}
              >
                {yearlyTotals[ri].ret > 0 ? "+" : ""}{(yearlyTotals[ri].ret * 100).toFixed(0)}%
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <Stat label="Лучший месяц" value={`+${(data.best_month * 100).toFixed(1)}%`} color="#10b981" />
        <Stat label="Худший месяц" value={`${(data.worst_month * 100).toFixed(1)}%`} color="#f43f5e" />
        <Stat label="Лучший год" value={`+${(data.best_year * 100).toFixed(0)}%`} color="#10b981" />
        <Stat label="Средний месяц" value={`${data.avg_month > 0 ? "+" : ""}${(data.avg_month * 100).toFixed(2)}%`}
              color={data.avg_month > 0 ? "#10b981" : "#f43f5e"} />
      </div>

      {/* Hover tooltip */}
      {hover && (
        <div className="text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
          {MONTH_LABELS[hover.month - 1]} {hover.year}:
          <span style={{ color: hover.ret >= 0 ? "#10b981" : "#f43f5e" }} className="ml-2">
            {hover.ret > 0 ? "+" : ""}{(hover.ret * 100).toFixed(2)}%
          </span>
          <span className="ml-2 text-[var(--text-faint)]">· {hover.n} {hover.n === 1 ? "сделка" : "сделок"}</span>
        </div>
      )}
    </section>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-lg border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-faint)]">{label}</div>
      <div
        className="mt-0.5 font-[family-name:var(--font-mono)] text-sm tabular-nums font-medium"
        style={{ color }}
      >
        {value}
      </div>
    </div>
  );
}
