"use client";

/**
 * FeatureContribution — что именно «увидела» модель для текущего сигнала.
 * Использует /api/explain (top-10 фичей с интерпретацией).
 */
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, ChevronRight } from "lucide-react";
import { api, type ExplainResponse } from "@/lib/api";

export default function FeatureContribution() {
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    api.explain().then(setData).catch(() => {});
  }, []);

  if (!data || data.insights.length === 0) return null;

  const visible = expanded ? data.insights : data.insights.slice(0, 5);

  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            <Sparkles className="h-3 w-3 text-emerald-400/60" />
            <span>Объяснение сигнала</span>
          </div>
          <h2 className="text-xl font-medium tracking-tight mt-1">
            На что смотрит модель
          </h2>
        </div>
        <span className="text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
          top {data.insights.length} фичей
        </span>
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] overflow-hidden">
        <ul className="divide-y divide-[var(--border-soft)]">
          {visible.map((insight, idx) => {
            // Importance bar — линейный градиент по позиции (1й = 100%, 10й = 50%)
            const importance = 100 - idx * 5;
            return (
              <motion.li
                key={insight.feature}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                className="px-6 py-4 hover:bg-[var(--bg-subtle)]/50 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="font-[family-name:var(--font-mono)] text-xs text-[var(--text-faint)] w-6 tabular-nums">
                    {String(idx + 1).padStart(2, "0")}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline justify-between gap-3 mb-1.5">
                      <div className="text-sm font-medium text-[var(--text-primary)]">
                        {insight.human_name}
                      </div>
                      <div className="text-[10px] text-[var(--text-faint)] font-[family-name:var(--font-mono)] hidden md:block">
                        {insight.feature}
                      </div>
                    </div>
                    {/* importance bar */}
                    <div className="h-1 rounded-full bg-[var(--bg-subtle)] overflow-hidden mb-1.5">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${importance}%` }}
                        transition={{ duration: 0.6, delay: idx * 0.05 }}
                        className="h-full rounded-full bg-emerald-500/60"
                      />
                    </div>
                    <div className="text-xs text-[var(--text-muted)] leading-relaxed">
                      {insight.interpretation}
                    </div>
                  </div>
                </div>
              </motion.li>
            );
          })}
        </ul>
        {data.insights.length > 5 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-6 py-2.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]/50 transition-colors flex items-center justify-center gap-1.5"
          >
            <span>{expanded ? "скрыть" : `показать остальные ${data.insights.length - 5}`}</span>
            <ChevronRight
              className={`h-3 w-3 transition-transform ${expanded ? "rotate-90" : ""}`}
            />
          </button>
        )}
      </div>
    </section>
  );
}
