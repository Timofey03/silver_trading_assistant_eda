"use client";

/**
 * PositionSummaryCard — на главной показывает summary открытых позиций (если есть).
 * Использует market P&L (live), не sandbox.
 */
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { TrendingUp, TrendingDown, ArrowRight } from "lucide-react";
import { api, type PositionsResponse } from "@/lib/api";
import { formatPct, formatRub } from "@/lib/utils";
import CountUp from "./CountUp";

export default function PositionSummaryCard() {
  const [data, setData] = useState<PositionsResponse | null>(null);

  useEffect(() => {
    api.positions().then(setData).catch(() => {});
  }, []);

  if (!data || data.positions.length === 0) return null;

  // Aggregate market P&L (live)
  const totalLots = data.positions.reduce((s, p) => s + p.lots, 0);
  const totalGrams = data.positions.reduce((s, p) => s + p.lots * p.lot_size_g, 0);
  // Volume-weighted avg P&L (по lots)
  const weightedPnl =
    data.positions.reduce(
      (s, p) => s + (p.market_pnl_pct ?? p.unrealized_pnl_pct) * p.lots,
      0,
    ) / Math.max(1, totalLots);
  const totalInvested = data.positions.reduce(
    (s, p) => s + p.entry_price * p.lots,
    0,
  );
  const totalCurrent = data.positions.reduce(
    (s, p) => s + (p.market_current_price ?? p.current_price) * p.lots,
    0,
  );
  const profitRub = totalCurrent - totalInvested;
  const isUp = weightedPnl > 0;
  const color = isUp ? "#10b981" : "#f43f5e";
  const Icon = isUp ? TrendingUp : TrendingDown;

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] px-7 py-5 hover:border-[var(--text-faint)]/30 transition-colors"
    >
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-lg"
            style={{ backgroundColor: `${color}15`, color }}
          >
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
              Твои позиции · live market
            </div>
            <div className="font-[family-name:var(--font-mono)] text-2xl font-medium tabular-nums">
              <CountUp
                value={weightedPnl * 100}
                decimals={2}
                duration={500}
                prefix={isUp ? "+" : ""}
                suffix="%"
                style={{ color }}
              />
              <span className="text-sm text-[var(--text-muted)] ml-3">
                <CountUp
                  value={profitRub}
                  decimals={0}
                  duration={500}
                  prefix={isUp ? "+" : ""}
                  suffix=" ₽"
                />
              </span>
            </div>
            <div className="text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)] mt-1">
              {data.positions.length} {data.positions.length === 1 ? "позиция" : "позиций"}{" "}
              · {totalLots} лот · {totalGrams.toLocaleString("ru-RU")} г серебра
            </div>
          </div>
        </div>

        <Link
          href="/positions"
          className="inline-flex items-center gap-1.5 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          <span>подробнее</span>
          <ArrowRight className="h-3 w-3" />
        </Link>
      </div>
    </motion.section>
  );
}
