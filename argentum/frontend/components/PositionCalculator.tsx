"use client";

/**
 * PositionCalculator — мини-инструмент: при покупке N лотов, какой риск/profit?
 */
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Calculator } from "lucide-react";
import { api } from "@/lib/api";
import { formatRub } from "@/lib/utils";

export default function PositionCalculator() {
  const [open, setOpen] = useState(false);
  const [lots, setLots] = useState(1);
  const [pricePerContract, setPricePerContract] = useState(17400);
  const [usdrub, setUsdrub] = useState(71);

  useEffect(() => {
    if (!open) return;
    api.fx().then((fx) => {
      if (fx.usdrub > 0) setUsdrub(fx.usdrub);
      // 1 контракт = 100г, 1oz = 31.1г
      if (fx.usd_silver > 0) {
        setPricePerContract(fx.usd_silver * (100 / 31.1035) * fx.usdrub);
      }
    });
  }, [open]);

  const SCENARIOS = [
    { label: "Серебро +10%", mult: 1.10, color: "#10b981" },
    { label: "Серебро +5%",  mult: 1.05, color: "#10b981" },
    { label: "Без изменений", mult: 1.00, color: "#a1a1aa" },
    { label: "Серебро -5%",  mult: 0.95, color: "#f43f5e" },
    { label: "Серебро -10%", mult: 0.90, color: "#f43f5e" },
  ];

  const totalCost = pricePerContract * lots;
  const margin = totalCost * 0.20; // приблизительно 20% ГО

  return (
    <section className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-7 py-4 flex items-center justify-between hover:bg-[var(--bg-subtle)]/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400">
            <Calculator className="h-4 w-4" />
          </div>
          <div className="text-left">
            <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
              Калькулятор позиции
            </div>
            <div className="text-sm text-[var(--text-secondary)]">
              «куплю X лотов, какой риск?»
            </div>
          </div>
        </div>
        <span className="text-[var(--text-faint)] text-xs">{open ? "▼" : "▶"}</span>
      </button>

      {open && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          className="px-7 pb-6 space-y-4"
        >
          <div className="flex items-center gap-4 flex-wrap">
            <label className="text-xs text-[var(--text-muted)]">Лотов:</label>
            <input
              type="number"
              min={1}
              max={100}
              value={lots}
              onChange={(e) => setLots(Math.max(1, Math.min(100, +e.target.value)))}
              className="w-20 px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--bg-base)] text-[var(--text-primary)] font-[family-name:var(--font-mono)] text-sm"
            />
            <div className="text-xs text-[var(--text-faint)] font-[family-name:var(--font-mono)]">
              = {lots * 100} г серебра
            </div>
            <div className="ml-auto text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
              Цена контракта: {formatRub(pricePerContract)}
            </div>
          </div>

          <div className="rounded-lg border border-[var(--border-soft)] bg-[var(--bg-base)] p-3 grid grid-cols-2 gap-y-1.5 text-xs font-[family-name:var(--font-mono)]">
            <span className="text-[var(--text-faint)]">Номинал позиции:</span>
            <span className="text-right text-[var(--text-primary)] font-medium">
              {formatRub(totalCost)}
            </span>
            <span className="text-[var(--text-faint)]">Гарантийное обеспечение (~20%):</span>
            <span className="text-right text-[var(--text-secondary)]">
              {formatRub(margin)}
            </span>
          </div>

          <div className="space-y-1.5">
            <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)] mb-2">
              Сценарии (P&L через 20 дней)
            </div>
            {SCENARIOS.map((s) => {
              const newPrice = pricePerContract * s.mult;
              const profit = (newPrice - pricePerContract) * lots;
              const pnlPct = (s.mult - 1) * 100;
              return (
                <div
                  key={s.label}
                  className="flex items-center justify-between text-xs font-[family-name:var(--font-mono)] py-1.5 px-3 rounded hover:bg-[var(--bg-subtle)]/30 transition-colors"
                >
                  <span className="text-[var(--text-secondary)]">{s.label}</span>
                  <div className="flex items-center gap-4 tabular-nums">
                    <span style={{ color: s.color }}>
                      {pnlPct > 0 ? "+" : ""}{pnlPct.toFixed(1)}%
                    </span>
                    <span className="w-32 text-right" style={{ color: s.color }}>
                      {profit > 0 ? "+" : ""}{formatRub(profit)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="text-[10px] text-[var(--text-faint)] italic">
            Внимание: цифры расчётные, реальное движение зависит от рынка
          </div>
        </motion.div>
      )}
    </section>
  );
}
