"use client";

/**
 * PositionsView — multi-position management dashboard.
 * - Master assistant card (BUY/WAIT/AVOID) сверху
 * - Список открытых позиций с HOLD/SELL advice
 * - Кнопки Open new / Close position
 */
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  TrendingUp, TrendingDown, AlertCircle, Check, X, Loader2,
  Clock, Target, Shield, Plus, Minus,
} from "lucide-react";
import { api, type PositionsResponse, type PositionRecord } from "@/lib/api";
import { formatRub, formatPct } from "@/lib/utils";

export default function PositionsView({ initial }: { initial: PositionsResponse }) {
  const [data, setData] = useState<PositionsResponse>(initial);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null); // id of position being closed, or "open"
  const [toast, setToast] = useState<{ type: "ok" | "err"; msg: string } | null>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const r = await api.positions();
      setData(r);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const interval = setInterval(refresh, 30_000);  // auto-refresh каждые 30 сек
    return () => clearInterval(interval);
  }, []);

  const showToast = (type: "ok" | "err", msg: string) => {
    setToast({ type, msg });
    setTimeout(() => setToast(null), 5000);
  };

  const handleOpen = async () => {
    setBusy("open");
    try {
      const res = await api.openPosition(1, "SLVRUBF");
      if (res.success) {
        showToast("ok", `Открыта позиция @ ₽${res.executed_price.toFixed(2)}`);
        await refresh();
      } else {
        showToast("err", res.error);
      }
    } catch (e: any) {
      showToast("err", String(e));
    } finally {
      setBusy(null);
    }
  };

  const handleClose = async (id: string) => {
    if (!confirm("Закрыть эту позицию (SELL через Tinkoff sandbox)?")) return;
    setBusy(id);
    try {
      const res = await api.closePosition(id);
      if (res.success) {
        showToast("ok", `Закрыта · P&L ${res.realized_pnl_pct >= 0 ? "+" : ""}${(res.realized_pnl_pct * 100).toFixed(2)}%`);
        await refresh();
      } else {
        showToast("err", res.error);
      }
    } catch (e: any) {
      showToast("err", String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-8">
      <MasterCard data={data} busy={busy === "open"} loading={loading} onOpen={handleOpen} />

      {data.positions.length > 0 ? (
        <section className="space-y-3">
          <div className="flex items-baseline justify-between">
            <h2 className="text-xl font-medium tracking-tight">
              Открытые позиции
            </h2>
            <span className="text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
              {data.n_open} активных
            </span>
          </div>
          <SandboxDisclaimer />
          <AnimatePresence>
            {data.positions.map((p, idx) => (
              <PositionCard
                key={p.id}
                position={p}
                idx={idx}
                busy={busy === p.id}
                onClose={() => handleClose(p.id)}
              />
            ))}
          </AnimatePresence>
        </section>
      ) : (
        <EmptyState />
      )}

      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className={`fixed bottom-6 right-6 z-50 px-5 py-3 rounded-xl border ${
              toast.type === "ok"
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                : "bg-rose-500/10 border-rose-500/30 text-rose-300"
            } text-sm`}
          >
            <div className="flex items-center gap-2">
              {toast.type === "ok" ? <Check className="h-4 w-4" /> : <X className="h-4 w-4" />}
              <span>{toast.msg}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MasterCard({
  data, busy, loading, onOpen,
}: {
  data: PositionsResponse;
  busy: boolean;
  loading: boolean;
  onOpen: () => void;
}) {
  const variant = {
    BUY:   { color: "#10b981", bg: "from-emerald-500/[0.08]", border: "border-emerald-500/20", icon: TrendingUp, label: "ОТКРЫТЬ", sub: "сигнал достаточно сильный для входа" },
    WAIT:  { color: "#f59e0b", bg: "from-amber-500/[0.06]",   border: "border-amber-500/20",   icon: Clock,      label: "ОЖИДАТЬ",  sub: "недостаточно условий для входа" },
    AVOID: { color: "#f43f5e", bg: "from-rose-500/[0.06]",    border: "border-rose-500/20",    icon: AlertCircle, label: "НЕ ВХОДИТЬ", sub: "рынок против сделки" },
  }[data.master_signal];

  const Icon = variant.icon;

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative overflow-hidden rounded-3xl border ${variant.border} bg-gradient-to-b ${variant.bg} via-transparent to-transparent bg-[var(--bg-elevated)] px-8 py-7`}
    >
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="space-y-3 flex-1 min-w-[200px]">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            <Icon className="h-3 w-3" style={{ color: variant.color }} />
            <span>Главный помощник</span>
            <span className="text-[var(--text-faint)]/50">·</span>
            <span>стоит ли открывать новую позицию?</span>
          </div>
          <div className="flex items-baseline gap-4 flex-wrap">
            <h2
              className="font-[family-name:var(--font-mono)] text-4xl md:text-5xl font-medium tracking-tighter"
              style={{ color: variant.color }}
            >
              {variant.label}
            </h2>
            <div className="font-[family-name:var(--font-mono)] text-xl text-[var(--text-secondary)] tabular-nums">
              p_up = <span style={{ color: variant.color }}>{(data.master_p_up * 100).toFixed(0)}%</span>
            </div>
          </div>
          <p className="text-sm text-[var(--text-muted)] max-w-xl">{variant.sub}</p>
          <p className="text-xs text-[var(--text-faint)] font-[family-name:var(--font-mono)] max-w-xl">
            {data.master_reason}
          </p>
        </div>

        {data.can_buy && (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onOpen}
            disabled={busy}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 text-sm font-medium transition-colors disabled:opacity-50"
          >
            {busy ? (
              <><Loader2 className="h-4 w-4 animate-spin" />Открываем…</>
            ) : (
              <><Plus className="h-4 w-4" />Открыть 1 лот</>
            )}
          </motion.button>
        )}
      </div>
    </motion.section>
  );
}

function PositionCard({
  position, idx, busy, onClose,
}: {
  position: PositionRecord; idx: number; busy: boolean; onClose: () => void;
}) {
  const isSell = position.advice === "SELL";
  const isUp = position.unrealized_pnl_pct > 0;
  const pnlColor = isUp ? "#10b981" : "#f43f5e";
  const cardBorder = isSell ? "border-rose-500/30" : "border-[var(--border)]";
  const cardBg = isSell
    ? "bg-gradient-to-br from-rose-500/[0.04] via-transparent to-transparent"
    : "";

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3, delay: idx * 0.05 }}
      className={`rounded-2xl border ${cardBorder} ${cardBg} bg-[var(--bg-elevated)] px-6 py-5`}
    >
      <div className="flex items-start gap-4">
        {/* Advice badge */}
        <div
          className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
          style={{
            backgroundColor: isSell ? "rgba(244,63,94,0.15)" : "rgba(16,185,129,0.10)",
            color: isSell ? "#f43f5e" : "#10b981",
          }}
        >
          {isSell ? <TrendingDown className="h-4 w-4" /> : <Shield className="h-4 w-4" />}
        </div>

        {/* Main info */}
        <div className="flex-1 min-w-0 space-y-2">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest"
               style={{ color: isSell ? "#f43f5e" : "#10b981" }}>
            <span>{isSell ? "РЕКОМЕНДУЕТ ПРОДАТЬ" : "ДЕРЖАТЬ"}</span>
            <span className="text-[var(--text-faint)]/50">·</span>
            <span className="text-[var(--text-faint)]">{position.ticker} · {position.lots} лот</span>
          </div>

          <div className="flex items-baseline gap-4 flex-wrap font-[family-name:var(--font-mono)] tabular-nums">
            <div>
              <span className="text-[var(--text-faint)] text-xs">вход </span>
              <span className="text-[var(--text-secondary)]">{formatRub(position.entry_price)}</span>
            </div>
            <div>
              <span className="text-[var(--text-faint)] text-xs">сейчас </span>
              <span className="text-[var(--text-primary)] font-medium">{formatRub(position.current_price)}</span>
            </div>
            <div>
              <span className="text-[var(--text-faint)] text-xs">P&L </span>
              <span className="text-lg font-medium" style={{ color: pnlColor }}>
                {isUp ? "+" : ""}{formatPct(position.unrealized_pnl_pct * 100, 2)}
              </span>
            </div>
          </div>

          <div className="text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
            {position.advice_reason}
          </div>

          <div className="flex items-center gap-4 text-[11px] text-[var(--text-faint)] font-[family-name:var(--font-mono)]">
            <span>открыто {position.opened_at.slice(0, 10)}</span>
            <span>·</span>
            <span>peak {formatRub(position.peak_price)}</span>
            <span>·</span>
            <span>{position.lots * position.lot_size_g} г серебра</span>
          </div>
        </div>

        {/* Close button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.95 }}
          onClick={onClose}
          disabled={busy}
          className={`px-4 py-2 rounded-lg border text-xs font-medium transition-colors disabled:opacity-50 ${
            isSell
              ? "border-rose-500/40 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300"
              : "border-[var(--border)] hover:bg-[var(--bg-subtle)] text-[var(--text-secondary)]"
          }`}
        >
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> :
            <span className="flex items-center gap-1.5"><Minus className="h-3 w-3" />Закрыть</span>}
        </motion.button>
      </div>
    </motion.div>
  );
}

function SandboxDisclaimer() {
  return (
    <div className="rounded-xl border border-amber-500/15 bg-amber-500/[0.03] px-4 py-2.5 flex items-start gap-2">
      <AlertCircle className="h-3.5 w-3.5 text-amber-400/70 flex-shrink-0 mt-0.5" />
      <p className="text-xs text-[var(--text-muted)] leading-relaxed">
        <span className="text-amber-400/80 font-medium">Sandbox-режим:</span> Tinkoff
        sandbox симулирует исполнение со <b>spread'ом до 4%</b> от рыночной цены.
        P&L может казаться хуже чем будет на live (где spread обычно 0.05-0.1%).
      </p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] px-8 py-12 text-center space-y-3">
      <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
        Нет открытых позиций
      </div>
      <p className="text-sm text-[var(--text-muted)] max-w-md mx-auto">
        Когда главный помощник скажет ОТКРЫТЬ, можно создать новую позицию.
        Каждая позиция будет отслеживаться независимо со своими HOLD/SELL рекомендациями.
      </p>
    </div>
  );
}
