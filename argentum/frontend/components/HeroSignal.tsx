"use client";

/**
 * HeroSignal — анимированная большая карточка BUY/HOLD/SELL.
 */
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { TrendingUp, TrendingDown, Minus, Info, Check, X, Loader2 } from "lucide-react";
import { api, type SignalResponse, type OrderResponse } from "@/lib/api";

const STRONG_THRESHOLD = 0.85;     // совпадает с backend strong-signal filter

const variants = {
  BUY: {
    label: "ПОКУПАТЬ",
    sub: "Сильный сигнал · уверенность ≥ 85% — помощник видит надёжный setup",
    color: "text-emerald-400",
    colorHex: "#10b981",
    icon: TrendingUp,
    gradient: "from-emerald-500/[0.08] via-emerald-500/[0.02] to-transparent",
    glow: "shadow-[0_0_120px_-20px_rgba(16,185,129,0.25)]",
    border: "border-emerald-500/20",
    cta: "Купить через Tinkoff",
  },
  SELL: {
    label: "ПРОДАВАТЬ",
    sub: "Помощник рекомендует закрыть позицию",
    color: "text-rose-400",
    colorHex: "#f43f5e",
    icon: TrendingDown,
    gradient: "from-rose-500/[0.08] via-rose-500/[0.02] to-transparent",
    glow: "shadow-[0_0_120px_-20px_rgba(244,63,94,0.25)]",
    border: "border-rose-500/20",
    cta: "Продать через Tinkoff",
  },
  HOLD: {
    label: "ОЖИДАТЬ",
    sub: "Уверенность в зоне шума (50-85%) — ждём более сильного сигнала",
    color: "text-amber-400",
    colorHex: "#f59e0b",
    icon: Minus,
    gradient: "from-amber-500/[0.06] to-transparent",
    glow: "",
    border: "border-amber-500/20",
    cta: "",
  },
} as const;

export default function HeroSignal({ signal }: { signal: SignalResponse }) {
  // Apply strong-signal filter — единая логика с /positions master assistant
  // Сырая модель: signal=BUY если p_up >= 0.48 (слабый порог)
  // Фактическое решение: BUY только если p_up >= 0.85 (strong filter)
  const passesStrongFilter = signal.p_up >= STRONG_THRESHOLD;
  const effectiveSignal: "BUY" | "HOLD" | "SELL" =
    signal.signal === "BUY" && !passesStrongFilter
      ? "HOLD"  // raw BUY но в шумовой зоне → реально HOLD
      : signal.signal;

  const v = variants[effectiveSignal];
  const confidence = Math.round(signal.p_up * 100);
  const Icon = v.icon;

  const [orderStatus, setOrderStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [orderRes, setOrderRes] = useState<OrderResponse | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const handleOrder = async () => {
    setOrderStatus("loading");
    try {
      const res = await api.order({
        direction: signal.signal === "SELL" ? "SELL" : "BUY",
        lots: 1,
        ticker: "SLVRUBF",
      });
      setOrderRes(res);
      setOrderStatus(res.success ? "success" : "error");
    } catch (e: any) {
      setOrderRes({
        success: false, order_id: "", executed_lots: 0, executed_price: 0,
        direction: "", figi: "", error: String(e),
      });
      setOrderStatus("error");
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className={`relative overflow-hidden rounded-3xl border ${v.border} bg-gradient-to-b ${v.gradient} ${v.glow} bg-[var(--bg-elevated)]`}
    >
      <div className="relative px-8 py-20 md:px-16 md:py-28 flex flex-col items-center text-center">
        {/* Live badge */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="absolute top-6 left-6 flex items-center gap-1.5 text-[11px] uppercase tracking-widest text-[var(--text-faint)]"
        >
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-70" />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400" />
          </span>
          Live · E3b
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="absolute top-6 right-6 font-[family-name:var(--font-mono)] text-[11px] text-[var(--text-faint)]"
        >
          {signal.date}
        </motion.div>

        {/* Icon */}
        <motion.div
          initial={{ scale: 0, rotate: -30 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ delay: 0.1, type: "spring", stiffness: 200, damping: 15 }}
          className={`mb-8 inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--bg-base)] border border-[var(--border)] ${v.color}`}
        >
          <Icon className="h-7 w-7" strokeWidth={2} />
        </motion.div>

        {/* Главное слово */}
        <motion.h1
          initial={{ opacity: 0, scale: 0.85, y: 30 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className={`font-[family-name:var(--font-mono)] ${v.color} text-6xl md:text-8xl lg:text-[8rem] font-medium tracking-tighter leading-none`}
          style={{ textShadow: `0 0 80px ${v.colorHex}30` }}
        >
          {v.label}
        </motion.h1>

        {/* Подзаголовок */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="mt-8 max-w-md text-base md:text-lg text-[var(--text-secondary)] leading-relaxed"
        >
          {v.sub}
        </motion.p>

        {/* Confidence */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.6 }}
          className="mt-12 flex flex-col items-center gap-3"
        >
          <div className="flex items-baseline gap-2">
            <span className={`font-[family-name:var(--font-mono)] text-3xl font-medium ${v.color}`}>
              {confidence}%
            </span>
            <span className="text-xs text-[var(--text-faint)] uppercase tracking-wider">
              уверенность
            </span>
          </div>
          <div className="w-56 h-1 rounded-full bg-[var(--bg-subtle)] overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidence}%` }}
              transition={{ delay: 0.8, duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
              className="h-full rounded-full"
              style={{ backgroundColor: v.colorHex }}
            />
          </div>
        </motion.div>

        {/* CTA — Купить через Tinkoff */}
        {v.cta && (
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 0.6 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setConfirmOpen(true)}
            className={`mt-10 inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-[var(--border)] bg-[var(--bg-base)] hover:bg-[var(--bg-subtle)] text-sm font-medium text-[var(--text-primary)] transition-colors`}
          >
            <span>{v.cta}</span>
            <span className="text-[var(--text-faint)] text-xs">→ sandbox</span>
          </motion.button>
        )}

        {/* Repeat indicator */}
        {signal.is_repeat && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.0, duration: 0.6 }}
            className="mt-10 flex items-start gap-2 max-w-md px-4 py-3 rounded-lg border border-[var(--border)] bg-[var(--bg-base)]/50"
          >
            <Info className="h-4 w-4 text-[var(--text-muted)] flex-shrink-0 mt-0.5" />
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed text-left">
              Сигнал не изменился с прошлого обновления — если ты уже отреагировал, ничего делать не нужно
            </p>
          </motion.div>
        )}
      </div>

      {/* Confirm modal */}
      <AnimatePresence>
        {confirmOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4"
            onClick={() => orderStatus === "idle" && setConfirmOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-md rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-7"
            >
              {orderStatus === "success" && orderRes ? (
                <div className="text-center space-y-4">
                  <div className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400">
                    <Check className="h-6 w-6" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium">Ордер исполнен</h3>
                    <p className="mt-1 text-sm text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
                      {orderRes.executed_lots} лот по ${orderRes.executed_price.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-[11px] text-[var(--text-faint)] font-[family-name:var(--font-mono)]">
                    order_id: {orderRes.order_id.slice(0, 8)}…
                  </div>
                  <button
                    onClick={() => { setConfirmOpen(false); setOrderStatus("idle"); }}
                    className="mt-2 w-full px-4 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg-base)] hover:bg-[var(--bg-subtle)] text-sm font-medium transition-colors"
                  >
                    Готово
                  </button>
                </div>
              ) : orderStatus === "error" && orderRes ? (
                <div className="text-center space-y-4">
                  <div className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-rose-500/10 text-rose-400">
                    <X className="h-6 w-6" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium">Ошибка ордера</h3>
                    <p className="mt-2 text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)] break-all">
                      {orderRes.error}
                    </p>
                  </div>
                  <button
                    onClick={() => { setConfirmOpen(false); setOrderStatus("idle"); }}
                    className="mt-2 w-full px-4 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg-base)] hover:bg-[var(--bg-subtle)] text-sm font-medium transition-colors"
                  >
                    Закрыть
                  </button>
                </div>
              ) : (
                <>
                  <h3 className="text-lg font-medium tracking-tight">
                    Подтвердить {signal.signal === "SELL" ? "продажу" : "покупку"}
                  </h3>
                  <p className="mt-2 text-sm text-[var(--text-muted)]">
                    Будет создан <span className="font-[family-name:var(--font-mono)] text-[var(--text-primary)]">market</span> ордер на <span className="font-[family-name:var(--font-mono)] text-[var(--text-primary)]">1 лот SLV</span> в sandbox-режиме Tinkoff. Реальные деньги не списываются.
                  </p>

                  <div className="mt-5 space-y-2 text-xs font-[family-name:var(--font-mono)] text-[var(--text-secondary)] rounded-lg border border-[var(--border-soft)] bg-[var(--bg-base)] p-3">
                    <div className="flex justify-between"><span className="text-[var(--text-faint)]">Тикер</span><span>SLVRUBF</span></div>
                    <div className="flex justify-between"><span className="text-[var(--text-faint)]">Направление</span><span className={v.color}>{signal.signal}</span></div>
                    <div className="flex justify-between"><span className="text-[var(--text-faint)]">Уверенность</span><span>{confidence}%</span></div>
                    <div className="flex justify-between"><span className="text-[var(--text-faint)]">Режим</span><span>sandbox</span></div>
                  </div>

                  <div className="mt-6 flex gap-2">
                    <button
                      onClick={() => setConfirmOpen(false)}
                      disabled={orderStatus === "loading"}
                      className="flex-1 px-4 py-2.5 rounded-lg border border-[var(--border)] text-sm text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] transition-colors disabled:opacity-50"
                    >
                      Отмена
                    </button>
                    <button
                      onClick={handleOrder}
                      disabled={orderStatus === "loading"}
                      className="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium text-white transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                      style={{ backgroundColor: v.colorHex }}
                    >
                      {orderStatus === "loading" ? (
                        <><Loader2 className="h-4 w-4 animate-spin" />Отправка…</>
                      ) : (
                        "Подтвердить"
                      )}
                    </button>
                  </div>
                </>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  );
}
