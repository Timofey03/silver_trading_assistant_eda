/**
 * /history — открытая позиция + метрики + таблица сделок.
 */
import {
  api,
  type MetricsResponse,
  type HistoryResponse,
  type OpenPositionResponse,
} from "@/lib/api";
import { formatPct, formatUsd } from "@/lib/utils";
import { TrendingUp, Clock, Target, Shield } from "lucide-react";
import HistoryDashboard from "@/components/HistoryDashboard";

export const revalidate = 60;

async function safeApi<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try { return await fn(); } catch { return fallback; }
}

export default async function HistoryPage() {
  const [metrics, history, position] = await Promise.all([
    safeApi(api.metrics, {
      sharpe: 0, sortino: 0, annual_return: 0, total_return: 0,
      max_drawdown: 0, profit_factor: 0, win_rate: 0, n_trades: 0,
      oos_accuracy: 0, psr: 0, period_years: 0,
      best_trade: 0, worst_trade: 0, model_name: "E3b", model_features: 30,
    } as MetricsResponse),
    safeApi(() => api.history(20), {
      equity_curve: [], trades: [], n_trades: 0,
      total_return: 0, period_start: "—", period_end: "—",
    } as HistoryResponse),
    safeApi(api.position, {
      is_open: false, entry_date: "", entry_price: 0, current_price: 0,
      unrealized_return: 0, days_held: 0, max_hold_days: 30, trail_pct: 0.12,
      stop_price: 0, target_close: 0, signal: "HOLD", p_up: 0, source: "offline",
    } as OpenPositionResponse),
  ]);

  return (
    <div className="space-y-12">
      <div className="space-y-2">
        <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          Walk-forward бэктест
        </div>
        <h1 className="text-3xl font-medium tracking-tight">
          Как работал помощник
        </h1>
        <p className="text-sm text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
          {history.period_start} — {history.period_end} · {metrics.period_years.toFixed(1)} лет
        </p>
      </div>

      {position.is_open ? (
        <OpenPositionCard pos={position} />
      ) : position.regime_reason ? (
        <RegimeWaitingCard pos={position} />
      ) : null}

      <HistoryDashboard initialMetrics={metrics} />
      <TradesList history={history} />
    </div>
  );
}

// ============================================================================
// Открытая позиция — большая яркая карточка
// ============================================================================
function OpenPositionCard({ pos }: { pos: OpenPositionResponse }) {
  const isUp = pos.unrealized_return >= 0;
  const color = isUp ? "#10b981" : "#f43f5e";
  const colorBg = isUp ? "from-emerald-500/[0.06]" : "from-rose-500/[0.06]";
  const colorBorder = isUp ? "border-emerald-500/20" : "border-rose-500/20";

  const timeProgress = Math.min(100, (pos.days_held / pos.max_hold_days) * 100);

  return (
    <section
      className={`relative overflow-hidden rounded-2xl border ${colorBorder} bg-gradient-to-br ${colorBg} via-transparent to-transparent bg-[var(--bg-elevated)]`}
      style={{ animation: "fadeInScale 0.6s cubic-bezier(0.16,1,0.3,1) both" }}
    >
      <div className="px-7 py-6 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
              </span>
              <span>Позиция открыта · BUY</span>
            </div>
            <h2 className="text-xl font-medium tracking-tight text-[var(--text-primary)]">
              Помощник сейчас в сделке
            </h2>
          </div>
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
              Доходность
            </div>
            <div
              className="font-[family-name:var(--font-mono)] text-3xl font-medium tabular-nums mt-1"
              style={{ color }}
            >
              {isUp ? "+" : ""}{formatPct(pos.unrealized_return * 100, 2)}
            </div>
          </div>
        </div>

        {/* Цены — вход / текущая / стоп / цель */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-5 pt-2 border-t border-[var(--border-soft)]">
          <PosStat
            icon={<TrendingUp className="h-3.5 w-3.5" />}
            label="Вход"
            value={formatUsd(pos.entry_price)}
            sub={pos.entry_date}
          />
          <PosStat
            icon={<TrendingUp className="h-3.5 w-3.5" />}
            label="Сейчас"
            value={formatUsd(pos.current_price)}
            sub="live"
            highlight
          />
          <PosStat
            icon={<Shield className="h-3.5 w-3.5" />}
            label="Стоп"
            value={formatUsd(pos.stop_price)}
            sub={`−${(pos.trail_pct * 100).toFixed(0)}% trailing`}
          />
          <PosStat
            icon={<Target className="h-3.5 w-3.5" />}
            label="Цель"
            value={formatUsd(pos.target_close)}
            sub={`+${(((pos.target_close - pos.entry_price) / pos.entry_price) * 100).toFixed(1)}%`}
          />
        </div>

        {/* Прогресс по времени удержания */}
        <div className="space-y-2 pt-2 border-t border-[var(--border-soft)]">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5 text-[var(--text-muted)]">
              <Clock className="h-3.5 w-3.5" />
              <span>Удерживается</span>
            </div>
            <div className="font-[family-name:var(--font-mono)] tabular-nums text-[var(--text-secondary)]">
              {pos.days_held} / {pos.max_hold_days} дн.
            </div>
          </div>
          <div className="h-1.5 w-full rounded-full bg-[var(--bg-subtle)] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${timeProgress}%`,
                backgroundColor: timeProgress > 80 ? "#f59e0b" : "#10b981",
              }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function RegimeWaitingCard({ pos }: { pos: OpenPositionResponse }) {
  return (
    <section
      className="relative overflow-hidden rounded-2xl border border-amber-500/20 bg-gradient-to-br from-amber-500/[0.04] via-transparent to-transparent bg-[var(--bg-elevated)] px-7 py-5"
      style={{ animation: "fadeInScale 0.6s cubic-bezier(0.16,1,0.3,1) both" }}
    >
      <div className="flex items-start gap-4">
        <div className="mt-0.5 flex h-9 w-9 items-center justify-center rounded-lg bg-amber-500/10 text-amber-400">
          <Clock className="h-4 w-4" />
        </div>
        <div className="flex-1 space-y-1.5">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            <span>Regime filter</span>
            <span className="text-amber-400/70">·</span>
            <span>модель ждёт</span>
          </div>
          <h2 className="text-base font-medium tracking-tight text-[var(--text-primary)]">
            Модель сейчас в зоне шума — ждёт сильный сигнал
          </h2>
          <p className="text-sm text-[var(--text-muted)] leading-relaxed">
            E3b предсказания имеют 2 режима: <span className="text-[var(--text-secondary)]">шум 0.4-0.7</span> (ложные срабатывания)
            и <span className="text-[var(--text-secondary)]">strong 0.85+</span> (надёжные тренды).
            По grid-search (<span className="font-[family-name:var(--font-mono)] text-[var(--text-secondary)]">Sharpe 1.20</span> vs 0.52 без фильтра),
            модель торгует только при <span className="font-[family-name:var(--font-mono)] text-[var(--text-secondary)]">smoothed p_up ≥ 0.85</span>.
          </p>
          <p className="text-xs text-[var(--text-faint)] font-[family-name:var(--font-mono)] mt-2">
            {pos.regime_reason}
          </p>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Сигнал {pos.signal} (p_up = {(pos.p_up * 100).toFixed(0)}%) — недостаточно сильный для входа.
          </p>
        </div>
      </div>
    </section>
  );
}

function PosStat({
  icon, label, value, sub, highlight = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
        {icon}
        <span>{label}</span>
      </div>
      <div
        className={`mt-1.5 font-[family-name:var(--font-mono)] text-lg tabular-nums ${
          highlight ? "text-[var(--text-primary)]" : "text-[var(--text-secondary)]"
        }`}
      >
        {value}
      </div>
      <div className="text-[11px] text-[var(--text-faint)] mt-0.5 font-[family-name:var(--font-mono)]">
        {sub}
      </div>
    </div>
  );
}

function MainMetrics({ metrics }: { metrics: MetricsResponse }) {
  const cards = [
    {
      label: "Накопленная доходность",
      value: formatPct(metrics.total_return * 100, 1),
      sub: `за ${metrics.period_years.toFixed(1)} года`,
      color: metrics.total_return > 0 ? "#10b981" : "#f43f5e",
    },
    {
      label: "Угадывает направление",
      value: `${Math.round(metrics.win_rate * 100)}%`,
      sub: `${metrics.n_trades} сделок всего`,
      color: "#fafafa",
    },
    {
      label: "Максимальная просадка",
      value: formatPct(metrics.max_drawdown * 100, 1),
      sub: `Sharpe ${metrics.sharpe.toFixed(2)}`,
      color: "#f43f5e",
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {cards.map((c, idx) => (
        <div
          key={c.label}
          className="lift-on-hover rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] px-6 py-7"
          style={{ animation: `fadeInUp 0.5s ease-out ${idx * 0.08}s both` }}
        >
          <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            {c.label}
          </div>
          <div
            className="mt-4 font-[family-name:var(--font-mono)] text-4xl font-medium tracking-tight tabular-nums"
            style={{ color: c.color }}
          >
            {c.value}
          </div>
          <div className="mt-2 text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
            {c.sub}
          </div>
        </div>
      ))}
    </div>
  );
}

function TradesList({ history }: { history: HistoryResponse }) {
  if (!history.trades.length) {
    return <p className="text-sm text-[var(--text-muted)]">Нет данных по сделкам.</p>;
  }

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-medium tracking-tight">Последние закрытые сделки</h2>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Показано {history.trades.length} из {history.n_trades}
        </p>
      </div>
      <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] overflow-hidden">
        <table className="w-full text-sm">
          <thead className="text-[11px] uppercase tracking-widest text-[var(--text-faint)] border-b border-[var(--border)]">
            <tr>
              <th className="text-left px-5 py-3 font-normal">Открыта</th>
              <th className="text-left px-5 py-3 font-normal">Закрыта</th>
              <th className="text-right px-5 py-3 font-normal">Вход</th>
              <th className="text-right px-5 py-3 font-normal">Выход</th>
              <th className="text-right px-5 py-3 font-normal">Дней</th>
              <th className="text-right px-5 py-3 font-normal">Результат</th>
            </tr>
          </thead>
          <tbody className="font-[family-name:var(--font-mono)] divide-y divide-[var(--border-soft)] tabular-nums">
            {history.trades.map((t, idx) => (
              <tr
                key={`${t.entry_date}-${t.exit_date}`}
                className="hover:bg-[var(--bg-subtle)] transition-colors"
                style={{ animation: `fadeInUp 0.4s ease-out ${idx * 0.03}s both` }}
              >
                <td className="px-5 py-3 text-[var(--text-secondary)] text-xs">{t.entry_date}</td>
                <td className="px-5 py-3 text-[var(--text-secondary)] text-xs">{t.exit_date}</td>
                <td className="px-5 py-3 text-right text-[var(--text-primary)]">${t.entry_price.toFixed(2)}</td>
                <td className="px-5 py-3 text-right text-[var(--text-primary)]">${t.exit_price.toFixed(2)}</td>
                <td className="px-5 py-3 text-right text-[var(--text-muted)] text-xs">{t.hold_days}</td>
                <td
                  className="px-5 py-3 text-right font-medium"
                  style={{ color: t.net_return > 0 ? "#10b981" : "#f43f5e" }}
                >
                  {formatPct(t.net_return * 100, 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
