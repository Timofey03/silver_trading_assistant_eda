/**
 * /history — открытая позиция + метрики + таблица сделок.
 */
import {
  api,
  type MetricsResponse,
  type HistoryResponse,
} from "@/lib/api";
import { formatPct } from "@/lib/utils";
import HistoryDashboard from "@/components/HistoryDashboard";

export const revalidate = 60;

async function safeApi<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try { return await fn(); } catch { return fallback; }
}

export default async function HistoryPage() {
  const [metrics, history] = await Promise.all([
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

      <HistoryDashboard initialMetrics={metrics} />
      <TradesList history={history} />
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
