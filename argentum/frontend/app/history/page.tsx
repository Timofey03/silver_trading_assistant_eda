/**
 * История — главные метрики + список последних сделок.
 * TradingView свечной график будет добавлен на след итерации.
 */
import { api, type MetricsResponse, type HistoryResponse } from "@/lib/api";
import { formatPct } from "@/lib/utils";

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
      <div>
        <h1 className="text-3xl font-medium tracking-tight mb-2">
          Как работал помощник
        </h1>
        <p className="text-sm text-neutral-500">
          Walk-forward бэктест на {metrics.period_years.toFixed(1)} лет ·{" "}
          {history.period_start} — {history.period_end}
        </p>
      </div>

      <MainMetrics metrics={metrics} />
      <TradesList history={history} />
    </div>
  );
}

function MainMetrics({ metrics }: { metrics: MetricsResponse }) {
  const cards = [
    {
      label: "Накопленная доходность",
      value: formatPct(metrics.total_return * 100, 1),
      sub: `за ${metrics.period_years.toFixed(1)} года`,
      color: metrics.total_return > 0 ? "text-emerald-400" : "text-rose-400",
    },
    {
      label: "Угадывает направление",
      value: `${Math.round(metrics.win_rate * 100)}%`,
      sub: `${metrics.n_trades} сделок`,
      color: "text-neutral-100",
    },
    {
      label: "Максимальная просадка",
      value: formatPct(metrics.max_drawdown * 100, 1),
      sub: `Sharpe ${metrics.sharpe.toFixed(2)}`,
      color: "text-rose-400",
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {cards.map((c) => (
        <div
          key={c.label}
          className="rounded-2xl border border-neutral-800 bg-neutral-900/30 px-6 py-8"
        >
          <div className="text-xs uppercase tracking-widest text-neutral-500">
            {c.label}
          </div>
          <div
            className={`mt-3 font-[family-name:var(--font-mono)] text-4xl font-medium ${c.color}`}
          >
            {c.value}
          </div>
          <div className="mt-1 text-xs text-neutral-500">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}

function TradesList({ history }: { history: HistoryResponse }) {
  if (!history.trades.length) {
    return <p className="text-sm text-neutral-500">Нет данных по сделкам.</p>;
  }

  return (
    <section>
      <h2 className="text-xl font-medium tracking-tight mb-4">
        Последние сделки
      </h2>
      <div className="rounded-xl border border-neutral-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-neutral-900/50 text-xs uppercase tracking-widest text-neutral-500">
            <tr>
              <th className="text-left px-4 py-3 font-normal">Дата</th>
              <th className="text-right px-4 py-3 font-normal">Цена входа</th>
              <th className="text-right px-4 py-3 font-normal">Цена выхода</th>
              <th className="text-right px-4 py-3 font-normal">Дней</th>
              <th className="text-right px-4 py-3 font-normal">Результат</th>
            </tr>
          </thead>
          <tbody className="font-[family-name:var(--font-mono)] divide-y divide-neutral-800">
            {history.trades.map((t) => (
              <tr key={`${t.entry_date}-${t.exit_date}`}>
                <td className="px-4 py-3 text-neutral-400">
                  {t.entry_date} → {t.exit_date}
                </td>
                <td className="px-4 py-3 text-right">${t.entry_price.toFixed(2)}</td>
                <td className="px-4 py-3 text-right">${t.exit_price.toFixed(2)}</td>
                <td className="px-4 py-3 text-right text-neutral-500">{t.hold_days}</td>
                <td
                  className={`px-4 py-3 text-right font-medium ${
                    t.net_return > 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {t.pnl_label}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
