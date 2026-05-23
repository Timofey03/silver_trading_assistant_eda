/**
 * Настройки — Tinkoff integration + параметры + о модели.
 */
import { api, type TinkoffBalance, type MetricsResponse } from "@/lib/api";
import { formatRub } from "@/lib/utils";

export const revalidate = 30;

async function safeApi<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try { return await fn(); } catch { return fallback; }
}

export default async function SettingsPage() {
  const [tinkoff, metrics] = await Promise.all([
    safeApi(api.tinkoff, {
      connected: false, total_rub: 0, expected_yield_rub: 0,
      open_positions: 0, error: "Backend offline",
    } as TinkoffBalance),
    safeApi(api.metrics, {
      sharpe: 0, sortino: 0, annual_return: 0, total_return: 0,
      max_drawdown: 0, profit_factor: 0, win_rate: 0, n_trades: 0,
      oos_accuracy: 0, psr: 0, period_years: 0,
      best_trade: 0, worst_trade: 0, model_name: "E3b", model_features: 30,
    } as MetricsResponse),
  ]);

  return (
    <div className="space-y-12">
      <div>
        <h1 className="text-3xl font-medium tracking-tight mb-2">
          Настройки
        </h1>
        <p className="text-sm text-neutral-500">
          Интеграции и параметры помощника
        </p>
      </div>

      <TinkoffSection tinkoff={tinkoff} />
      <ModelSection metrics={metrics} />
    </div>
  );
}

function TinkoffSection({ tinkoff }: { tinkoff: TinkoffBalance }) {
  return (
    <section>
      <h2 className="text-xl font-medium tracking-tight mb-4">
        Tinkoff Invest
      </h2>
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/30 px-6 py-6">
        {tinkoff.connected ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-sm">Подключено к sandbox-аккаунту</span>
            </div>
            <div className="grid grid-cols-3 gap-4 pt-2">
              <div>
                <div className="text-xs text-neutral-500">Баланс</div>
                <div className="font-[family-name:var(--font-mono)] text-2xl mt-1">
                  {formatRub(tinkoff.total_rub)}
                </div>
              </div>
              <div>
                <div className="text-xs text-neutral-500">Доходность</div>
                <div
                  className={`font-[family-name:var(--font-mono)] text-2xl mt-1 ${
                    tinkoff.expected_yield_rub >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {formatRub(tinkoff.expected_yield_rub)}
                </div>
              </div>
              <div>
                <div className="text-xs text-neutral-500">Открытых позиций</div>
                <div className="font-[family-name:var(--font-mono)] text-2xl mt-1">
                  {tinkoff.open_positions}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-neutral-600" />
              <span className="text-sm text-neutral-400">Не подключено</span>
            </div>
            <p className="text-sm text-neutral-500">
              {tinkoff.error || "Установите TINKOFF_TOKEN в .env для интеграции"}
            </p>
            <code className="block text-xs text-neutral-600 font-[family-name:var(--font-mono)] mt-2">
              # .env<br />
              TINKOFF_TOKEN=t.xxxxxxx...
            </code>
          </div>
        )}
      </div>
    </section>
  );
}

function ModelSection({ metrics }: { metrics: MetricsResponse }) {
  return (
    <section>
      <h2 className="text-xl font-medium tracking-tight mb-4">
        О модели
      </h2>
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/30 px-6 py-6 space-y-4">
        <div>
          <div className="text-xs text-neutral-500">Модель</div>
          <div className="mt-1 font-[family-name:var(--font-mono)]">
            {metrics.model_name} · multi-asset + adaptive barriers
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
          {[
            { label: "Sharpe", value: metrics.sharpe.toFixed(2) },
            { label: "Win Rate", value: `${Math.round(metrics.win_rate * 100)}%` },
            { label: "Признаков", value: metrics.model_features.toString() },
            { label: "Сделок", value: metrics.n_trades.toString() },
          ].map((m) => (
            <div key={m.label}>
              <div className="text-xs text-neutral-500">{m.label}</div>
              <div className="font-[family-name:var(--font-mono)] text-lg mt-1">
                {m.value}
              </div>
            </div>
          ))}
        </div>
        <div className="pt-4 border-t border-neutral-800 flex gap-4 text-sm">
          <a
            href="https://github.com/Timofey03/silver_trading_assistant_eda"
            target="_blank"
            rel="noopener"
            className="text-neutral-400 hover:text-neutral-100 transition-colors"
          >
            GitHub →
          </a>
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener"
            className="text-neutral-400 hover:text-neutral-100 transition-colors"
          >
            API Docs →
          </a>
        </div>
      </div>
    </section>
  );
}
