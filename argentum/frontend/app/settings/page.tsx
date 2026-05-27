/**
 * /settings — Tinkoff + параметры + о модели.
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
      <div className="space-y-2">
        <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          Settings
        </div>
        <h1 className="text-3xl font-medium tracking-tight">Настройки</h1>
        <p className="text-sm text-[var(--text-muted)]">
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
    <section className="space-y-3">
      <h2 className="text-base font-medium tracking-tight text-[var(--text-secondary)]">
        Tinkoff Invest
      </h2>
      <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] px-6 py-6">
        {tinkoff.connected ? (
          <div className="space-y-5">
            <div className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-sm text-[var(--text-primary)]">
                Подключено к sandbox-аккаунту
              </span>
            </div>
            <div className="grid grid-cols-3 gap-6 pt-2">
              <StatItem label="Баланс" value={formatRub(tinkoff.total_rub)} />
              <StatItem
                label="Доходность"
                value={formatRub(tinkoff.expected_yield_rub)}
                color={tinkoff.expected_yield_rub >= 0 ? "#10b981" : "#f43f5e"}
              />
              <StatItem label="Открытых позиций" value={String(tinkoff.open_positions)} />
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-[var(--text-faint)]" />
              <span className="text-sm text-[var(--text-muted)]">Не подключено</span>
            </div>
            <p className="text-sm text-[var(--text-muted)]">
              {tinkoff.error || "Установите TINKOFF_TOKEN в .env для интеграции"}
            </p>
            <code className="block text-xs text-[var(--text-faint)] font-[family-name:var(--font-mono)] mt-2 px-3 py-2 rounded bg-[var(--bg-base)] border border-[var(--border-soft)]">
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
    <section className="space-y-3">
      <h2 className="text-base font-medium tracking-tight text-[var(--text-secondary)]">
        О модели
      </h2>
      <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] px-6 py-6 space-y-5">
        <div>
          <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            Модель
          </div>
          <div className="mt-1 font-[family-name:var(--font-mono)] text-[var(--text-primary)]">
            {metrics.model_name}{" "}
            <span className="text-[var(--text-muted)]">· multi-asset + adaptive barriers</span>
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <StatItem label="Sharpe" value={metrics.sharpe.toFixed(2)} />
          <StatItem label="Win Rate" value={`${Math.round(metrics.win_rate * 100)}%`} />
          <StatItem label="Признаков" value={String(metrics.model_features)} />
          <StatItem label="Сделок" value={String(metrics.n_trades)} />
        </div>
        <div className="pt-4 border-t border-[var(--border-soft)] flex gap-4 text-sm">
          <a
            href="https://github.com/Timofey03/silver_trading_assistant_eda"
            target="_blank"
            rel="noopener"
            className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
          >
            GitHub →
          </a>
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener"
            className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
          >
            API Docs →
          </a>
        </div>
      </div>
    </section>
  );
}

function StatItem({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
        {label}
      </div>
      <div
        className="mt-1.5 font-[family-name:var(--font-mono)] text-xl tabular-nums"
        style={{ color: color || undefined }}
      >
        {value}
      </div>
    </div>
  );
}
