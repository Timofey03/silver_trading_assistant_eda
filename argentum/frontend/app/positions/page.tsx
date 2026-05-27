/**
 * /positions — мульти-position управление.
 *
 * - Главный помощник: BUY/WAIT/AVOID для открытия новой позиции
 * - Список открытых позиций с per-position HOLD/SELL advisor
 * - Sync с Tinkoff sandbox: каждая позиция = реальный ордер
 */
import { api, type PositionsResponse } from "@/lib/api";
import PositionsView from "@/components/PositionsView";

export const revalidate = 0;  // всегда свежие данные

async function safeApi<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try { return await fn(); } catch { return fallback; }
}

export default async function PositionsPage() {
  const data = await safeApi(api.positions, {
    positions: [], master_signal: "WAIT", master_reason: "загрузка…",
    master_p_up: 0, n_open: 0, can_buy: false,
  } as PositionsResponse);

  return (
    <div className="space-y-12">
      <div className="space-y-2">
        <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
          Multi-position management · Tinkoff sandbox
        </div>
        <h1 className="text-3xl font-medium tracking-tight">Мои позиции</h1>
        <p className="text-sm text-[var(--text-muted)] max-w-2xl">
          Каждая открытая позиция отслеживается независимо. Главный помощник
          решает когда открывать новую, per-position advisor — когда закрывать.
        </p>
      </div>

      <PositionsView initial={data} />
    </div>
  );
}
