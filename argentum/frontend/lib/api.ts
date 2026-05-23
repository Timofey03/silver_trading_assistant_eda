/**
 * API client для FastAPI backend (port 8000).
 *
 * Все типы соответствуют Pydantic моделям в backend/routers/.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

// ===========================================================================
// Types
// ===========================================================================

export interface SignalResponse {
  signal: "BUY" | "HOLD" | "SELL";
  date: string;
  close: number;
  p_up: number;
  entry_threshold: number;
  exit_threshold: number;
  trail_pct: number;
  max_hold_days: number;
  cooldown_days: number;
  alert_type?: "action" | "info";
  is_repeat?: boolean;
  previous_signal?: string;
  n_features_used?: number;
  selected_features?: string[];
  source: string;
  report_dir?: string;
}

export interface PricePoint {
  date: string;
  close: number;
}

export interface PriceResponse {
  current: number;
  previous: number;
  change_pct: number;
  currency: string;
  ticker: string;
  sparkline: PricePoint[];
  last_update: string;
}

export interface MetricsResponse {
  sharpe: number;
  sortino: number;
  annual_return: number;
  total_return: number;
  max_drawdown: number;
  profit_factor: number;
  win_rate: number;
  n_trades: number;
  oos_accuracy: number;
  psr: number;
  period_years: number;
  best_trade: number;
  worst_trade: number;
  model_name: string;
  model_features: number;
}

export interface TradeItem {
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  net_return: number;
  hold_days: number;
  exit_reason: string;
  pnl_label: string;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface HistoryResponse {
  equity_curve: EquityPoint[];
  trades: TradeItem[];
  n_trades: number;
  total_return: number;
  period_start: string;
  period_end: string;
}

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface Marker {
  time: string;
  price: number;
  type: "BUY" | "SELL";
  text?: string;
  return_pct?: number;
}

export interface CandleResponse {
  candles: Candle[];
  markers: Marker[];
  range_start: string;
  range_end: string;
}

export interface TinkoffBalance {
  connected: boolean;
  total_rub: number;
  expected_yield_rub: number;
  open_positions: number;
  error: string;
}

export interface FeatureInsight {
  feature: string;
  human_name: string;
  interpretation: string;
}

export interface ExplainResponse {
  insights: FeatureInsight[];
  model_version: string;
  last_updated: string;
}

// ===========================================================================
// Fetcher
// ===========================================================================

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`API ${path} → ${res.status}`);
  }
  return res.json();
}

// ===========================================================================
// API functions
// ===========================================================================

export const api = {
  signal:   () => fetchJson<SignalResponse>("/api/signal"),
  price:    () => fetchJson<PriceResponse>("/api/price"),
  history:  (limit = 10) => fetchJson<HistoryResponse>(`/api/history?limit=${limit}`),
  metrics:  () => fetchJson<MetricsResponse>("/api/metrics"),
  candles:  (period: "1m" | "3m" | "6m" | "1y" | "3y" | "all" = "all") =>
    fetchJson<CandleResponse>(`/api/candles?period=${period}`),
  tinkoff:  () => fetchJson<TinkoffBalance>("/api/tinkoff/balance"),
  explain:  () => fetchJson<ExplainResponse>("/api/explain"),
};
