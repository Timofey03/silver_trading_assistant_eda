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

export interface TinkoffPosition {
  figi: string;
  instrument_type: string;
  qty: number;
  avg_price: number;
  current_price: number;
  yield_rub: number;
}

export interface TinkoffBalance {
  connected: boolean;
  total_rub: number;
  expected_yield_rub: number;
  free_cash_rub?: number;
  open_positions: number;
  positions?: TinkoffPosition[];
  error: string;
}

export interface OrderRequest {
  direction: "BUY" | "SELL";
  lots?: number;
  ticker?: string;
}

export interface OrderResponse {
  success: boolean;
  order_id: string;
  executed_lots: number;
  executed_price: number;
  direction: string;
  figi: string;
  error: string;
}

export interface OpenPositionResponse {
  is_open: boolean;
  entry_date: string;
  entry_price: number;
  current_price: number;
  unrealized_return: number;
  days_held: number;
  max_hold_days: number;
  trail_pct: number;
  stop_price: number;
  target_close: number;
  signal: string;
  p_up: number;
  source: string;
  regime_allows_trade?: boolean;
  regime_reason?: string;
}

export interface EquityPoint {
  date: string;
  model: number;
  buy_hold: number;
}

export interface EquityResponse {
  points: EquityPoint[];
  model_final: number;
  buy_hold_final: number;
  outperformance_pp: number;
  period_start: string;
  period_end: string;
}

export interface MonthlyCell {
  year: number;
  month: number;
  return_pct: number;
  n_trades: number;
}

export interface MonthlyResponse {
  cells: MonthlyCell[];
  years: number[];
  best_month: number;
  worst_month: number;
  best_year: number;
  worst_year: number;
  avg_month: number;
}

export interface ExperimentMetrics {
  id: string;
  name: string;
  description: string;
  stage: string;
  sharpe: number;
  sortino: number;
  total_return: number;
  annual_return: number;
  max_dd: number;
  win_rate: number;
  n_trades: number;
  profit_factor: number;
  period_years: number;
  available: boolean;
}

export interface EvolutionResponse {
  experiments: ExperimentMetrics[];
  best_sharpe: string;
  best_return: string;
}

export interface PositionRecord {
  id: string;
  ticker: string;
  figi: string;
  opened_at: string;
  entry_price: number;
  lots: number;
  lot_size_g: number;
  peak_price: number;
  source: string;
  current_price: number;
  days_held: number;
  unrealized_pnl_pct: number;
  market_pnl_pct?: number;
  market_entry_price?: number;
  market_current_price?: number;
  advice: "HOLD" | "SELL";
  advice_reason: string;
}

export interface PositionsResponse {
  positions: PositionRecord[];
  master_signal: "BUY" | "WAIT" | "AVOID";
  master_reason: string;
  master_p_up: number;
  n_open: number;
  can_buy: boolean;
}

export interface OpenPositionAPIResponse {
  success: boolean;
  position?: PositionRecord;
  tinkoff_order_id: string;
  executed_price: number;
  error: string;
}

export interface ClosePositionResponse {
  success: boolean;
  closed_at: string;
  exit_price: number;
  realized_pnl_pct: number;
  tinkoff_order_id: string;
  error: string;
}

export interface FxRates {
  usd_silver: number;
  usdrub: number;
  rub_silver: number;
  usdrub_change_5d_pct: number;
  fx_volatility_flag: boolean;
  source: string;
  last_update: string;
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
  position: () => fetchJson<OpenPositionResponse>("/api/position"),
  fx:       () => fetchJson<FxRates>("/api/fx"),
  equity:   (period: "1m" | "3m" | "6m" | "1y" | "3y" | "all" = "all") =>
    fetchJson<EquityResponse>(`/api/equity?period=${period}`),
  monthly:  () => fetchJson<MonthlyResponse>("/api/monthly"),
  evolution: () => fetchJson<EvolutionResponse>("/api/evolution"),
  positions: () => fetchJson<PositionsResponse>("/api/positions"),
  openPosition: (lots = 1, ticker = "SLVRUBF") =>
    fetch(`${API_BASE}/api/positions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lots, ticker }),
    }).then((r) => r.json() as Promise<OpenPositionAPIResponse>),
  closePosition: (id: string) =>
    fetch(`${API_BASE}/api/positions/${id}`, { method: "DELETE" })
      .then((r) => r.json() as Promise<ClosePositionResponse>),
  order:    (req: OrderRequest) =>
    fetch(`${API_BASE}/api/tinkoff/order`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        direction: req.direction,
        lots:      req.lots ?? 1,
        ticker:    req.ticker ?? "SLV",
      }),
    }).then((r) => r.json() as Promise<OrderResponse>),
};
