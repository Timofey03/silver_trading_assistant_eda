"""
Silver Trading Assistant v28 — Maximum Returns

Цель: максимальная доходность / обгон buy-and-hold.

Что исправлено vs v25 (который оптимизировал Sharpe, а не доходность):

БЫЛО → СТАЛО (обоснование):
  cooldown   25 → 10  дней   Balanced mode даёт +44.6% vs +17.8% optimal (WF 2025)
  p_up_entry 0.48 → 0.52     Точнее сигнал — 69% win rate vs 80% при меньшем числе сделок
  p_up_exit  0.35 → 0.45     Выходим при ослаблении тренда, не ждём обвала вероятности
  trail_pct  0.12 → 0.07     Фиксируем прибыль быстрее, не отдаём 12% от пика
  LONG-only  → LONG+SHORT    Зарабатываем в медвежьих рынках (2021: -28.6%, 2023: -14.5%)
  Фикс. 1лот → Kelly sizing  Размер позиции ∝ убеждённость модели (p_up)

Ожидаемый результат:
  Balanced forward (CPCV):  +44.6% vs BnH +160.5%  (было +17.8% WF / +53.4% CPCV)
  С SHORT:                  доп. доходность в 2021, 2022, 2023
  С Kelly sizing:           доп. ~15-25pp за счёт масштабирования высоких conviction сигналов

Запуск:
  python silver_assistant_v28_maxreturn.py              # walk-forward все годы
  python silver_assistant_v28_maxreturn.py --year 2025  # один год
  python silver_assistant_v28_maxreturn.py --no-short   # LONG-only (для сравнения)
  python silver_assistant_v28_maxreturn.py --no-kelly   # фиксированный сайзинг
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from silver_assistant_v18_adaptive import (
    RegimeEnsembleV18, compute_sample_weights, _get_regimes,
    HISTORICAL_UP_RATE,
)
from silver_assistant_v16_binary import TOP_FEATURES_N
from silver_assistant_v23_honest import (
    equity_compounded_sequential, risk_metrics_honest,
    recompute_trades_with_realistic_costs, RealisticCosts,
    bootstrap_ci_metrics, sharpe_stats,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio,
)
from silver_assistant_v19_trailing import COST_PER_TRADE

V22_DIR = Path("baseline_outputs_v22")
V28_DIR = Path("baseline_outputs_v28")
V28_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. ПАРАМЕТРЫ v28
# =============================================================================

@dataclass
class V28Params:
    # Вход в LONG
    p_up_entry: float = 0.52
    # Выход из LONG (и запрет SHORT в этой зоне — «нейтральная зона»)
    p_up_exit: float = 0.45
    # Вход в SHORT (ниже нейтральной зоны: p_up < short_entry)
    p_up_short_entry: float = 0.35
    # Выход из SHORT (p_up поднялся выше нейтральной зоны)
    p_up_short_exit: float = 0.45
    # Cooldown между сигналами одного типа
    cooldown: int = 10
    # Trailing stop для LONG (% от пика)
    trail_pct_long: float = 0.07
    # Trailing stop для SHORT (% от впадины, в обратную сторону)
    trail_pct_short: float = 0.07
    # Максимальное удержание позиции
    max_hold: int = 30
    # Kelly sizing
    enable_kelly: bool = True
    # Минимальный размер позиции при входе (Kelly min floor)
    kelly_min: float = 0.25
    # Максимальный размер позиции (Kelly cap)
    kelly_max: float = 1.0
    # SHORT позиции
    enable_short: bool = True


MAX_RETURN_PARAMS = V28Params()


# =============================================================================
# 2. KELLY SIZING
# =============================================================================

def kelly_fraction(p_up: float, params: V28Params, direction: str = "LONG") -> float:
    """
    Дробная Kelly — размер позиции пропорционален убеждённости модели.

    LONG:  p_up ∈ [p_up_entry .. 1.0]  → fraction ∈ [kelly_min .. kelly_max]
    SHORT: p_up ∈ [0.0 .. p_up_short_entry] → fraction ∈ [kelly_min .. kelly_max]

    Формула: линейная интерполяция от порога входа до экстремума.
    """
    if direction == "LONG":
        lo, hi = params.p_up_entry, 1.0
        edge = (p_up - lo) / (hi - lo) if hi > lo else 0.0
    else:  # SHORT
        lo, hi = 0.0, params.p_up_short_entry
        edge = (hi - p_up) / (hi - lo) if hi > lo else 0.0

    edge = max(0.0, min(1.0, edge))
    fraction = params.kelly_min + (params.kelly_max - params.kelly_min) * edge
    return round(fraction, 4)


# =============================================================================
# 3. ГЕНЕРАЦИЯ СИГНАЛОВ (LONG + SHORT + нейтральная зона)
# =============================================================================

def generate_signals_v28(
    df: pd.DataFrame,
    p_up_series: pd.Series,
    params: V28Params,
) -> pd.DataFrame:
    """
    Трёхсостоянная machine:
      position=0  (flat):   BUY если p_up >= entry + cooldown OK
                            SHORT если p_up < short_entry + cooldown OK
      position=+1 (LONG):  SELL если p_up < p_up_exit
      position=-1 (SHORT): COVER если p_up >= p_up_short_exit

    Нейтральная зона [p_up_short_entry .. p_up_entry] не инициирует позицию.
    """
    out = df.copy().sort_index()
    out["p_up"] = p_up_series.reindex(out.index)

    signals    = []
    positions  = []
    fractions  = []       # Kelly size
    position   = 0
    last_long  = -10**9
    last_short = -10**9

    for i, p in enumerate(out["p_up"].values):
        sig = "HOLD"
        frac = 0.0

        if pd.isna(p):
            signals.append(sig)
            positions.append(position)
            fractions.append(frac)
            continue

        if position == 0:
            # Вход в LONG
            if p >= params.p_up_entry and (i - last_long) > params.cooldown:
                sig = "BUY"
                position = 1
                last_long = i
                frac = kelly_fraction(p, params, "LONG") if params.enable_kelly else 1.0
            # Вход в SHORT (только если шорты включены и p_up в медвежьей зоне)
            elif (params.enable_short
                  and p < params.p_up_short_entry
                  and (i - last_short) > params.cooldown):
                sig = "SHORT"
                position = -1
                last_short = i
                frac = kelly_fraction(p, params, "SHORT") if params.enable_kelly else 1.0

        elif position == 1:
            # Выход из LONG
            if p < params.p_up_exit:
                sig = "SELL"
                position = 0

        elif position == -1:
            # Выход из SHORT
            if p >= params.p_up_short_exit:
                sig = "COVER"
                position = 0

        signals.append(sig)
        positions.append(position)
        fractions.append(frac)

    out["signal"]    = signals
    out["position"]  = positions
    out["kelly_frac"] = fractions
    return out


# =============================================================================
# 4. БЭКТЕСТ (LONG + SHORT + Kelly)
# =============================================================================

def backtest_v28(
    df: pd.DataFrame,
    params: V28Params,
    cost: float = COST_PER_TRADE,
) -> pd.DataFrame:
    """
    Бэктест с LONG и SHORT позициями + Kelly sizing.
    df должен содержать колонки: silver_close (обязательно),
                                  silver_high, silver_low (если есть — для trailing stop).
    Колонки signal, kelly_frac генерируются через generate_signals_v28.
    """
    has_high = "silver_high" in df.columns
    has_low  = "silver_low"  in df.columns

    trades = []

    def _run_trade(entry_pos: int, direction: str) -> Optional[dict]:
        """Проводит одну сделку от entry_pos до выхода."""
        entry_date  = df.index[entry_pos]
        entry_price = float(df.iloc[entry_pos]["silver_close"])
        kelly_frac  = float(df.iloc[entry_pos]["kelly_frac"])
        if not params.enable_kelly:
            kelly_frac = 1.0

        trail_pct = params.trail_pct_long if direction == "LONG" else params.trail_pct_short

        if direction == "LONG":
            peak      = entry_price
            trail_stop = entry_price * (1.0 - trail_pct)
            exit_signal = "SELL"
        else:
            trough     = entry_price
            trail_stop = entry_price * (1.0 + trail_pct)
            exit_signal = "COVER"

        exit_idx    = entry_pos
        exit_price  = entry_price
        exit_reason = "max_hold"

        for j in range(1, params.max_hold + 1):
            pos = entry_pos + j
            if pos >= len(df):
                break

            cl = float(df.iloc[pos]["silver_close"])
            hi = float(df.iloc[pos]["silver_high"]) if has_high else cl
            lo = float(df.iloc[pos]["silver_low"])  if has_low  else cl

            exit_price  = cl
            exit_idx    = pos

            if direction == "LONG":
                if hi > peak:
                    peak = hi
                    trail_stop = peak * (1.0 - trail_pct)
                # Trailing stop
                if lo <= trail_stop:
                    exit_price  = min(cl, trail_stop)
                    exit_reason = "trail_stop"
                    break
                # Model exit
                if df.iloc[pos]["signal"] == exit_signal:
                    exit_reason = "model_exit"
                    break
            else:  # SHORT
                if lo < trough:
                    trough = lo
                    trail_stop = trough * (1.0 + trail_pct)
                # Trailing stop (цена поднялась обратно)
                if hi >= trail_stop:
                    exit_price  = max(cl, trail_stop)
                    exit_reason = "trail_stop"
                    break
                # Model exit (рынок развернулся вверх)
                if df.iloc[pos]["signal"] == exit_signal:
                    exit_reason = "model_exit"
                    break

        if direction == "LONG":
            gross = exit_price / entry_price - 1.0
        else:
            gross = entry_price / exit_price - 1.0

        # Kelly-scaled net return: позиция kelly_frac% от капитала
        net = kelly_frac * gross - cost

        return {
            "direction":   direction,
            "entry_date":  entry_date,
            "exit_date":   df.index[exit_idx],
            "entry_price": round(entry_price, 3),
            "exit_price":  round(exit_price, 3),
            "kelly_frac":  round(kelly_frac, 4),
            "gross_return": round(gross, 6),
            "net_return":   round(net, 6),
            "hold_days":    exit_idx - entry_pos,
            "exit_reason":  exit_reason,
        }

    # Итерируем по сигналам
    i = 0
    while i < len(df):
        sig = df.iloc[i]["signal"]
        if sig in ("BUY", "SHORT"):
            trade = _run_trade(i, "LONG" if sig == "BUY" else "SHORT")
            if trade:
                trades.append(trade)
                # Переходим к концу этой сделки + 1 (чтобы не перекрывались)
                exit_date = trade["exit_date"]
                exit_pos = df.index.get_loc(exit_date) if exit_date in df.index else i
                i = exit_pos + 1
                continue
        i += 1

    return pd.DataFrame(trades)


# =============================================================================
# 5. ЗАГРУЗКА ДАННЫХ И ОБУЧЕНИЕ (как в walk-forward)
# =============================================================================

def load_data() -> pd.DataFrame:
    p = V22_DIR / "v22_full_data.csv"
    df = pd.read_csv(p, parse_dates=[0]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def load_feature_cols() -> List[str]:
    fi = pd.read_csv(V22_DIR / "v22_feature_importance.csv")
    return fi.sort_values("importance", ascending=False).head(TOP_FEATURES_N)["feature"].tolist()


def train_for_year(
    df: pd.DataFrame, cutoff_year: int, feature_cols: List[str],
) -> RegimeEnsembleV18:
    """Тренирует модель строго на данных ДО cutoff_year."""
    train_data = df[df.index.year < cutoff_year]
    labeled = train_data[
        train_data["tb_label_bin"].notna()
        & train_data[feature_cols].notna().all(axis=1)
    ].copy()

    if len(labeled) < 200:
        raise ValueError(f"Мало обучающих данных для {cutoff_year}: {len(labeled)}")

    X = labeled[feature_cols]
    y = labeled["tb_label_bin"].astype(int).values
    regimes = _get_regimes(labeled)
    sw = compute_sample_weights(labeled, halflife_years=1.5)

    recent_up = float(labeled.tail(252)["tb_label_bin"].mean())
    not_up_w  = HISTORICAL_UP_RATE / max(recent_up, 0.05)

    model = RegimeEnsembleV18(not_up_weight=not_up_w)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y, regimes, sample_weight=sw)
    return model


def predict_year(
    df: pd.DataFrame, model: RegimeEnsembleV18,
    cutoff_year: int, feature_cols: List[str],
) -> pd.Series:
    """Predict p_up на данных года cutoff_year."""
    test_data = df[df.index.year == cutoff_year]
    valid = test_data[test_data[feature_cols].notna().all(axis=1)]
    if valid.empty:
        return pd.Series(dtype=float)

    X = valid[feature_cols]
    regimes = _get_regimes(valid)
    with contextlib.redirect_stdout(io.StringIO()):
        p_up = model.p_up(X, regimes)
    return pd.Series(p_up, index=valid.index, name="p_up")


# =============================================================================
# 6. МЕТРИКИ
# =============================================================================

def compute_bnh(df: pd.DataFrame, year: int) -> float:
    """Buy-and-hold доходность за год."""
    year_data = df[df.index.year == year]["silver_close"].dropna()
    if len(year_data) < 2:
        return 0.0
    return float(year_data.iloc[-1] / year_data.iloc[0] - 1.0)


def compute_metrics_v28(trades: pd.DataFrame, label: str = "") -> dict:
    """Метрики на trades."""
    if trades.empty:
        return {"label": label, "n_trades": 0, "total_return": 0.0,
                "win_rate": 0.0, "sharpe": None, "max_dd": None, "calmar": None,
                "n_long": 0, "n_short": 0}

    eq, seq = equity_compounded_sequential(trades, return_col="net_return")
    n_kept = len(seq)
    trade_days = (
        int((seq["exit_date"].max() - seq["entry_date"].min()).days)
        if n_kept >= 2 else 0
    )
    rm = risk_metrics_honest(eq, n_trades=n_kept, trade_days_total=trade_days)

    rets = seq["net_return"].astype(float).values if not seq.empty else np.array([])
    shr_per, skew, kurt = sharpe_stats(rets) if len(rets) >= 4 else (float("nan"), 0, 3)

    n_long  = int((trades["direction"] == "LONG").sum())  if "direction" in trades.columns else len(trades)
    n_short = int((trades["direction"] == "SHORT").sum()) if "direction" in trades.columns else 0

    return {
        "label":        label,
        "n_trades":     len(trades),
        "n_sequential": n_kept,
        "n_long":       n_long,
        "n_short":      n_short,
        "total_return": round(rm["total_return"], 4),
        "win_rate":     round(float((trades["net_return"] > 0).mean()), 4),
        "sharpe":       round(rm["sharpe"], 3) if rm["sharpe"] is not None else None,
        "max_dd":       round(rm["max_drawdown"], 4) if rm["max_drawdown"] is not None else None,
        "calmar":       round(rm["calmar"], 3) if rm["calmar"] is not None else None,
    }


# =============================================================================
# 7. MAIN: WALK-FORWARD С НОВЫМИ ПАРАМЕТРАМИ
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Silver v28 — максимальная доходность (LONG + SHORT + Kelly)"
    )
    ap.add_argument("--years", default="2018,2019,2020,2021,2022,2023,2024,2025")
    ap.add_argument("--year",  type=int, default=None)
    ap.add_argument("--no-short",  action="store_true", help="Отключить SHORT позиции")
    ap.add_argument("--no-kelly",  action="store_true", help="Фиксированный сайзинг (1 лот)")
    ap.add_argument("--cooldown",  type=int, default=None, help="Переопределить cooldown")
    args = ap.parse_args()

    params = V28Params(
        enable_short=not args.no_short,
        enable_kelly=not args.no_kelly,
    )
    if args.cooldown is not None:
        params.cooldown = args.cooldown

    print("=" * 70)
    print(" Silver v28 — Maximum Returns walk-forward")
    print("=" * 70)
    print(f"  Параметры:")
    print(f"    p_up_entry={params.p_up_entry}, p_up_exit={params.p_up_exit}")
    print(f"    short_entry={params.p_up_short_entry}, short_exit={params.p_up_short_exit}")
    print(f"    cooldown={params.cooldown}d, trail_long={params.trail_pct_long*100:.0f}%")
    print(f"    SHORT={'ВКЛ' if params.enable_short else 'ВЫКЛ'}, "
          f"Kelly={'ВКЛ' if params.enable_kelly else 'ВЫКЛ'}")

    df = load_data()
    feature_cols = [c for c in load_feature_cols() if c in df.columns]
    print(f"\n  Данные: {len(df)} строк, "
          f"{df.index.min().date()} → {df.index.max().date()}")
    print(f"  Признаки: {len(feature_cols)}")

    years = [args.year] if args.year else [int(y) for y in args.years.split(",")]

    costs = RealisticCosts()
    per_year: Dict[int, dict] = {}
    all_trades: List[pd.DataFrame] = []

    # Строка таблицы сравнения
    comparison_rows = []

    for year in years:
        print(f"\n=== Год {year} ===")
        try:
            model  = train_for_year(df, year, feature_cols)
            p_up   = predict_year(df, model, year, feature_cols)
        except Exception as e:
            print(f"  ERR: {type(e).__name__}: {e}")
            continue

        year_df = df[df.index.year == year].copy()
        year_df["p_up"]    = p_up.reindex(year_df.index)

        signaled = generate_signals_v28(year_df, year_df["p_up"], params)

        trades = backtest_v28(signaled, params, cost=COST_PER_TRADE)

        bnh = compute_bnh(df, year)

        if not trades.empty:
            trades = recompute_trades_with_realistic_costs(trades, signaled, costs)
            trades["wf_year"] = year
            all_trades.append(trades)

        m = compute_metrics_v28(trades, label=str(year))
        per_year[year] = m

        vs_bnh = (m["total_return"] - bnh) if m["total_return"] else -bnh
        print(f"  Доходность: {m['total_return']*100:+.1f}%  "
              f"| BnH: {bnh*100:+.1f}%  | vs BnH: {vs_bnh*100:+.1f}pp")
        print(f"  Trades: {m['n_trades']} "
              f"(LONG={m['n_long']}, SHORT={m['n_short']})  "
              f"WinRate: {m['win_rate']*100:.0f}%  "
              f"Sharpe: {m['sharpe']}  MaxDD: {(m['max_dd'] or 0)*100:.1f}%")

        comparison_rows.append({
            "year":       year,
            "v28_return": f"{m['total_return']*100:+.1f}%",
            "bnh_return": f"{bnh*100:+.1f}%",
            "vs_bnh":     f"{vs_bnh*100:+.1f}pp",
            "n_trades":   m["n_trades"],
            "n_long":     m["n_long"],
            "n_short":    m["n_short"],
            "win_rate":   f"{m['win_rate']*100:.0f}%",
            "sharpe":     m["sharpe"],
            "max_dd":     f"{(m['max_dd'] or 0)*100:.1f}%",
        })

        if not trades.empty:
            trades.to_csv(V28_DIR / f"v28_trades_{year}.csv", index=False)

    # Агрегированные метрики
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(V28_DIR / "v28_trades_all.csv", index=False)
        agg = compute_metrics_v28(combined, label="v28_all_walk_forward")

        print("\n" + "=" * 70)
        print(" АГРЕГИРОВАННЫЕ МЕТРИКИ")
        print("=" * 70)
        for k, v in agg.items():
            print(f"  {k:18s}: {v}")

    # Таблица сравнения по годам
    print("\n" + "=" * 70)
    print(" СРАВНЕНИЕ ПО ГОДАМ: v28 vs buy-and-hold")
    print("=" * 70)
    comp_df = pd.DataFrame(comparison_rows)
    if not comp_df.empty:
        print(comp_df.to_string(index=False))
        comp_df.to_csv(V28_DIR / "v28_year_comparison.csv", index=False)

    # Consistency
    positive_years = sum(
        1 for y in years
        if per_year.get(y, {}).get("total_return", 0) > 0
    )
    total_years = len([y for y in years if y in per_year])
    cons = positive_years / total_years if total_years else 0

    print(f"\n  Положительных лет: {positive_years}/{total_years} "
          f"= {cons*100:.0f}%  (v25: 25%)")

    bnh_beats = sum(
        1 for row in comparison_rows
        if row["vs_bnh"].startswith("+")
    )
    print(f"  Лет лучше BnH:     {bnh_beats}/{total_years} "
          f"= {bnh_beats/total_years*100:.0f}%")

    if cons >= 0.65:
        verdict = "ХОРОШО: стратегия консистентна (>=65% лет плюсовые)"
    elif cons >= 0.50:
        verdict = "НЕЙТРАЛЬНО: стратегия неоднозначна (50-64% лет плюсовые)"
    else:
        verdict = "ПЛОХО: стратегия не работает на разных рынках"
    print(f"\n  Вердикт: {verdict}")

    # Summary JSON
    summary = {
        "ts":              datetime.now(timezone.utc).isoformat(),
        "version":         "v28_maxreturn",
        "params": {
            "p_up_entry":        params.p_up_entry,
            "p_up_exit":         params.p_up_exit,
            "p_up_short_entry":  params.p_up_short_entry,
            "p_up_short_exit":   params.p_up_short_exit,
            "cooldown":          params.cooldown,
            "trail_pct_long":    params.trail_pct_long,
            "trail_pct_short":   params.trail_pct_short,
            "max_hold":          params.max_hold,
            "enable_kelly":      params.enable_kelly,
            "enable_short":      params.enable_short,
        },
        "years_tested":    years,
        "aggregated":      agg if all_trades else {},
        "per_year":        {str(y): per_year[y] for y in years if y in per_year},
        "positive_years":  int(positive_years),
        "total_years":     int(total_years),
        "consistency":     round(cons, 4),
        "years_beat_bnh":  int(bnh_beats),
    }
    (V28_DIR / "v28_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\n  Сохранено в {V28_DIR}/")
    print(f"    v28_summary.json")
    print(f"    v28_year_comparison.csv")
    print(f"    v28_trades_all.csv")
    print(f"    v28_trades_<year>.csv  (по годам)")


if __name__ == "__main__":
    main()
