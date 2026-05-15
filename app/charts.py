"""
app/charts.py — Plotly чарты для всех страниц.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "bg":         "rgba(0,0,0,0)",
    "grid":       "rgba(255,255,255,0.1)",
    "text":       "#e0e0e0",
    "primary":    "#00BCD4",
    "buy":        "#00C853",
    "sell":       "#D32F2F",
    "hold":       "#9E9E9E",
    "bnh":        "#FFB300",
    "drawdown":   "#FF6F61",
    "volume":     "rgba(0,188,212,0.25)",
    "candle_up":  "#26A69A",
    "candle_dn":  "#EF5350",
}


def _base_layout(title: str = "", height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=COLORS["text"], size=18)),
        height=height,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor=COLORS["grid"], showline=False),
        yaxis=dict(gridcolor=COLORS["grid"], showline=False),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )


# =============================================================================
# 1. Equity curve: strategy vs BnH
# =============================================================================

def equity_curve(
    trades: pd.DataFrame,
    bnh_series: pd.Series,
    title: str = "Equity curve: стратегия vs Buy-and-Hold",
) -> go.Figure:
    fig = go.Figure()

    # BnH equity (нормализуем на 1.0 в начале)
    if not bnh_series.empty:
        bnh = bnh_series / bnh_series.iloc[0]
        fig.add_trace(go.Scatter(
            x=bnh.index, y=bnh.values,
            mode="lines", name="Buy-and-Hold",
            line=dict(color=COLORS["bnh"], width=2, dash="dot"),
        ))

    # Strategy equity (compound на каждом trade)
    if not trades.empty:
        t = trades.sort_values("exit_date").copy()
        t["exit_date"] = pd.to_datetime(t["exit_date"])
        eq = np.cumprod(1.0 + t["net_return"].astype(float).values)
        fig.add_trace(go.Scatter(
            x=t["exit_date"], y=eq,
            mode="lines+markers", name="Стратегия v25",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8),
        ))

    fig.update_layout(**_base_layout(title))
    fig.update_yaxes(title_text="Equity (1.0 = начальный капитал)")
    return fig


# =============================================================================
# 2. Свечной график с сигналами
# =============================================================================

def candlestick_with_signals(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    title: str = "Серебро + сигналы",
    height: int = 600,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.03,
        subplot_titles=("Цена", "p_up (уверенность модели)"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["silver_open"], high=df["silver_high"],
        low=df["silver_low"],   close=df["silver_close"],
        name="SLV",
        increasing_line_color=COLORS["candle_up"],
        decreasing_line_color=COLORS["candle_dn"],
        showlegend=False,
    ), row=1, col=1)

    # Сигналы как маркеры
    if not decisions.empty:
        buys = decisions[decisions["signal_long"] == "BUY"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["silver_close"] * 0.97,
                mode="markers", name="🟢 BUY",
                marker=dict(symbol="triangle-up", size=14,
                            color=COLORS["buy"],
                            line=dict(width=1, color="white")),
            ), row=1, col=1)
        shorts = decisions[decisions["signal_short"] == "SHORT"] \
            if "signal_short" in decisions.columns else pd.DataFrame()
        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts.index, y=shorts["silver_close"] * 1.03,
                mode="markers", name="🔴 SHORT",
                marker=dict(symbol="triangle-down", size=14,
                            color=COLORS["sell"],
                            line=dict(width=1, color="white")),
            ), row=1, col=1)

    # p_up
    if "p_up" in decisions.columns:
        fig.add_trace(go.Scatter(
            x=decisions.index, y=decisions["p_up"],
            mode="lines", name="p_up",
            line=dict(color=COLORS["primary"], width=1.5),
        ), row=2, col=1)
        # Threshold line
        fig.add_hline(y=0.55, line_dash="dash",
                      line_color=COLORS["buy"],
                      annotation_text="threshold 0.55",
                      annotation_position="right",
                      row=2, col=1)
        fig.add_hline(y=0.50, line_dash="dot",
                      line_color="rgba(255,255,255,0.3)",
                      row=2, col=1)

    fig.update_layout(**_base_layout(title, height=height))
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Цена ($)", row=1, col=1)
    fig.update_yaxes(title_text="p_up", range=[0, 1], row=2, col=1)
    return fig


# =============================================================================
# 3. Drawdown chart
# =============================================================================

def drawdown_chart(equity: np.ndarray, dates: Optional[pd.DatetimeIndex] = None,
                   title: str = "Просадка") -> go.Figure:
    if len(equity) == 0:
        return go.Figure()
    running_max = np.maximum.accumulate(equity)
    dd = (equity / running_max - 1.0) * 100

    x = dates if dates is not None else np.arange(len(dd))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=dd, mode="lines", fill="tozeroy",
        line=dict(color=COLORS["drawdown"], width=1),
        fillcolor="rgba(255,111,97,0.3)",
        name="Drawdown %",
    ))
    fig.update_layout(**_base_layout(title))
    fig.update_yaxes(title_text="Drawdown (%)")
    return fig


# =============================================================================
# 4. Portfolio donut chart
# =============================================================================

def portfolio_donut(cash: float, futures: float, shares: float = 0,
                    etf: float = 0) -> go.Figure:
    labels, values, colors = [], [], []
    for name, val, col in [
        ("Cash (RUB)", cash, COLORS["primary"]),
        ("Futures",    futures, COLORS["buy"]),
        ("Shares",     shares, COLORS["bnh"]),
        ("ETF",        etf, "#9C27B0"),
    ]:
        if val > 0:
            labels.append(name)
            values.append(val)
            colors.append(col)

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=2)),
        textinfo="label+percent",
    ))
    total = sum(values)
    fig.update_layout(
        **_base_layout("Структура портфеля", height=380),
        annotations=[dict(
            text=f"{total/1000:.1f}k ₽",
            x=0.5, y=0.5, font=dict(size=24, color=COLORS["text"]),
            showarrow=False,
        )],
    )
    return fig


# =============================================================================
# 5. Trade scatter (P&L per trade)
# =============================================================================

def trades_scatter(trades: pd.DataFrame, title: str = "P&L каждой сделки") -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(**_base_layout(title))
        return fig

    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["ret_pct"] = t["net_return"] * 100
    t["color"] = t["ret_pct"].apply(
        lambda x: COLORS["buy"] if x > 0 else COLORS["sell"]
    )

    fig.add_trace(go.Bar(
        x=t["entry_date"], y=t["ret_pct"],
        marker_color=t["color"],
        name="Net return %",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P&L: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
    fig.update_layout(**_base_layout(title))
    fig.update_yaxes(title_text="Return (%)")
    return fig


# =============================================================================
# 6. Bootstrap CI fan chart
# =============================================================================

def bootstrap_fan(boot_df: pd.DataFrame, metric: str = "total_return",
                  title: str = "Bootstrap 95% CI") -> go.Figure:
    fig = go.Figure()
    if boot_df.empty:
        fig.update_layout(**_base_layout(title))
        return fig

    splits = boot_df["split"].tolist()
    lower = boot_df[f"tr_lower" if metric == "total_return" else f"{metric}_lower"].values * 100
    median = boot_df[f"tr_median" if metric == "total_return" else f"{metric}_median"].values * 100
    upper = boot_df[f"tr_upper" if metric == "total_return" else f"{metric}_upper"].values * 100

    fig.add_trace(go.Scatter(
        x=splits, y=upper, mode="lines", name="Upper 97.5%",
        line=dict(color="rgba(0,188,212,0.4)", width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=splits, y=lower, mode="lines", name="95% CI",
        line=dict(color="rgba(0,188,212,0.4)", width=0),
        fill="tonexty", fillcolor="rgba(0,188,212,0.25)",
    ))
    fig.add_trace(go.Scatter(
        x=splits, y=median, mode="lines+markers", name="Median",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=12),
    ))
    fig.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.5)

    fig.update_layout(**_base_layout(title))
    fig.update_yaxes(title_text="Total return (%)")
    return fig
