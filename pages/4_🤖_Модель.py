"""Страница: здоровье модели — DSR, PSR, drift, bootstrap CI."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils import (
    load_pnl_summary, load_dsr_psr, load_bootstrap, load_drift_report,
    load_policy, get_current_signal,
    pct, inject_styles, top_signal_badge,
)
from app.charts import bootstrap_fan

st.set_page_config(page_title="Модель", page_icon="🤖", layout="wide")
inject_styles()

st.markdown("# 🤖 Здоровье модели")
st.caption("Насколько можно доверять текущим сигналам — статистические тесты")

top_signal_badge(get_current_signal())

# =============================================================================
# Светофор health-check
# =============================================================================

st.markdown("### 🚦 Светофор")

pnl = load_pnl_summary()
dsr = load_dsr_psr()
boot = load_bootstrap()

fwd_pnl  = pnl[pnl["split"] == "forward"].iloc[0]  if not pnl.empty and (pnl["split"] == "forward").any() else None
fwd_dsr  = dsr[dsr["split"] == "forward"].iloc[0]  if not dsr.empty and (dsr["split"] == "forward").any() else None
fwd_boot = boot[boot["split"] == "forward"].iloc[0] if not boot.empty and (boot["split"] == "forward").any() else None


def _light(value, threshold, kind: str = "greater_better") -> str:
    if value is None or pd.isna(value):
        return "⚪"
    if kind == "greater_better":
        if value >= threshold:
            return "🟢"
        elif value >= threshold * 0.7:
            return "🟡"
        else:
            return "🔴"
    else:  # less_better (drawdown)
        if value >= threshold:  # threshold is negative number like -0.25
            return "🟢"
        elif value >= threshold * 1.3:
            return "🟡"
        else:
            return "🔴"


cols = st.columns(5)
metrics = []
if fwd_pnl is not None:
    metrics = [
        ("CAGR",  fwd_pnl.get("cagr"),       0.0,    "greater_better", lambda x: f"{x*100:.1f}%"),
        ("MaxDD", fwd_pnl.get("max_dd"),    -0.25,   "less_better",     lambda x: f"{x*100:.1f}%"),
        ("Sharpe", fwd_pnl.get("sharpe_ann"), 1.0,    "greater_better", lambda x: f"{x:.2f}"),
    ]
if fwd_dsr is not None:
    metrics += [
        ("PSR",  fwd_dsr.get("psr"),  0.95, "greater_better", lambda x: f"{x*100:.1f}%"),
        ("DSR",  fwd_dsr.get("dsr"),  0.70, "greater_better", lambda x: f"{x*100:.1f}%"),
    ]

for col, (name, value, thr, kind, fmt) in zip(cols, metrics):
    with col:
        light = _light(value, thr, kind)
        if value is not None and pd.notna(value):
            st.markdown(f"### {light}")
            st.metric(name, fmt(value))
            if kind == "greater_better":
                st.caption(f"норма > {fmt(thr) if not isinstance(thr, int) else thr}")
            else:
                st.caption(f"норма > {fmt(thr)}")


# =============================================================================
# Объяснение
# =============================================================================

st.markdown("### 💡 Что значит каждая метрика")

with st.expander("📚 Гайд по метрикам — кликни чтобы развернуть"):
    st.markdown("""
| Метрика | Что измеряет | Норма |
|---|---|---|
| **CAGR** (Compound Annual Growth Rate) | Средний годовой рост капитала | > 0 (любой плюс) |
| **MaxDD** (Max Drawdown) | Самая большая просадка от пика | > −25% |
| **Sharpe Ratio** | Доходность на единицу риска (волатильности) | > 1.0 — хорошо, > 2 — отлично |
| **PSR** (Probabilistic Sharpe Ratio) | Вероятность что истинный Sharpe > 0 (с учётом малой выборки) | > 95% |
| **DSR** (Deflated Sharpe Ratio) | PSR с поправкой на количество протестированных стратегий (защита от data snooping) | > 70% |

**Bootstrap 95% CI** — 95% доверительный интервал из 2000 стационарных block bootstrap симуляций.
Показывает, какой диапазон результатов вы могли бы получить при разной "удаче" внутри той же стратегии.
""")


# =============================================================================
# Bootstrap CI
# =============================================================================

st.markdown("---")
st.markdown("### 📊 Bootstrap 95% CI (диапазон возможных результатов)")
st.caption("Stationary block bootstrap, 2000 симуляций, block_len=5")

if not boot.empty:
    fig = bootstrap_fan(boot, metric="total_return",
                        title="Total return по splits (95% доверительный интервал)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Числовое представление**:")
    boot_display = boot[["split", "n_obs", "tr_lower", "tr_median", "tr_upper",
                         "shr_lower", "shr_median", "shr_upper"]].copy()
    boot_display.columns = ["Split", "N", "Total ↓",
                            "Total med.", "Total ↑",
                            "Sharpe ↓", "Sharpe med.", "Sharpe ↑"]
    for c in ["Total ↓", "Total med.", "Total ↑"]:
        boot_display[c] = boot_display[c].apply(lambda x: f"{x*100:+.1f}%")
    for c in ["Sharpe ↓", "Sharpe med.", "Sharpe ↑"]:
        boot_display[c] = boot_display[c].apply(lambda x: f"{x:.2f}")
    st.dataframe(boot_display, hide_index=True, use_container_width=True)


# =============================================================================
# Drift detection
# =============================================================================

st.markdown("---")
st.markdown("### ⚠ Drift detection — изменился ли рынок vs обучающей выборки?")

drift = load_drift_report()

# Drift over time chart — собираем из всех daily_reports
from pathlib import Path
import os
drift_history = []
training_dir = Path("daily_reports") / "training"
if training_dir.exists():
    for date_dir in sorted(training_dir.iterdir()):
        if date_dir.is_dir():
            drift_file = date_dir / "feature_drift_train_vs_recent.csv"
            if drift_file.exists():
                try:
                    df_d = pd.read_csv(drift_file)
                    if "drift" in df_d.columns:
                        n_total = len(df_d)
                        n_drift = int(df_d["drift"].sum())
                        drift_history.append({
                            "date":       date_dir.name,
                            "drift_rate": n_drift / max(n_total, 1),
                            "n_drifted":  n_drift,
                            "n_total":    n_total,
                        })
                except Exception:
                    pass

if drift_history:
    import plotly.graph_objects as go
    history_df = pd.DataFrame(drift_history)
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=history_df["date"], y=history_df["drift_rate"] * 100,
        mode="lines+markers", name="Drift rate %",
        line=dict(color="#FF6F61", width=3),
        marker=dict(size=10),
        fill="tozeroy", fillcolor="rgba(255,111,97,0.2)",
    ))
    fig_drift.add_hline(y=50, line_dash="dash", line_color="orange",
                        annotation_text="50% — стоит мониторить")
    fig_drift.add_hline(y=20, line_dash="dot", line_color="green",
                        annotation_text="20% — зона комфорта")
    fig_drift.update_layout(
        title=dict(text="Drift rate во времени (% фичей с p<0.01)", font=dict(size=16)),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title="Drift rate (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(title="Дата training run", gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_drift, use_container_width=True)
    st.caption(f"Истории {len(history_df)} daily runs. Тренд показывает, "
                "стабилизируется ли модель vs изменяющегося рынка.")

if drift.empty:
    st.info("Drift отчёт ещё не сгенерирован. Появится после следующего daily run.")
else:
    n_checked = len(drift)
    n_drift   = drift["drift"].sum() if "drift" in drift.columns else 0
    drift_rate = n_drift / max(n_checked, 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Проверено фичей", n_checked)
    with col2:
        st.metric("С drift (p<0.01)", int(n_drift),
                  delta=f"{drift_rate:.0%}", delta_color="inverse")
    with col3:
        if drift_rate > 0.5:
            st.error("⚠ Высокий drift — модель работает на изменённом рынке")
        elif drift_rate > 0.2:
            st.warning("🟡 Умеренный drift — стоит мониторить")
        else:
            st.success("✅ Низкий drift — модель в зоне комфорта")

    with st.expander("🔍 Топ-15 дрейфующих фичей"):
        top = drift[drift["drift"] == True].head(15) if "drift" in drift.columns else drift.head(15)
        if not top.empty:
            top_display = top.copy()
            for c in ["mean_train", "mean_recent"]:
                if c in top_display.columns:
                    top_display[c] = top_display[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
            top_display.columns = ["Фича", "N train", "N recent", "p-value",
                                    "Drift", "Mean train", "Mean recent"][:len(top_display.columns)]
            st.dataframe(top_display, hide_index=True, use_container_width=True)


# =============================================================================
# Policy
# =============================================================================

st.markdown("---")
st.markdown("### ⚙ Текущая policy")

policy = load_policy()
if policy:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("up_threshold", policy.get("up_threshold", "—"))
        st.metric("cooldown (дней)", policy.get("cooldown", "—"))
    with col2:
        st.metric("Valid buys (training)", policy.get("valid_buys", "—"))
        st.metric("Valid precision", f"{policy.get('valid_precision', 0)*100:.1f}%")

    st.caption("Policy выбрана автоматически на этапе CPCV training: "
               "оптимизирует precision × volume на valid split")


# =============================================================================
# Per-split PnL summary
# =============================================================================

st.markdown("---")
st.markdown("### 📋 P&L по всем splits")

if not pnl.empty:
    show = pnl.copy()
    for c in ["v22_honest_total", "v25_honest_total", "improvement_pp",
              "true_bnh", "vs_bnh", "cagr", "max_dd"]:
        if c in show.columns:
            show[c] = show[c].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
    show["sharpe_ann"] = show["sharpe_ann"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    show["calmar"]     = show["calmar"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    show.columns = [c.replace("_", " ") for c in show.columns]
    st.dataframe(show, hide_index=True, use_container_width=True)
