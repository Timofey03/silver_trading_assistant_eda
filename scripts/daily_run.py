"""
scripts/daily_run.py — Главный скрипт ежедневного запуска

Что делает (последовательно):
1. Скачивает свежие OHLC + macro + COT за сегодня
2. Переобучает v25 CPCV модель
3. Генерирует сигнал на текущий день
4. Опционально: исполняет сигнал в Tinkoff sandbox через --live
5. Создаёт ДВА отдельных отчёта:
   - daily_reports/training/YYYY-MM-DD/  — метрики обучения для анализа
   - daily_reports/trading/YYYY-MM-DD/   — действие с серебром (BUY/HOLD/SELL)

Запуск:
  python scripts/daily_run.py                  # полный цикл
  python scripts/daily_run.py --skip-training  # только сигнал + paper trade
  python scripts/daily_run.py --no-paper-trade # без отправки в Tinkoff
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

V22_DIR = REPO_ROOT / "baseline_outputs_v22"
V25_DIR = REPO_ROOT / "baseline_outputs_v25"
V23_DIR = REPO_ROOT / "baseline_outputs_v23"
REPORTS_DIR = REPO_ROOT / "daily_reports"
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TRAINING_DIR = REPORTS_DIR / "training" / TODAY
TRADING_DIR  = REPORTS_DIR / "trading"  / TODAY


def _run(cmd: list, check: bool = True, timeout: int = 1800) -> tuple[int, str]:
    """Запускает подкоманду, возвращает (rc, output)."""
    print(f"\n  $ {' '.join(cmd)}")
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        if check and r.returncode != 0:
            print(f"  ❌ rc={r.returncode}")
            print(out[-2000:])
        else:
            print(f"  ✅ rc={r.returncode}, output length={len(out)}")
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


# ===========================================================================
# 1. RUN v22 DATA REFRESH (если нет свежего v22_full_data.csv)
# ===========================================================================

def refresh_data_if_stale(max_age_hours: int = 24) -> None:
    """Перезапускает v22 для обновления данных, если файл старше N часов."""
    f = V22_DIR / "v22_full_data.csv"
    if not f.exists():
        print("  v22_full_data.csv нет → полный refresh v22")
        _run([sys.executable, "silver_assistant_v22_risk_aware.py",
              "--no-wf", "--no-mh"], timeout=1800)
        return
    age_h = (datetime.now(tz=None).timestamp() - f.stat().st_mtime) / 3600.0
    if age_h > max_age_hours:
        print(f"  v22_full_data.csv устарел ({age_h:.1f}h) → refresh")
        _run([sys.executable, "silver_assistant_v22_risk_aware.py",
              "--no-wf", "--no-mh"], timeout=1800)
    else:
        print(f"  v22_full_data.csv свежий ({age_h:.1f}h), не обновляю")


# ===========================================================================
# 2. v25 CPCV RETRAIN
# ===========================================================================

def retrain_v25() -> tuple[int, str]:
    print("\n=== Шаг 2a: v25 CPCV retrain ===")
    rc, out = _run([sys.executable, "silver_assistant_v25_cpcv.py"], timeout=1800)
    print("\n=== Шаг 2b: production inference (signal на сегодня) ===")
    rc_prod, out_prod = _run(
        [sys.executable, "silver_production_inference.py"], timeout=600, check=False,
    )
    if rc_prod != 0:
        print(f"  WARN: production inference failed, will fallback to CPCV signals")
    return rc, out + "\n--- production ---\n" + out_prod


# ===========================================================================
# 3. TRAINING REPORT (детальная аналитика)
# ===========================================================================

def build_training_report(v25_output: str) -> None:
    print(f"\n=== Шаг 3: training report → {TRAINING_DIR} ===")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # 3.1 — Копируем главные csv
    artifacts = [
        ("v25_pnl_summary.csv",      V25_DIR / "v25_pnl_summary.csv"),
        ("v25_dsr_psr.csv",          V25_DIR / "v25_dsr_psr.csv"),
        ("v25_bootstrap_ci.csv",     V25_DIR / "v25_bootstrap_ci.csv"),
        ("v25_p_up_cpcv.csv",        V25_DIR / "v25_p_up_cpcv.csv"),
        ("v25_decisions.csv",        V25_DIR / "v25_decisions.csv"),
        ("v25_policy.json",          V25_DIR / "v25_policy.json"),
        ("v22_feature_importance.csv", V22_DIR / "v22_feature_importance.csv"),
    ]
    for fname, src in artifacts:
        if src.exists():
            (TRAINING_DIR / fname).write_bytes(src.read_bytes())

    # 3.2 — Подсчитываем агрегированные метрики
    summary = {
        "run_date": TODAY,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "v25_cpcv",
    }
    try:
        pnl = pd.read_csv(V25_DIR / "v25_pnl_summary.csv")
        dsr = pd.read_csv(V25_DIR / "v25_dsr_psr.csv")
        boot = pd.read_csv(V25_DIR / "v25_bootstrap_ci.csv")
        policy = json.loads((V25_DIR / "v25_policy.json").read_text(encoding="utf-8"))

        summary["pnl"] = pnl.to_dict(orient="records")
        summary["dsr_psr"] = dsr.to_dict(orient="records")
        summary["bootstrap_ci_95"] = boot.to_dict(orient="records")
        summary["policy"] = policy

        # Health-check метрики
        fwd = pnl[pnl["split"] == "forward"]
        if not fwd.empty:
            f = fwd.iloc[0]
            summary["healthcheck"] = {
                "forward_total_return": float(f.get("v25_honest_total", 0)),
                "forward_beats_bnh":    bool(f.get("vs_bnh", -1) > 0),
                "forward_sharpe":       float(f.get("sharpe_ann", 0) or 0),
                "forward_max_dd":       float(f.get("max_dd", 0) or 0),
                "n_sequential_trades":  int(f.get("n_sequential", 0)),
            }
        fwd_dsr = dsr[dsr["split"] == "forward"]
        if not fwd_dsr.empty:
            d = fwd_dsr.iloc[0]
            summary["healthcheck"]["dsr"] = float(d.get("dsr", 0) or 0)
            summary["healthcheck"]["psr"] = float(d.get("psr", 0) or 0)

        fwd_b = boot[boot["split"] == "forward"]
        if not fwd_b.empty:
            b = fwd_b.iloc[0]
            summary["healthcheck"]["bootstrap_lower_95"] = float(b.get("tr_lower", 0))

    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()

    # 3.3 — Drift detection
    try:
        full = pd.read_csv(V22_DIR / "v22_full_data.csv", parse_dates=[0])
        full = full.set_index(full.columns[0])
        recent = full[full.index >= (full.index.max() - pd.Timedelta(days=60))]
        train = full[full["split"] == "train"]

        feature_cols = [c for c in full.columns if full[c].dtype.kind in "fi" and c != "split"][:50]

        from silver_assistant_v23_honest import feature_drift_report
        drift = feature_drift_report(train, recent, feature_cols, alert_p=0.01)
        drift.to_csv(TRAINING_DIR / "feature_drift_train_vs_recent.csv", index=False)
        drifted = drift[drift["drift"] == True]
        summary["drift"] = {
            "features_checked":  int(len(drift)),
            "features_drifted":  int(len(drifted)),
            "drift_rate":        round(len(drifted) / max(len(drift), 1), 4),
            "top_drifted":       drifted.head(10)["feature"].tolist(),
        }
    except Exception as e:
        summary["drift_error"] = str(e)

    # 3.4 — Сохраняем JSON + Markdown
    (TRAINING_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md = _format_training_md(summary)
    (TRAINING_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(f"  ✅ Training report готов: {TRAINING_DIR}")


def _format_training_md(s: dict) -> str:
    hc = s.get("healthcheck", {})
    lines = [
        f"# Training report — {s['run_date']}",
        "",
        f"**Run time (UTC)**: {s['run_time_utc']}",
        f"**Model**: {s['model_version']}",
        "",
        "## Health check",
        "",
        f"| Метрика | Значение | Норма |",
        f"|---|---|---|",
        f"| Forward total return | {hc.get('forward_total_return', 'n/a'):.2%} | > 0 |"
        if isinstance(hc.get('forward_total_return'), (int, float)) else "| Forward total return | n/a | > 0 |",
        f"| Forward Sharpe (ann.) | {hc.get('forward_sharpe', 0):.2f} | > 1.0 |",
        f"| Forward Max DD | {hc.get('forward_max_dd', 0):.2%} | > -25% |",
        f"| Forward DSR | {hc.get('dsr', 0):.3f} | > 0.7 |",
        f"| Forward PSR | {hc.get('psr', 0):.3f} | > 0.95 |",
        f"| Bootstrap 95% lower | {hc.get('bootstrap_lower_95', 0):.2%} | > 0 |",
        f"| Beats BnH | {'✅' if hc.get('forward_beats_bnh') else '❌'} | True |",
        f"| N sequential trades | {hc.get('n_sequential_trades', 0)} | > 30 |",
        "",
        "## Policy",
        "```json",
        json.dumps(s.get("policy", {}), indent=2),
        "```",
        "",
        "## PnL Summary (compound equity, realistic costs)",
        "",
    ]
    pnl_rows = s.get("pnl", [])
    if pnl_rows:
        lines.append("| Split | v22 honest | v25 CPCV | Δ | BnH | vs BnH | CAGR | MaxDD | Sharpe |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in pnl_rows:
            lines.append(
                f"| {r['split']} | "
                f"{(r.get('v22_honest_total') or 0):.2%} | "
                f"{(r.get('v25_honest_total') or 0):.2%} | "
                f"{(r.get('improvement_pp') or 0):.2%} | "
                f"{(r.get('true_bnh') or 0):.2%} | "
                f"{(r.get('vs_bnh') or 0):.2%} | "
                f"{(r.get('cagr') or 0):.2%} | "
                f"{(r.get('max_dd') or 0):.2%} | "
                f"{r.get('sharpe_ann') or 'n/a'} |"
            )

    lines.append("")
    lines.append("## Statistical robustness")
    lines.append("")
    for r in s.get("dsr_psr", []):
        lines.append(f"- **{r['split']}**: Sharpe={r.get('sharpe_per_trade')}, "
                     f"PSR={r.get('psr')}, DSR={r.get('dsr')} (n={r.get('n_obs')})")

    lines.append("")
    drift = s.get("drift", {})
    if drift:
        lines.append("## Feature drift (train vs последние 60 дней)")
        lines.append("")
        lines.append(f"- Проверено фичей: **{drift.get('features_checked')}**")
        lines.append(f"- С drift (p<0.01): **{drift.get('features_drifted')}** "
                     f"({drift.get('drift_rate', 0):.0%})")
        if drift.get("top_drifted"):
            lines.append(f"- Топ дрейфующих: {', '.join(drift['top_drifted'])}")

    return "\n".join(lines)


# ===========================================================================
# 4. TRADING ACTION REPORT (что делать сегодня с серебром)
# ===========================================================================

def build_trading_report(do_paper_trade: bool = True) -> None:
    print(f"\n=== Шаг 4: trading action report → {TRADING_DIR} ===")
    TRADING_DIR.mkdir(parents=True, exist_ok=True)

    action = {
        "date":            TODAY,
        "time_utc":        datetime.now(timezone.utc).isoformat(),
        "ticker":          os.getenv("TINKOFF_SILVER_TICKER", "SLVRUBF"),
        "signal":          "HOLD",
        "p_up":            None,
        "p_short":         None,
        "rationale":       "",
        "current_price":   None,
        "recommendation":  "HOLD — нет сигнала на сегодня",
    }

    # 4.1 — Сначала пытаемся загрузить production-сигнал (свежий, на сегодня)
    prod_sig_path = REPO_ROOT / "baseline_outputs_prod" / "production_signal_today.json"
    try:
        if prod_sig_path.exists():
            prod = json.loads(prod_sig_path.read_text(encoding="utf-8"))
            if prod.get("ok"):
                action["source"] = "production_inference"
                action["actual_signal_date"] = prod["date"]
                action["signal"] = prod["signal"]
                action["p_up"] = prod["p_up"]
                action["regime"] = prod["regime"]
                action["current_price"] = prod["silver_close"]
                action["p_up_trend_5d"] = prod.get("p_up_trend_5d")
                action["p_up_trend_10d"] = prod.get("p_up_trend_10d")
                action["above_threshold"] = prod.get("above_threshold")
                action["cooldown_remaining"] = prod.get("cooldown_remaining")
                action["rationale"] = (
                    f"p_up={prod['p_up']:.3f}, "
                    f"trend5d={prod.get('p_up_trend_5d', 'n/a')}, "
                    f"режим={prod['regime']}, "
                    f"cooldown ещё {prod.get('cooldown_remaining', 0)}d"
                )
                if prod["signal"] == "BUY":
                    action["recommendation"] = (
                        f"🟢 КУПИТЬ {action['ticker']} — production-модель видит UP "
                        f"(p_up={prod['p_up']:.0%}). Trailing stop 7%, max hold 45d."
                    )
                elif prod["signal"] == "SHORT":
                    action["recommendation"] = (
                        f"🔴 ПРОДАТЬ {action['ticker']} в шорт (p={prod['p_up']:.0%})."
                    )
                elif prod.get("above_threshold") and prod.get("cooldown_remaining", 0) > 0:
                    action["recommendation"] = (
                        f"⚪ ДЕРЖАТЬ — модель уверена ({prod['p_up']:.0%} > 55%), "
                        f"но cooldown ещё {prod['cooldown_remaining']}d. "
                        f"После cooldown ожидается BUY-сигнал."
                    )
                else:
                    action["recommendation"] = (
                        f"⚪ ДЕРЖАТЬ — p_up={prod['p_up']:.0%} < threshold 55%."
                    )
    except Exception as e:
        action["prod_inference_error"] = str(e)

    # 4.2 — Fallback на v25 CPCV decisions, если production не доступен
    if not action.get("source"):
        try:
            dec = pd.read_csv(V25_DIR / "v25_decisions.csv", parse_dates=[0])
            dec = dec.set_index(dec.columns[0])
            dec.index = pd.to_datetime(dec.index)
            valid = dec[dec["p_up"].notna()]
            if not valid.empty:
                last = valid.iloc[-1]
                action["source"] = "cpcv_fallback"
                action["actual_signal_date"] = last.name.strftime("%Y-%m-%d")
                action["signal"] = str(last.get("signal_long", "HOLD"))
                action["p_up"] = float(last.get("p_up", 0) or 0)
                action["p_short"] = float(last.get("p_short", 0) or 0)
                action["regime"] = str(last.get("regime", ""))
                action["current_price"] = float(last.get("silver_close", 0) or 0)
                action["rationale"] = (
                    f"[CPCV fallback] p_up={action['p_up']:.3f}, режим={action['regime']}"
                )
                action["recommendation"] = (
                    f"{'🟢' if action['signal']=='BUY' else '⚪'} "
                    f"{action['signal']} {action['ticker']} (CPCV signal, может быть устаревшим)"
                )
        except Exception as e:
            action["error"] = f"both sources failed: {e}"

    # 4.2 — Paper trading через Tinkoff (опционально)
    paper = {"executed": False, "skipped_reason": None}
    if not do_paper_trade:
        paper["skipped_reason"] = "--no-paper-trade"
    elif not os.getenv("TINKOFF_TOKEN"):
        paper["skipped_reason"] = "TINKOFF_TOKEN не задан в env"
    elif action["signal"] not in ("BUY", "SHORT"):
        paper["skipped_reason"] = f"signal={action['signal']} — paper trade не нужен"
    else:
        # Compounding: пересчитываем размер позиции от ТЕКУЩЕГО баланса Tinkoff
        user_cfg_path = REPO_ROOT / "baseline_outputs_prod" / "user_trading_config.json"
        live_args = [sys.executable, "silver_paper_tinkoff.py", "--live",
                     "--ticker", action["ticker"]]
        if user_cfg_path.exists():
            try:
                user_cfg = json.loads(user_cfg_path.read_text(encoding="utf-8"))

                # Получаем текущий баланс из Tinkoff sandbox
                try:
                    sys.path.insert(0, str(REPO_ROOT))
                    from silver_paper_tinkoff import TinkoffClient, _load_account_id
                    client = TinkoffClient(os.getenv("TINKOFF_TOKEN", ""))
                    account_id = _load_account_id(client)
                    portfolio = client.sandbox_portfolio(account_id)
                    total = portfolio.get("totalAmountPortfolio", {})
                    current_balance = int(total.get("units", 0)) + int(total.get("nano", 0)) / 1e9
                except Exception as e:
                    current_balance = float(user_cfg.get("savings_rub", 1_000_000))
                    paper["balance_fetch_error"] = str(e)

                allocation_pct = float(user_cfg.get("allocation_pct", 20))
                risk_pct = float(user_cfg.get("risk_pct_chosen", 1.5))

                # Compounding: размер позиции в % от ТЕКУЩЕГО баланса
                LOT_NOTIONAL = 20_000  # SLVRUBF
                STOP_PCT = 0.08

                allocation_rub = current_balance * allocation_pct / 100
                max_loss = current_balance * risk_pct / 100
                position_by_risk = max_loss / STOP_PCT
                position_actual = min(position_by_risk, allocation_rub)
                lots = max(1, int(position_actual / LOT_NOTIONAL))

                live_args += ["--futures-max-lots", str(lots)]
                paper["user_config_applied"] = {
                    "current_balance_rub": round(current_balance, 2),
                    "allocation_pct":      allocation_pct,
                    "allocation_rub":      round(allocation_rub, 2),
                    "risk_pct":            risk_pct,
                    "lots_dynamic":        lots,
                    "lots_static":         int(user_cfg.get("lots_target", 0)),
                    "compounding":         True,
                }
            except Exception as e:
                paper["user_config_error"] = str(e)

        try:
            rc, out = _run(live_args, check=False, timeout=120)
            paper["executed"] = (rc == 0)
            paper["raw_output"] = out[-2000:]
        except Exception as e:
            paper["error"] = str(e)

    action["paper_trade"] = paper

    # 4.3 — Текущий портфель Tinkoff
    try:
        rc, out = _run(
            [sys.executable, "silver_paper_tinkoff.py", "--status"],
            check=False, timeout=30,
        )
        action["portfolio_status"] = out[-2000:] if rc == 0 else f"ERROR rc={rc}: {out[-500:]}"
    except Exception as e:
        action["portfolio_status"] = f"ERROR: {e}"

    # 4.4 — Сохраняем
    (TRADING_DIR / "action.json").write_text(
        json.dumps(action, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md = _format_trading_md(action)
    (TRADING_DIR / "action.md").write_text(md, encoding="utf-8")

    # 4.5 — Aлерт-файл (для уведомлений) — только если есть сигнал
    if action["signal"] in ("BUY", "SHORT"):
        alert = {
            "date":         TODAY,
            "ticker":       action["ticker"],
            "signal":       action["signal"],
            "p":            action["p_up"] if action["signal"] == "BUY" else action["p_short"],
            "price":        action["current_price"],
            "rationale":    action["rationale"],
            "paper_traded": action["paper_trade"]["executed"],
        }
        (TRADING_DIR / "ALERT.json").write_text(
            json.dumps(alert, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"  ✅ Trading report готов: {TRADING_DIR}")


def _format_trading_md(a: dict) -> str:
    emoji = {"BUY": "🟢", "SHORT": "🔴", "HOLD": "⚪"}.get(a["signal"], "❔")
    lines = [
        f"# {emoji} {a['date']} — {a['recommendation']}",
        "",
        f"**Тикер**: `{a['ticker']}`",
        f"**Сигнал**: `{a['signal']}`",
        f"**p_up**: {a.get('p_up', 'n/a')}",
        f"**p_short**: {a.get('p_short', 'n/a')}",
        f"**Текущая цена**: {a.get('current_price', 'n/a')}",
        f"**Режим**: {a.get('regime', 'n/a')}",
        f"**Дата сигнала**: {a.get('actual_signal_date', 'n/a')}",
        f"**Обоснование**: {a.get('rationale', '')}",
        "",
        "## Paper trading status",
        "",
    ]
    pt = a.get("paper_trade", {})
    if pt.get("executed"):
        lines.append("✅ **Ордер исполнен в Tinkoff sandbox**")
    elif pt.get("skipped_reason"):
        lines.append(f"⏭ **Пропущено**: {pt['skipped_reason']}")
    elif pt.get("error"):
        lines.append(f"❌ **Ошибка**: {pt['error']}")
    lines.append("")
    lines.append("## Портфель Tinkoff")
    lines.append("```")
    lines.append(a.get("portfolio_status", "n/a"))
    lines.append("```")
    return "\n".join(lines)


# ===========================================================================
# 5. INDEX HTML (для удобного просмотра в браузере на GitHub Pages)
# ===========================================================================

def update_index() -> None:
    """Создаёт daily_reports/INDEX.md со ссылками на все отчёты."""
    print(f"\n=== Шаг 5: обновляю daily_reports/INDEX.md ===")
    train_dates = sorted([d.name for d in (REPORTS_DIR / "training").iterdir()
                          if d.is_dir()], reverse=True) if (REPORTS_DIR / "training").exists() else []
    trade_dates = sorted([d.name for d in (REPORTS_DIR / "trading").iterdir()
                          if d.is_dir()], reverse=True) if (REPORTS_DIR / "trading").exists() else []

    lines = [
        "# Silver Trading Assistant — Daily Reports Index",
        "",
        f"**Last updated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Latest",
        "",
    ]
    if train_dates:
        lines.append(f"- 📊 [Сегодняшний training report](training/{train_dates[0]}/summary.md)")
    if trade_dates:
        lines.append(f"- 💰 [Сегодняшнее торговое действие](trading/{trade_dates[0]}/action.md)")
    lines.append("")
    lines.append("## История training reports")
    lines.append("")
    for d in train_dates[:30]:
        lines.append(f"- [{d}](training/{d}/summary.md)")
    if len(train_dates) > 30:
        lines.append(f"\n*...и ещё {len(train_dates) - 30}*")
    lines.append("")
    lines.append("## История trading actions")
    lines.append("")
    for d in trade_dates[:30]:
        lines.append(f"- [{d}](trading/{d}/action.md)")
    if len(trade_dates) > 30:
        lines.append(f"\n*...и ещё {len(trade_dates) - 30}*")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ INDEX обновлён")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-training",  action="store_true",
                    help="Пропустить retrain (использовать сохранённые v25 результаты)")
    ap.add_argument("--no-paper-trade", action="store_true",
                    help="Не отправлять ордер в Tinkoff")
    ap.add_argument("--no-data-refresh", action="store_true",
                    help="Пропустить refresh v22_full_data")
    args = ap.parse_args()

    print("=" * 70)
    print(f" Silver Trading Assistant — Daily Run {TODAY}")
    print("=" * 70)
    print(f"  Working dir: {REPO_ROOT}")
    print(f"  TINKOFF_TOKEN set: {'YES' if os.getenv('TINKOFF_TOKEN') else 'NO'}")

    v25_output = ""

    if not args.no_data_refresh:
        try:
            refresh_data_if_stale(max_age_hours=20)
        except Exception as e:
            print(f"  ⚠ data refresh failed: {e}")

    if not args.skip_training:
        try:
            _rc, v25_output = retrain_v25()
        except Exception as e:
            print(f"  ⚠ v25 retrain failed: {e}")
            v25_output = traceback.format_exc()

    try:
        build_training_report(v25_output)
    except Exception:
        print(traceback.format_exc())

    try:
        build_trading_report(do_paper_trade=(not args.no_paper_trade))
    except Exception:
        print(traceback.format_exc())

    try:
        update_index()
    except Exception:
        print(traceback.format_exc())

    print("\n" + "=" * 70)
    print(f" ✅ Done. Reports: {REPORTS_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
