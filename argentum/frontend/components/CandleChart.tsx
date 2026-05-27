"use client";

/**
 * CandleChart — TradingView Lightweight Charts v5
 * 10-летний график серебра + BUY/SELL маркеры из trades.csv
 */
import { useEffect, useRef, useState } from "react";
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  type IChartApi,
  type ISeriesApi,
  type Time,
  type SeriesMarker,
} from "lightweight-charts";
import { api, type CandleResponse } from "@/lib/api";

type Period = "1m" | "3m" | "6m" | "1y" | "3y" | "all";

const PERIODS: { id: Period; label: string }[] = [
  { id: "1m", label: "1М" },
  { id: "3m", label: "3М" },
  { id: "6m", label: "6М" },
  { id: "1y", label: "1Г" },
  { id: "3y", label: "3Г" },
  { id: "all", label: "Всё" },
];

interface Props {
  period?: Period;
  onPeriodChange?: (p: Period) => void;
}

export default function CandleChart({ period: extPeriod, onPeriodChange }: Props = {}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const [internalPeriod, setInternalPeriod] = useState<Period>("1y");
  const period = extPeriod ?? internalPeriod;
  const setPeriod = (p: Period) => {
    if (onPeriodChange) onPeriodChange(p);
    else setInternalPeriod(p);
  };
  const [data, setData] = useState<CandleResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // ── Создаём chart 1 раз ──────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { color: "transparent" },
        textColor:  "#a1a1aa",
        fontFamily: "var(--font-mono), monospace",
        fontSize:   11,
      },
      grid: {
        vertLines: { color: "#1f1f23" },
        horzLines: { color: "#1f1f23" },
      },
      timeScale: {
        borderColor: "#27272a",
        timeVisible: false,
      },
      rightPriceScale: {
        borderColor: "#27272a",
      },
      crosshair: {
        mode: 1, // Magnet
        vertLine: { color: "#52525b", width: 1, style: 3 },
        horzLine: { color: "#52525b", width: 1, style: 3 },
      },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor:         "#10b981",
      downColor:       "#f43f5e",
      borderUpColor:   "#10b981",
      borderDownColor: "#f43f5e",
      wickUpColor:     "#10b981",
      wickDownColor:   "#f43f5e",
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const onResize = () => {
      if (chartRef.current && containerRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // ── Fetch при смене периода ──────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .candles(period)
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [period]);

  // ── Update данных в series ───────────────────────────────────────
  useEffect(() => {
    if (!seriesRef.current || !chartRef.current || !data) return;
    const series = seriesRef.current;

    series.setData(
      data.candles.map((c) => ({
        time: c.time as Time,
        open: c.open,
        high: c.high,
        low:  c.low,
        close: c.close,
      })),
    );

    if (data.markers.length) {
      const markers: SeriesMarker<Time>[] = data.markers.map((m) => {
        if (m.type === "OPEN") {
          // Наша активная позиция — большой amber-круг
          return {
            time:     m.time as Time,
            position: "belowBar" as const,
            color:    "#f59e0b",     // amber
            shape:    "circle" as const,
            text:     m.text || "АКТИВНА",
            size:     2,
          };
        }
        return {
          time:     m.time as Time,
          position: m.type === "BUY" ? ("belowBar" as const) : ("aboveBar" as const),
          color:    m.type === "BUY" ? "#10b981" : "#f43f5e",
          shape:    m.type === "BUY" ? ("arrowUp" as const) : ("arrowDown" as const),
          text:     m.text || m.type,
          size:     1,
        };
      });
      createSeriesMarkers(series, markers);
    }

    chartRef.current.timeScale().fitContent();
  }, [data]);

  return (
    <section className="space-y-3">
      <div className="flex items-end justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            Цена серебра + сделки модели
          </div>
          <h2 className="text-xl font-medium tracking-tight mt-1">
            История за {PERIODS.find((p) => p.id === period)?.label.toLowerCase()}
          </h2>
        </div>
        <div className="flex gap-1 text-xs font-[family-name:var(--font-mono)]">
          {PERIODS.map((p) => (
            <button
              key={p.id}
              onClick={() => setPeriod(p.id)}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                period === p.id
                  ? "bg-[var(--bg-subtle)] text-[var(--text-primary)] border border-[var(--border)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]/50 border border-transparent"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-[var(--text-faint)] font-[family-name:var(--font-mono)] z-10">
            загрузка свечей…
          </div>
        )}
        <div ref={containerRef} className="w-full" style={{ height: 420 }} />
      </div>

      {/* Легенда */}
      <div className="flex items-center gap-5 text-xs text-[var(--text-muted)] font-[family-name:var(--font-mono)] flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-emerald-400">▲</span>
          <span>BUY (вход)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-rose-400">▼</span>
          <span>SELL (выход + результат)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-amber-400">●</span>
          <span>АКТИВНА (твоя открытая позиция)</span>
        </div>
        {data && (
          <div className="ml-auto">
            {data.markers.filter((m) => m.type !== "OPEN").length / 2} backtest · {data.markers.filter((m) => m.type === "OPEN").length} live
          </div>
        )}
      </div>
    </section>
  );
}
