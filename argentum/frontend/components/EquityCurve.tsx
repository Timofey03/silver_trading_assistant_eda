"use client";

/**
 * EquityCurve — линейный график equity модели vs buy-and-hold серебра.
 */
import { useEffect, useRef, useState } from "react";
import {
  createChart, LineSeries, type IChartApi, type ISeriesApi, type Time,
} from "lightweight-charts";
import { api, type EquityResponse } from "@/lib/api";

interface Props {
  period?: "1m" | "3m" | "6m" | "1y" | "3y" | "all";
}

export default function EquityCurve({ period = "all" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const modelSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bhSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const [data, setData] = useState<EquityResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width:  containerRef.current.clientWidth,
      height: 320,
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
      timeScale: { borderColor: "#27272a", timeVisible: false },
      rightPriceScale: { borderColor: "#27272a" },
      crosshair: {
        mode: 1,
        vertLine: { color: "#52525b", width: 1, style: 3 },
        horzLine: { color: "#52525b", width: 1, style: 3 },
      },
    });

    const modelSeries = chart.addSeries(LineSeries, {
      color: "#10b981",
      lineWidth: 2,
      title: "Модель",
    });
    const bhSeries = chart.addSeries(LineSeries, {
      color: "#71717a",
      lineWidth: 1,
      lineStyle: 2, // dashed
      title: "Buy & Hold",
    });

    chartRef.current = chart;
    modelSeriesRef.current = modelSeries;
    bhSeriesRef.current = bhSeries;

    const onResize = () => {
      if (chart && containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api.equity(period)
      .then((d) => { if (!cancelled) { setData(d); setLoading(false); } })
      .catch(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [period]);

  useEffect(() => {
    if (!modelSeriesRef.current || !bhSeriesRef.current || !chartRef.current || !data) return;

    modelSeriesRef.current.setData(
      data.points.map((p) => ({ time: p.date as Time, value: p.model })),
    );
    bhSeriesRef.current.setData(
      data.points.map((p) => ({ time: p.date as Time, value: p.buy_hold })),
    );
    chartRef.current.timeScale().fitContent();
  }, [data]);

  const isWin = data && data.outperformance_pp > 0;
  const outColor = isWin ? "#10b981" : "#f43f5e";

  return (
    <section className="space-y-3">
      <div className="flex items-end justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
            Equity curve · модель vs buy-and-hold
          </div>
          <h2 className="text-xl font-medium tracking-tight mt-1">
            Накопленный рост капитала
          </h2>
        </div>
        {data && (
          <div className="text-right space-y-0.5">
            <div className="text-[11px] uppercase tracking-widest text-[var(--text-faint)]">
              Превосходство
            </div>
            <div
              className="font-[family-name:var(--font-mono)] text-lg font-medium tabular-nums"
              style={{ color: outColor }}
            >
              {isWin ? "+" : ""}{data.outperformance_pp.toFixed(1)} pp
            </div>
          </div>
        )}
      </div>

      <div className="relative rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-[var(--text-faint)] font-[family-name:var(--font-mono)] z-10">
            загрузка equity…
          </div>
        )}
        <div ref={containerRef} className="w-full" style={{ height: 320 }} />
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-xs font-[family-name:var(--font-mono)]">
        <div className="flex items-center gap-2 text-[var(--text-secondary)]">
          <span className="w-4 h-[2px] bg-emerald-400" />
          <span>Модель E3b ensemble {data && `→ ${(data.model_final).toFixed(2)}×`}</span>
        </div>
        <div className="flex items-center gap-2 text-[var(--text-muted)]">
          <span className="w-4 h-[2px] border-t border-dashed border-[var(--text-muted)]" />
          <span>Buy & Hold silver {data && `→ ${(data.buy_hold_final).toFixed(2)}×`}</span>
        </div>
        {data && (
          <div className="ml-auto text-[var(--text-faint)]">
            {data.period_start} → {data.period_end}
          </div>
        )}
      </div>
    </section>
  );
}
