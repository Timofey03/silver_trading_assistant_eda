"use client";

/**
 * PositionSparkline — мини-график цены от даты входа до сейчас.
 */
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Point { date: string; close: number; }

export default function PositionSparkline({
  openedAt, entryPrice, currentPrice,
}: {
  openedAt: string;
  entryPrice: number;
  currentPrice: number;
}) {
  const [points, setPoints] = useState<Point[]>([]);

  useEffect(() => {
    const period = daysBack(openedAt) > 30 ? "3m" : "1m";
    api.candles(period as any)
      .then((d) => {
        const opened = new Date(openedAt.split("_")[0]);
        const recent = d.candles
          .filter((c) => new Date(c.time) >= opened)
          .map((c) => ({ date: c.time, close: c.close }));
        setPoints(recent);
      })
      .catch(() => {});
  }, [openedAt]);

  if (points.length < 2) return null;

  const values = points.map((p) => p.close);
  const min = Math.min(...values, entryPrice / 100, currentPrice / 100);
  const max = Math.max(...values, entryPrice / 100, currentPrice / 100);
  const range = max - min || 1;
  const W = 120, H = 32;
  const xStep = W / (points.length - 1);

  const path = points
    .map((p, i) => {
      const x = i * xStep;
      const y = H - ((p.close - min) / range) * (H - 4) - 2;
      return (i === 0 ? "M" : "L") + ` ${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const lastY = H - ((points[points.length - 1].close - min) / range) * (H - 4) - 2;
  const isUp = points[points.length - 1].close >= points[0].close;
  const color = isUp ? "#10b981" : "#f43f5e";

  return (
    <svg width={W} height={H} className="overflow-visible">
      <path d={path} fill="none" stroke={color} strokeWidth={1.2} strokeLinejoin="round" />
      <circle cx={(points.length - 1) * xStep} cy={lastY} r={2} fill={color} />
    </svg>
  );
}

function daysBack(iso: string): number {
  try {
    const d = new Date(iso.split("_")[0]);
    return Math.floor((Date.now() - d.getTime()) / 86400000);
  } catch { return 0; }
}
