"use client";

/**
 * CountUp — анимирует число с тикающим эффектом (как trading tickers).
 * Перерисовывает плавно при изменении value.
 */
import { useEffect, useRef, useState } from "react";

interface Props {
  value: number;
  decimals?: number;
  duration?: number;       // ms
  prefix?: string;
  suffix?: string;
  className?: string;
  style?: React.CSSProperties;
}

export default function CountUp({
  value,
  decimals = 0,
  duration = 600,
  prefix = "",
  suffix = "",
  className,
  style,
}: Props) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    fromRef.current = display;
    startTimeRef.current = null;
    let rafId: number;

    const step = (t: number) => {
      if (startTimeRef.current === null) startTimeRef.current = t;
      const elapsed = t - startTimeRef.current;
      const progress = Math.min(1, elapsed / duration);
      // easeOutCubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = fromRef.current + (value - fromRef.current) * eased;
      setDisplay(current);
      if (progress < 1) rafId = requestAnimationFrame(step);
    };

    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, duration]);

  const formatted = display.toLocaleString("ru-RU", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <span className={className} style={style}>
      {prefix}{formatted}{suffix}
    </span>
  );
}
