"use client";

/**
 * Confetti — лёгкая SVG-конфетти на N секунд.
 * Запускается setShow(true) при profit-event.
 */
import { useEffect, useState } from "react";

export default function Confetti({ show }: { show: boolean }) {
  const [pieces, setPieces] = useState<{ id: number; left: number; delay: number; color: string }[]>([]);

  useEffect(() => {
    if (!show) return;
    const colors = ["#10b981", "#34d399", "#f59e0b", "#fafafa"];
    const ps = Array.from({ length: 40 }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      delay: Math.random() * 0.5,
      color: colors[Math.floor(Math.random() * colors.length)],
    }));
    setPieces(ps);
    const t = setTimeout(() => setPieces([]), 3500);
    return () => clearTimeout(t);
  }, [show]);

  if (!pieces.length) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-[60] overflow-hidden">
      {pieces.map((p) => (
        <div
          key={p.id}
          className="absolute top-0 w-2 h-3 rounded-sm"
          style={{
            left: `${p.left}%`,
            backgroundColor: p.color,
            animation: `confettiFall 3s ${p.delay}s linear forwards`,
            transform: "rotate(0deg)",
          }}
        />
      ))}
    </div>
  );
}
