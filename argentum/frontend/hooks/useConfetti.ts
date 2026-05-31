"use client";

/**
 * useConfetti — простая canvas-конфетти без зависимостей.
 * Запускается при закрытии позиции в плюсе.
 */
export function fireConfetti() {
  if (typeof window === "undefined") return;

  const canvas = document.createElement("canvas");
  canvas.style.position = "fixed";
  canvas.style.inset = "0";
  canvas.style.pointerEvents = "none";
  canvas.style.zIndex = "9999";
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    canvas.remove();
    return;
  }

  const colors = ["#10b981", "#34d399", "#f59e0b", "#fbbf24", "#fafafa"];
  const particles: Array<{
    x: number; y: number; vx: number; vy: number;
    color: string; size: number; rotation: number; vr: number;
  }> = [];

  const N = 120;
  for (let i = 0; i < N; i++) {
    particles.push({
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
      vx: (Math.random() - 0.5) * 18,
      vy: -Math.random() * 18 - 4,
      color: colors[Math.floor(Math.random() * colors.length)],
      size: Math.random() * 6 + 3,
      rotation: Math.random() * 360,
      vr: (Math.random() - 0.5) * 10,
    });
  }

  let frame = 0;
  const gravity = 0.4;
  const friction = 0.98;

  const animate = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let alive = false;
    particles.forEach((p) => {
      if (p.y < canvas.height + 50) {
        alive = true;
        p.vy += gravity;
        p.vx *= friction;
        p.x += p.vx;
        p.y += p.vy;
        p.rotation += p.vr;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate((p.rotation * Math.PI) / 180);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
        ctx.restore();
      }
    });
    if (alive && frame++ < 200) {
      requestAnimationFrame(animate);
    } else {
      canvas.remove();
    }
  };
  animate();
}
