/**
 * Skeleton — placeholder loader с shimmer-эффектом.
 */
export function Skeleton({
  width = "100%", height = "1rem", className = "", style,
}: {
  width?: string | number;
  height?: string | number;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={`rounded ${className}`}
      style={{
        width,
        height,
        background:
          "linear-gradient(90deg, var(--bg-subtle) 0%, var(--bg-elevated) 50%, var(--bg-subtle) 100%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.5s ease-in-out infinite",
        ...style,
      }}
    />
  );
}

export function SkeletonCard({ height = 80 }: { height?: number }) {
  return (
    <div
      className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5 space-y-3"
      style={{ height }}
    >
      <Skeleton width="40%" height="0.75rem" />
      <Skeleton width="60%" height="1.5rem" />
      <Skeleton width="80%" height="0.75rem" />
    </div>
  );
}
