import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Скрыть встроенный Next.js dev indicator (тот N-кружок с Route/Bundler info).
  // Это UI самого Next.js, не наш — не переводится. В production его и так нет.
  devIndicators: false,
};

export default nextConfig;
