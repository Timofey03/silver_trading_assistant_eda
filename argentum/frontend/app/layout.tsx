import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin", "cyrillic"],
});

const jbMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "argentum. — AI-помощник для рынка серебра",
  description:
    "ML-помощник для торговли серебром. Модель E3b: Sharpe 0.47, " +
    "Win Rate 68%, накопленная доходность +106% за 10 лет walk-forward.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="ru"
      className={`${inter.variable} ${jbMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-neutral-950 text-neutral-100 font-[family-name:var(--font-inter)]">
        <header className="sticky top-0 z-50 border-b border-neutral-800/50 bg-neutral-950/80 backdrop-blur-xl">
          <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
            <Link
              href="/"
              className="font-[family-name:var(--font-mono)] text-base font-medium tracking-tight text-neutral-100 hover:text-white"
            >
              argentum<span className="text-emerald-400">.</span>
            </Link>
            <nav className="flex items-center gap-6 text-sm text-neutral-400">
              <Link href="/" className="hover:text-neutral-100 transition-colors">
                Сейчас
              </Link>
              <Link href="/history" className="hover:text-neutral-100 transition-colors">
                История
              </Link>
              <Link href="/settings" className="hover:text-neutral-100 transition-colors">
                Настройки
              </Link>
            </nav>
          </div>
        </header>
        <main className="flex-1 mx-auto w-full max-w-6xl px-6 py-12">{children}</main>
        <footer className="mx-auto w-full max-w-6xl px-6 py-8 text-center text-xs text-neutral-500">
          Дипломный проект 2026 · FastAPI + Next.js · Модель E3b · cross-asset
        </footer>
      </body>
    </html>
  );
}
