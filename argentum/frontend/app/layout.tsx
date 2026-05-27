import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import ThemeToggle from "@/components/ThemeToggle";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin", "cyrillic"],
  display: "swap",
});

const jbMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
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
      className={`${inter.variable} ${jbMono.variable} dark`}
      data-scroll-behavior="smooth"
    >
      <body className="bg-noise min-h-screen antialiased">
        <header className="sticky top-0 z-50 border-b border-[var(--border-soft)] bg-[var(--bg-base)]/80 backdrop-blur-xl">
          <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-6">
            <Link
              href="/"
              className="font-[family-name:var(--font-mono)] text-[15px] font-medium tracking-tight text-[var(--text-primary)] hover:opacity-90 transition-opacity"
            >
              argentum<span className="text-emerald-400">.</span>
            </Link>
            <nav className="flex items-center gap-1 text-[12px] md:text-[13px] flex-wrap">
              <NavLink href="/" label="Сейчас" />
              <NavLink href="/positions" label="Позиции" />
              <NavLink href="/history" label="История" />
              <NavLink href="/methodology" label="Методология" />
              <NavLink href="/settings" label="Настройки" />
              <ThemeToggle />
            </nav>
          </div>
        </header>
        <main className="relative z-10 mx-auto w-full max-w-5xl px-6 py-16">{children}</main>
        <footer className="relative z-10 mx-auto w-full max-w-5xl px-6 py-10 text-center text-[11px] text-[var(--text-faint)]">
          <span className="font-[family-name:var(--font-mono)]">
            argentum<span className="text-emerald-400/60">.</span> — diploma
            project 2026 · FastAPI · Next.js · модель E3b
          </span>
        </footer>
      </body>
    </html>
  );
}

function NavLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="rounded-md px-3 py-1.5 text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)] transition-colors"
    >
      {label}
    </Link>
  );
}
