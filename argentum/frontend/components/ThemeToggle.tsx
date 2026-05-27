"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const stored = (localStorage.getItem("argentum-theme") as "dark" | "light" | null) || "dark";
    setTheme(stored);
    applyTheme(stored);
  }, []);

  const applyTheme = (t: "dark" | "light") => {
    const html = document.documentElement;
    html.classList.remove("dark", "light");
    html.classList.add(t);
  };

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
    localStorage.setItem("argentum-theme", next);
  };

  const Icon = theme === "dark" ? Sun : Moon;
  return (
    <button
      onClick={toggle}
      className="rounded-md p-1.5 text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)] transition-colors"
      aria-label="Toggle theme"
      title={`Сейчас: ${theme}. Кликни для смены.`}
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}
