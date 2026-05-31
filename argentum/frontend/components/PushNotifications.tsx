"use client";

/**
 * PushNotifications — подписка на браузерные уведомления о смене сигнала.
 * Polls /api/signal каждую минуту; если signal изменился — notification.
 */
import { useEffect, useState } from "react";
import { Bell, BellOff } from "lucide-react";
import { api } from "@/lib/api";

const LAST_SIGNAL_KEY = "argentum-last-signal";

export default function PushNotifications() {
  const [permission, setPermission] = useState<NotificationPermission>("default");
  const [active, setActive] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || !("Notification" in window)) return;
    setPermission(Notification.permission);
    setActive(localStorage.getItem("argentum-push-enabled") === "1");
  }, []);

  useEffect(() => {
    if (!active || permission !== "granted") return;

    const check = async () => {
      try {
        const sig = await api.signal();
        const lastSig = localStorage.getItem(LAST_SIGNAL_KEY);
        if (lastSig && lastSig !== sig.signal) {
          new Notification("Argentum · сигнал изменился", {
            body: `${lastSig} → ${sig.signal} (уверенность ${(sig.p_up * 100).toFixed(0)}%)`,
            icon: "/favicon.ico",
            tag: "argentum-signal",
          });
        }
        localStorage.setItem(LAST_SIGNAL_KEY, sig.signal);
      } catch {}
    };

    check();
    const id = setInterval(check, 60_000);  // каждую минуту
    return () => clearInterval(id);
  }, [active, permission]);

  const handleToggle = async () => {
    if (typeof window === "undefined" || !("Notification" in window)) {
      alert("Браузер не поддерживает уведомления");
      return;
    }
    if (active) {
      setActive(false);
      localStorage.setItem("argentum-push-enabled", "0");
      return;
    }
    let p = Notification.permission;
    if (p === "default") {
      p = await Notification.requestPermission();
      setPermission(p);
    }
    if (p === "granted") {
      setActive(true);
      localStorage.setItem("argentum-push-enabled", "1");
      new Notification("Argentum уведомления включены", {
        body: "Получишь уведомление когда сигнал изменится",
        icon: "/favicon.ico",
      });
    } else {
      alert("Уведомления заблокированы в настройках браузера");
    }
  };

  const Icon = active ? Bell : BellOff;
  return (
    <button
      onClick={handleToggle}
      className={`rounded-md p-1.5 transition-colors ${
        active
          ? "text-emerald-400 hover:bg-emerald-500/10"
          : "text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]"
      }`}
      title={active ? "Уведомления включены — клик чтобы выключить" : "Получать уведомления о смене сигнала"}
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}
