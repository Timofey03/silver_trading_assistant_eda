"use client";

/**
 * Глушит ошибки браузерных расширений (MetaMask, Phantom и прочих
 * Web3-кошельков), которые инжектят в каждую страницу свой inpage.js
 * и при отсутствии своего background-worker'а выбрасывают
 * «Failed to connect to MetaMask» / «extension not found».
 *
 * Argentum не использует Web3, поэтому эти ошибки — чистый noise.
 * Next.js dev-overlay подхватывает любую unhandled error, включая
 * чужие inpage.js, и пугает оверлеем «Runtime Error».
 *
 * Фильтр работает ТОЛЬКО для источников, начинающихся с
 * chrome-extension:// / moz-extension:// / safari-extension://, чтобы
 * не глушить настоящие ошибки приложения.
 */
import { useEffect } from "react";

const EXTENSION_PROTOCOLS = [
  "chrome-extension://",
  "moz-extension://",
  "safari-extension://",
  "safari-web-extension://",
];

function isExtensionSource(source?: string | null): boolean {
  if (!source) return false;
  return EXTENSION_PROTOCOLS.some((p) => source.startsWith(p));
}

function isExtensionError(err: unknown): boolean {
  if (!err) return false;
  if (typeof err === "object" && err !== null) {
    const e = err as { stack?: string; message?: string; fileName?: string };
    if (isExtensionSource(e.fileName)) return true;
    if (e.stack && EXTENSION_PROTOCOLS.some((p) => e.stack!.includes(p))) {
      return true;
    }
    if (typeof e.message === "string" && /MetaMask|extension not found/i.test(e.message)) {
      return true;
    }
  }
  return false;
}

export default function ExtensionErrorFilter() {
  useEffect(() => {
    const onError = (e: ErrorEvent) => {
      if (isExtensionSource(e.filename) || isExtensionError(e.error)) {
        e.stopImmediatePropagation();
        e.preventDefault();
      }
    };
    const onRejection = (e: PromiseRejectionEvent) => {
      if (isExtensionError(e.reason)) {
        e.stopImmediatePropagation();
        e.preventDefault();
      }
    };
    window.addEventListener("error", onError, true);
    window.addEventListener("unhandledrejection", onRejection, true);
    return () => {
      window.removeEventListener("error", onError, true);
      window.removeEventListener("unhandledrejection", onRejection, true);
    };
  }, []);
  return null;
}
