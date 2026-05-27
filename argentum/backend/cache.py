"""
argentum/backend/cache.py — простой in-memory TTL cache.

Использование:
    @ttl_cache(ttl_seconds=60)
    def get_silver_price():
        # тяжёлый HTTP запрос
        return ...
"""
from __future__ import annotations

import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

CACHE_FILE = Path(__file__).resolve().parent / "data" / "cache.pkl"
_cache: dict[str, tuple[float, Any]] = {}


def cache_save_to_disk() -> None:
    """Сохранить кеш в файл при shutdown."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Filter out non-picklable
        clean = {}
        for k, (t, v) in _cache.items():
            try:
                pickle.dumps(v)
                clean[k] = (t, v)
            except Exception:
                pass
        CACHE_FILE.write_bytes(pickle.dumps(clean))
    except Exception:
        pass


def cache_load_from_disk() -> int:
    """Восстановить кеш из файла при startup. Возвращает кол-во записей."""
    if not CACHE_FILE.exists():
        return 0
    try:
        data = pickle.loads(CACHE_FILE.read_bytes())
        # Drop entries older than 1 day
        now = time.time()
        _cache.update({k: v for k, v in data.items() if now - v[0] < 86400})
        return len(_cache)
    except Exception:
        return 0


def ttl_cache(ttl_seconds: int = 60, key_args: bool = False):
    """
    Кеширует результат функции на ttl_seconds.

    Args:
        ttl_seconds: время жизни кеша
        key_args:    включить аргументы в ключ (для функций с параметрами)
    """
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if key_args:
                key = f"{fn.__name__}:{args}:{sorted(kwargs.items())}"
            else:
                key = fn.__name__
            now = time.time()
            cached = _cache.get(key)
            if cached and now - cached[0] < ttl_seconds:
                return cached[1]
            result = fn(*args, **kwargs)
            _cache[key] = (now, result)
            return result
        wrapped._cache_key = fn.__name__
        return wrapped
    return decorator


def cache_clear():
    """Сбросить весь кеш (для тестов)."""
    _cache.clear()
