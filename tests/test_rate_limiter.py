import pytest
from fastapi import HTTPException

from api import rate_limiter
from api.rate_limiter import RateLimiter
from core.config import Settings


def test_refuses_the_request_over_the_limit_with_retry_after(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(rate_limiter.time, "time", lambda: now[0])
    limiter = RateLimiter(max_requests=3, window=60)
    for _ in range(3):
        assert limiter.check_limit("203.0.113.5") is True
        now[0] += 1
    with pytest.raises(HTTPException) as e:
        limiter.check_limit("203.0.113.5")
    assert e.value.status_code == 429
    assert e.value.detail == {"error": "Too many requests", "retry_after": 57, "limit": 3, "window": 60}


def test_clients_are_counted_separately(monkeypatch):
    limiter = RateLimiter(max_requests=1, window=60)
    assert limiter.check_limit("a") and limiter.check_limit("b")
    with pytest.raises(HTTPException):
        limiter.check_limit("a")


def test_window_slides(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(rate_limiter.time, "time", lambda: now[0])
    limiter = RateLimiter(max_requests=1, window=60)
    limiter.check_limit("a")
    now[0] = 59.9
    with pytest.raises(HTTPException):
        limiter.check_limit("a")
    now[0] = 60.1
    assert limiter.check_limit("a") is True


def test_defaults_are_sixty_per_minute(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_REQUESTS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_WINDOW", raising=False)
    s = Settings(_env_file=None)
    assert (s.rate_limit_requests, s.rate_limit_window) == (60, 60)


def test_limits_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "30")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "10")
    s = Settings(_env_file=None)
    assert (s.rate_limit_requests, s.rate_limit_window) == (30, 10)


def test_the_global_limiter_is_built_from_the_settings():
    from core.config import settings
    assert (rate_limiter.limiter.max_requests, rate_limiter.limiter.window) == (settings.rate_limit_requests, settings.rate_limit_window)


def test_the_global_limiter_follows_the_environment():
    """In a fresh interpreter, since the settings object is built once at import."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, vds1_ip="127.0.0.1", RATE_LIMIT_REQUESTS="7", RATE_LIMIT_WINDOW="11")
    out = subprocess.run([sys.executable, "-c", "from api.rate_limiter import limiter; print(limiter.max_requests, limiter.window)"],
                         cwd=root, env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["7", "11"]
