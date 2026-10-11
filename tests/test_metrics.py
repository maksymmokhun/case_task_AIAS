"""Тести метрик."""
import numpy as np
import pandas as pd

from airalerts.metrics import block_bootstrap_bss, brier_skill


def test_bootstrap_point_estimate_matches_bss():
    rng = np.random.default_rng(0)
    t = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 24 * 60, 4000), unit="h")
    y = (rng.random(4000) < 0.4).astype(float)
    p_ref = np.full(4000, 0.4)
    p = np.clip(0.4 + 0.3 * (y - 0.4) + rng.normal(0, 0.05, 4000), 0.01, 0.99)
    point, lo, hi = block_bootstrap_bss(y, p, p_ref, t, n_boot=300)
    assert abs(point - brier_skill(y, p, p_ref)) < 1e-12
    assert lo < point < hi and lo > 0                 # p побудована з y, тож явно краща за константу


def test_bootstrap_interval_contains_zero_for_noise():
    rng = np.random.default_rng(1)
    t = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(np.arange(6000), unit="h")
    y = (rng.random(6000) < 0.4).astype(float)
    p_ref = np.full(6000, 0.4)
    p = np.clip(0.4 + rng.normal(0, 0.001, 6000), 0.01, 0.99)  # майже константа: покращення немає
    _, lo, hi = block_bootstrap_bss(y, p, p_ref, t, n_boot=300)
    assert lo < 0 < hi
