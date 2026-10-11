import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

T0 = pd.Timestamp("2026-01-01", tz="UTC")
DAYS = 150


def make_events(seed=0):
    """Синтетичні тривоги для 3 областей: без перекриттів, з добовою сезонністю, ~6% naive (кінець +30 хв)."""
    rng = np.random.default_rng(seed)
    rows = []
    for oblast, scale in (("A oblast", 3.0), ("B oblast", 6.0), ("C oblast", 12.0)):
        t = T0
        while t < T0 + pd.Timedelta(days=DAYS):
            night = 1.0 if (t.hour >= 20 or t.hour < 5) else 2.5
            t = t + pd.Timedelta(hours=float(rng.exponential(scale * night)))
            naive = rng.random() < 0.06
            dur = pd.Timedelta(minutes=30) if naive else pd.Timedelta(minutes=float(5 + rng.exponential(25)))
            rows.append((oblast, t, t + dur, naive))
            t = t + dur + (pd.Timedelta(hours=3.2) if naive else pd.Timedelta(0))
    df = pd.DataFrame(rows, columns=["oblast", "started_at", "finished_at", "naive"])
    df = df[df.started_at < T0 + pd.Timedelta(days=DAYS)]
    return df.sort_values(["oblast", "started_at"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def events():
    return make_events()


@pytest.fixture(scope="session")
def t_end():
    return T0 + pd.Timedelta(days=DAYS - 1)


@pytest.fixture(scope="session")
def probe_times():
    rng = np.random.default_rng(1)
    base = T0 + pd.Timedelta(days=110)
    return [base + pd.Timedelta(hours=int(h)) for h in rng.integers(0, 24 * 30, size=6)]
