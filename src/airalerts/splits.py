"""Часові розбиття: валідаційні вікна rolling-origin і маски навчання з purge."""
import numpy as np
import pandas as pd

from .config import HORIZON_H, VAL_LAST_T, VAL_WEEKS, VAL_WINDOWS


def val_windows():
    """Послідовні вікна [start, end) по VAL_WEEKS тижнів безпосередньо перед тестом."""
    end = VAL_LAST_T + pd.Timedelta(hours=1)
    step = pd.Timedelta(weeks=VAL_WEEKS)
    return [(end - (VAL_WINDOWS - k) * step, end - (VAL_WINDOWS - k - 1) * step) for k in range(VAL_WINDOWS)]


def train_mask(grid: pd.DataFrame, w0: pd.Timestamp, days: int | None = None) -> np.ndarray:
    """Рядки навчання для вікна, що починається в w0. Purge: вікно цілі (t, t+H] має закінчитись до w0,
    тому t <= w0 - H - 1 год. days=None означає всю історію, інакше останні `days` діб перед purge."""
    last = w0 - pd.Timedelta(hours=HORIZON_H + 1)
    m = (grid.t <= last).to_numpy()
    if days is not None:
        m = m & (grid.t > last - pd.Timedelta(days=days)).to_numpy()
    return m


def pre_window():
    """Вікно V0 тієї ж довжини безпосередньо перед V1. Лише для вкладеного вибору гіперпараметрів
    (вибір для V1 спирається на V0); у метриках валідації не бере участі."""
    first = val_windows()[0][0]
    return first - pd.Timedelta(weeks=VAL_WEEKS), first
