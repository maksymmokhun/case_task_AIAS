"""Базові моделі B0–B3. Кожна передбачає P(початок у (t, t+H]) для придатних зразків.

Правило без витоку: у момент t відомі лише мітки z(t') для t' <= t - H (вікно цілі t' закінчується до t).
Усі «трейлінгові» оцінки нижче беруть тільки такі t'.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .config import HORIZON_H

PRIOR_RATE, K_B2, K_B0 = 0.25, 5.0, 4.0   # згладжування малих вибірок; впливає лише на перші тижні історії
P_MIN, P_MAX = 0.005, 0.995               # обрізання, щоб логарифмічні втрати були скінченні


def _clip(p):
    return np.clip(p, P_MIN, P_MAX)


def b2_trailing_rate(grid: pd.DataFrame, elig_col: str, days: int = 90) -> np.ndarray:
    """B2: частка початків серед придатних зразків області за останні `days` діб, мітки лише з t' <= t - H."""
    w = days * 24
    e = grid[elig_col].to_numpy(float)
    p = e * grid.z.to_numpy(float)
    out = np.empty(len(grid))
    for _, ix in grid.groupby("oblast", observed=True).indices.items():
        ce, cp = np.concatenate([[0.0], np.cumsum(e[ix])]), np.concatenate([[0.0], np.cumsum(p[ix])])
        i = np.arange(len(ix))
        hi, lo = np.clip(i - HORIZON_H + 1, 0, len(ix)), np.clip(i - HORIZON_H + 1 - w, 0, len(ix))
        out[ix] = (cp[hi] - cp[lo] + K_B2 * PRIOR_RATE) / (ce[hi] - ce[lo] + K_B2)
    return _clip(out)


def b0_hour_rate(grid: pd.DataFrame, elig_col: str, days: int) -> np.ndarray:
    """B0: частота за область × година доби (Europe/Kyiv) на вікні `days` діб, згладжена до B2.

    Попередній зразок тієї ж локальної години лежить не ближче ніж на 23 год, отже його мітка вже відома в t.
    """
    hk = grid.t.dt.tz_convert("Europe/Kyiv").dt.hour
    e = grid[elig_col].to_numpy(float)
    d = pd.DataFrame({"oblast": grid.oblast, "hk": hk, "t": grid.t, "n": e, "pos": e * grid.z.to_numpy(float)})
    d["row"] = np.arange(len(d))
    d = d.sort_values(["oblast", "hk", "t"]).set_index("t")
    r = d.groupby(["oblast", "hk"], observed=True)[["n", "pos"]].rolling(f"{days}D", closed="left").sum()
    n, pos = r["n"].to_numpy(), r["pos"].to_numpy()
    n, pos = np.nan_to_num(n), np.nan_to_num(pos)
    res = np.empty(len(d))
    res[d["row"].to_numpy()] = (pos + K_B0 * b2_trailing_rate(grid, elig_col)[d["row"].to_numpy()]) / (n + K_B0)
    return _clip(res)


def _b1_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Чи був початок у тому ж вікні вчора і тиждень тому. z(t-24h) і z(t-168h) відомі в t."""
    g = grid.groupby("oblast", observed=True).z
    x = pd.DataFrame({"z24": g.shift(24).fillna(0).astype(float), "z168": g.shift(168).fillna(0).astype(float)})
    return pd.concat([x, pd.get_dummies(grid.oblast, dtype=float)], axis=1)


def b1_fit_predict(grid: pd.DataFrame, elig_col: str, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """B1: сезонний наївний базис. Логістична регресія на z(t-24h), z(t-168h) та інтерцептах областей.

    Навчається лише на train_mask (мітки яких закінчуються до початку прогнозного вікна).
    """
    x = _b1_features(grid)
    tr = train_mask & grid[elig_col].to_numpy()
    model = LogisticRegression(C=1.0, max_iter=300, fit_intercept=False).fit(x[tr], grid.z[tr])
    out = np.full(len(grid), np.nan)
    out[pred_mask] = model.predict_proba(x[pred_mask])[:, 1]
    return _clip(out)


# ---------- B3: класичні часові ряди на добових лічильниках початків ----------

def daily_hour_counts(events: pd.DataFrame, day_first: pd.Timestamp, n_days: int):
    """Лічильники початків: {oblast: масив [n_days, 24]} за днем і годиною UTC."""
    out = {}
    for oblast, g in events.groupby("oblast"):
        d = (g.started_at.dt.floor("D") - day_first).dt.days.to_numpy()
        h = g.started_at.dt.hour.to_numpy()
        ok = (d >= 0) & (d < n_days)
        m = np.zeros((n_days, 24))
        np.add.at(m, (d[ok], h[ok]), 1.0)
        out[oblast] = m
    return out


def forecast_day(daily: np.ndarray, d: int, method: str, window: int = 180) -> float:
    """Прогноз кількості початків у добу d за даними діб < d. method: 'ets' (згладжування з демпфованим трендом)
    або 'arima' (ARIMA(1,1,1), порядок зафіксований наперед і не підбирається)."""
    y = daily[max(0, d - window):d]
    if len(y) < 30 or y.sum() == 0:
        return float(y[-7:].mean()) if len(y) else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if method == "ets":
                fc = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                          initialization_method="estimated").fit().forecast(1)[0]
            else:
                fc = ARIMA(y, order=(1, 1, 1)).fit().forecast(1)[0]
        except Exception:
            fc = y[-7:].mean()
    return float(max(fc, 0.0))


def b3_probabilities(grid: pd.DataFrame, events: pd.DataFrame, method: str, rows: np.ndarray) -> np.ndarray:
    """B3: P(N >= 1) = 1 - exp(-lambda) для рядків grid[rows].

    lambda = прогноз добового лічильника (дані <= кінець попередньої доби UTC) × частка початків у годинах
    (h, h+1, h+2) за останні 28 діб. Для вікон через північ добовий прогноз той самий (спрощення).
    Не залежить від поточного стану області (як і B0–B2).
    """
    day0 = grid.t.min().floor("D")
    n_days = int((grid.t.max().floor("D") - day0).days) + 1
    counts = daily_hour_counts(events, day0, n_days)
    out = np.full(len(grid), np.nan)
    t = grid.t
    day_idx = ((t.dt.floor("D") - day0).dt.days).to_numpy()
    hour = t.dt.hour.to_numpy()
    obl = grid.oblast.astype(str).to_numpy()
    for oblast in np.unique(obl[rows]):
        m = counts[oblast]
        daily = m.sum(axis=1)
        cum = np.vstack([np.zeros((1, 24)), np.cumsum(m, axis=0)])
        sel = rows[obl[rows] == oblast]
        lam_cache = {}
        for d in np.unique(day_idx[sel]):
            lam_cache[d] = forecast_day(daily, int(d), method)
        for r in sel:
            d, h = int(day_idx[r]), int(hour[r])
            prof = cum[d] - cum[max(0, d - 28)]
            tot = prof.sum()
            share = prof / tot if tot > 0 else np.full(24, 1 / 24)
            lam = lam_cache[d] * (share[h] + share[(h + 1) % 24] + share[(h + 2) % 24])
            out[r] = 1.0 - np.exp(-lam)
    return _clip(out)


def recalibrate(grid: pd.DataFrame, elig_col: str, p_raw: np.ndarray, train_mask: np.ndarray,
                pred_mask: np.ndarray) -> np.ndarray:
    """Логістична рекалібрація прогнозу: sigma(a * logit(p_raw) + b), коефіцієнти лише з train_mask.

    Використовується для B3 (ETS/ARIMA): вони добре ранжують, але зміщені за ймовірністю.
    """
    logit = np.log(p_raw / (1.0 - p_raw))
    tr = train_mask & grid[elig_col].to_numpy() & ~np.isnan(p_raw)
    model = LogisticRegression(C=1e6, max_iter=300).fit(logit[tr][:, None], grid.z[tr])
    out = np.full(len(grid), np.nan)
    pm = pred_mask & ~np.isnan(p_raw)
    out[pm] = model.predict_proba(logit[pm][:, None])[:, 1]
    return _clip(out)
