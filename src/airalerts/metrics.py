"""Метрики ймовірнісного прогнозу бінарної цілі."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def brier_skill(y, p, p_ref):
    """BSS = 1 - Brier(p) / Brier(p_ref); додатний, якщо p краща за еталон."""
    return 1.0 - brier(y, p) / brier(y, p_ref)


def summary(y, p) -> dict:
    y = np.asarray(y).astype(int)
    out = {"n": len(y), "base_rate": float(y.mean()), "brier": brier(y, p), "log_loss": float(log_loss(y, p, labels=[0, 1]))}
    if 0 < y.sum() < len(y):
        out |= {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}
    return out


def reliability(y, p, bins=10):
    """Таблиця калібрування: середній прогноз і фактична частка за квантильними кошиками."""
    y, p = np.asarray(y).astype(float), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows = [(p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum())) for b in range(bins) if (idx == b).any()]
    return np.array(rows)


def auc_within_groups(y, p, groups, min_class=30) -> float:
    """Середній ROC-AUC всередині груп (областей); групи, де менше min_class зразків одного з класів, пропускаються.

    Загальний ROC-AUC здебільшого відображає різницю між областями, а це вміє будь-яка база за частотою області.
    """
    y, p, groups = np.asarray(y).astype(int), np.asarray(p), np.asarray(groups)
    aucs = []
    for g in np.unique(groups):
        m = groups == g
        if min(y[m].sum(), (1 - y[m]).sum()) >= min_class:
            aucs.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(aucs))


def block_bootstrap_bss(y, p, p_ref, t, block_days=3, n_boot=1000, seed=0):
    """BSS = 1 - Brier(p) / Brier(p_ref) з блоковим бутстрепом за часом: зразки розбиті на блоки по `block_days` діб
    (усі області разом), блоки ресемплюються з поверненням. Повертає (точкова оцінка, нижня, верхня межа 95% ІН)."""
    y, p, p_ref = (np.asarray(a, float) for a in (y, p, p_ref))
    t = pd.Series(pd.to_datetime(t))
    b = ((t - t.min()) // pd.Timedelta(days=block_days)).to_numpy()
    sum_m, sum_r = np.bincount(b, (p - y) ** 2), np.bincount(b, (p_ref - y) ** 2)
    draws = np.random.default_rng(seed).integers(0, len(sum_m), size=(n_boot, len(sum_m)))
    boot = 1.0 - sum_m[draws].sum(axis=1) / sum_r[draws].sum(axis=1)
    return 1.0 - sum_m.sum() / sum_r.sum(), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def newey_west_se(d, lags) -> float:
    """Стандартна похибка середнього ряду d з вагами Бартлетта до `lags`."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    v = np.sum(d * d) / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * np.sum(d[k:] * d[:n - k]) / n
    return float(np.sqrt(v / n))
