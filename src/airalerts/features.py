"""Ознаки основної моделі (ітерація 4).

Правило без витоку: ознака в момент t залежить лише від того, що було відомо в t.
- Початки: тільки з started_at <= t.
- Відбої: тільки спостережені (naive=False) і лише з finished_at <= t. Синтетичні кінці та прапорець naive
  як ознаки не використовуються: «відбою ще не було» означає, що він не спостерігався до t.
- Стан області «активна» береться за консервативною схемою (active_cons у state.build_grid): це функція
  лише відомого на момент самого t', отже значення для t' <= t відомі й у t.
- Мітки z(t') дозволені лише для t' <= t - H (див. baselines.py). У цьому модулі їх читають z_lag24, z_lag168
  та ознаки-пріори B2/B0.
- Календар рахується в Europe/Kyiv.

Ознаки «чи має остання тривога спостережений відбій» навмисно немає: за схемою датасету придатні зразки після
синтетичного кінця однозначно вказують на naive-тривогу, а цей прапорець визначається майбутньою подією.
Схема датасету сама використовує ретроспективні кінці для вибору рядків (див. docs/definition_of_done.md),
тож для станових моделей вона оптимістична; основна схема консервативна.
"""
import numpy as np
import pandas as pd

from . import baselines as bl
from .config import HORIZON_H
from .state import to_ns

H_NS = np.timedelta64(1, "h")
CAP_H = 24 * 30                    # «давно» = 30 діб; довші проміжки обрізаються, щоб логарифм не розтягував хвіст
COUNT_WINDOWS_H = (1, 3, 6, 24, 168, 672)
NEIGHBOR_WINDOWS_H = (3, 24)       # сусіди: початки за 3 і 24 год
COUNTRY_WINDOWS_H = (1, 3, 24)     # решта країни: початки за 1, 3 і 24 год
ACTIVE_FRAC_WINDOWS_H = (24, 168)

# Сусідство областей складено вручну за географією (не перевірялось за офіційним переліком кордонів).
# Крим і Луганська відсутні: Луганська виключена з даних. Київ межує з Київською областю.
_ADJ = {
    "Volynska oblast": ["Rivnenska oblast", "Lvivska oblast"],
    "Rivnenska oblast": ["Volynska oblast", "Lvivska oblast", "Ternopilska oblast", "Khmelnytska oblast", "Zhytomyrska oblast"],
    "Lvivska oblast": ["Volynska oblast", "Rivnenska oblast", "Ternopilska oblast", "Ivano-Frankivska oblast", "Zakarpatska oblast"],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
    "Ivano-Frankivska oblast": ["Lvivska oblast", "Ternopilska oblast", "Chernivetska oblast", "Zakarpatska oblast"],
    "Chernivetska oblast": ["Ivano-Frankivska oblast", "Ternopilska oblast", "Khmelnytska oblast", "Vinnytska oblast"],
    "Ternopilska oblast": ["Lvivska oblast", "Rivnenska oblast", "Khmelnytska oblast", "Chernivetska oblast", "Ivano-Frankivska oblast"],
    "Khmelnytska oblast": ["Rivnenska oblast", "Ternopilska oblast", "Chernivetska oblast", "Vinnytska oblast", "Zhytomyrska oblast"],
    "Zhytomyrska oblast": ["Rivnenska oblast", "Khmelnytska oblast", "Vinnytska oblast", "Kyivska oblast"],
    "Vinnytska oblast": ["Zhytomyrska oblast", "Khmelnytska oblast", "Chernivetska oblast", "Odeska oblast",
                         "Kirovohradska oblast", "Cherkaska oblast", "Kyivska oblast"],
    "Kyivska oblast": ["Zhytomyrska oblast", "Vinnytska oblast", "Cherkaska oblast", "Poltavska oblast",
                       "Chernihivska oblast", "Kyiv City"],
    "Kyiv City": ["Kyivska oblast"],
    "Chernihivska oblast": ["Kyivska oblast", "Poltavska oblast", "Sumska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Poltavska oblast", "Kharkivska oblast"],
    "Poltavska oblast": ["Kyivska oblast", "Chernihivska oblast", "Sumska oblast", "Kharkivska oblast",
                         "Dnipropetrovska oblast", "Kirovohradska oblast", "Cherkaska oblast"],
    "Kharkivska oblast": ["Sumska oblast", "Poltavska oblast", "Dnipropetrovska oblast", "Donetska oblast"],
    "Cherkaska oblast": ["Kyivska oblast", "Vinnytska oblast", "Kirovohradska oblast", "Poltavska oblast"],
    "Kirovohradska oblast": ["Cherkaska oblast", "Vinnytska oblast", "Odeska oblast", "Mykolaivska oblast",
                             "Dnipropetrovska oblast", "Poltavska oblast"],
    "Dnipropetrovska oblast": ["Kirovohradska oblast", "Poltavska oblast", "Kharkivska oblast", "Donetska oblast",
                               "Zaporizka oblast", "Khersonska oblast", "Mykolaivska oblast"],
    "Donetska oblast": ["Kharkivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"],
    "Zaporizka oblast": ["Dnipropetrovska oblast", "Donetska oblast", "Khersonska oblast"],
    "Khersonska oblast": ["Zaporizka oblast", "Dnipropetrovska oblast", "Mykolaivska oblast"],
    "Mykolaivska oblast": ["Khersonska oblast", "Dnipropetrovska oblast", "Kirovohradska oblast", "Odeska oblast"],
    "Odeska oblast": ["Mykolaivska oblast", "Kirovohradska oblast", "Vinnytska oblast"],
}
NEIGHBORS = _ADJ

# Групи ознак: для оцінки важливості й абляції.
GROUPS = {
    "own_state": ["since_start", "since_clear", "last_dur", "act_frac_24h", "act_frac_168h"],
    "own_counts": [f"n_start_{w}h" for w in COUNT_WINDOWS_H],
    "priors": ["logit_b2_90", "logit_b0_28", "logit_b0_90", "z_lag24", "z_lag168"],
    "neighbors": ["nb_active"] + [f"nb_start_{w}h" for w in NEIGHBOR_WINDOWS_H],
    "country": ["all_active"] + [f"all_start_{w}h" for w in COUNTRY_WINDOWS_H],
    "calendar": ["hour_local", "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2", "dow", "is_weekend"],
    "oblast": ["oblast_code"],
}
ALL_COLS = [c for cols in GROUPS.values() for c in cols]
STATE_COLS = [c for c in ALL_COLS if c not in GROUPS["priors"] and c != "oblast_code"]


def _log_h(x):
    return np.log1p(np.minimum(x, CAP_H))


def _oblast_matrix(values: np.ndarray, idx: dict, order: list) -> np.ndarray:
    return np.vstack([values[idx[o]] for o in order])


def adjacency(order: list) -> np.ndarray:
    """Матриця сусідства [n, n] у порядку `order`, нормована по рядках (середнє по сусідах)."""
    pos = {o: i for i, o in enumerate(order)}
    a = np.zeros((len(order), len(order)))
    for o, nbs in NEIGHBORS.items():
        if o in pos:
            for n in nbs:
                if n in pos:
                    a[pos[o], pos[n]] = 1.0
    return a / a.sum(axis=1, keepdims=True)


def state_features(events: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Ознаки стану, лічильників, сусідів, країни й календаря. Не залежать від схеми розмітки і від міток z."""
    idx = grid.groupby("oblast", observed=True).indices
    order = sorted(idx)
    hours = grid.t.iloc[idx[order[0]]]
    for o in order:
        assert (grid.t.iloc[idx[o]].to_numpy() == hours.to_numpy()).all(), "сітка має бути спільною для областей"
    tt = to_ns(hours)
    n_o, n_h = len(order), len(tt)

    mats = {}                                                      # назва -> [n_o, n_h]
    for j, o in enumerate(order):
        g = events[events.oblast == o]
        s, e = to_ns(g.started_at), to_ns(g.finished_at)
        observed = ~(g.naive | g.finished_at.isna()).to_numpy()
        cs, ce = s[observed], e[observed]
        srt = np.argsort(ce, kind="stable")
        cs, ce = cs[srt], ce[srt]
        n_started = np.searchsorted(s, tt, "right")
        cols = {f"n_start_{w}h": n_started - np.searchsorted(s, tt - w * H_NS, "right")
                for w in sorted(set(COUNT_WINDOWS_H) | set(NEIGHBOR_WINDOWS_H) | set(COUNTRY_WINDOWS_H))}
        last = n_started - 1
        cols["since_start"] = _log_h(np.where(last >= 0, (tt - s[np.clip(last, 0, None)]) / H_NS, CAP_H))
        k = np.searchsorted(ce, tt, "right") - 1                   # останній спостережений відбій <= t
        kk = np.clip(k, 0, None)
        cols["since_clear"] = _log_h(np.where(k >= 0, (tt - ce[kk]) / H_NS, CAP_H))
        cols["last_dur"] = np.log1p(np.where(k >= 0, (ce[kk] - cs[kk]) / H_NS, 0.5))
        for name, v in cols.items():
            mats.setdefault(name, np.zeros((n_o, n_h)))[j] = v

    active = _oblast_matrix(grid.active_cons.to_numpy(float), idx, order)
    csum = np.concatenate([np.zeros((n_o, 1)), np.cumsum(active, axis=1)], axis=1)
    for w in ACTIVE_FRAC_WINDOWS_H:
        lo = np.clip(np.arange(1, n_h + 1) - w, 0, None)
        mats[f"act_frac_{w}h"] = (csum[:, 1:] - csum[:, lo]) / w

    adj = adjacency(order)
    mats["nb_active"] = adj @ active
    for w in NEIGHBOR_WINDOWS_H:
        mats[f"nb_start_{w}h"] = adj @ mats[f"n_start_{w}h"]
    mats["all_active"] = (active.sum(axis=0) - active) / (n_o - 1)
    for w in COUNTRY_WINDOWS_H:
        m = mats[f"n_start_{w}h"]
        mats[f"all_start_{w}h"] = (m.sum(axis=0) - m) / (n_o - 1)

    local = hours.dt.tz_convert("Europe/Kyiv")
    ang = 2 * np.pi * local.dt.hour.to_numpy() / 24
    cal = {"hour_local": local.dt.hour.to_numpy(float), "hour_sin1": np.sin(ang), "hour_cos1": np.cos(ang),
           "hour_sin2": np.sin(2 * ang), "hour_cos2": np.cos(2 * ang), "dow": local.dt.dayofweek.to_numpy(float),
           "is_weekend": (local.dt.dayofweek.to_numpy() >= 5).astype(float)}
    for name, v in cal.items():
        mats[name] = np.tile(v, (n_o, 1))

    out = {}
    for name in STATE_COLS:
        arr = np.empty(len(grid), np.float32)
        for j, o in enumerate(order):
            arr[idx[o]] = mats[name][j]
        out[name] = arr
    out["oblast_code"] = grid.oblast.cat.codes.to_numpy().astype(np.float32)
    return pd.DataFrame(out, index=grid.index)


def _logit(p):
    return np.log(p / (1.0 - p))


def prior_features(grid: pd.DataFrame, elig_col: str) -> pd.DataFrame:
    """Пріори з міток (правило t' <= t - H): трейлінгові частоти B2/B0 і сезонні лаги z(t-24h), z(t-168h).

    Залежать від схеми розмітки через придатні зразки (elig_col).
    """
    z = grid.groupby("oblast", observed=True).z
    return pd.DataFrame({
        "logit_b2_90": _logit(bl.b2_trailing_rate(grid, elig_col, 90)),
        "logit_b0_28": _logit(bl.b0_hour_rate(grid, elig_col, 28)),
        "logit_b0_90": _logit(bl.b0_hour_rate(grid, elig_col, 90)),
        "z_lag24": z.shift(24).fillna(0).astype(float),
        "z_lag168": z.shift(168).fillna(0).astype(float)}, index=grid.index).astype(np.float32)


def build_features(grid: pd.DataFrame, elig_col: str, state: pd.DataFrame) -> pd.DataFrame:
    """Повна матриця ознак у порядку ALL_COLS. `state` = state_features(...) (рахується один раз для обох схем)."""
    assert HORIZON_H <= 24, "z_lag24 відомий у t лише якщо горизонт не більший за 24 год"
    return pd.concat([state, prior_features(grid, elig_col)], axis=1)[ALL_COLS]
