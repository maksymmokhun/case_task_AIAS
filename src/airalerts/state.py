"""Погодинна сітка (область, t): стан «активна тривога», ціль, придатність за двома схемами розмітки.

Схеми (docs/definition_of_done.md):
- conservative (основна): тривога без спостережуваного відбою (naive=True або кінець ще не настав) активна.
  naive=False означає, що відбій було опубліковано в момент finished_at, тож майбутнього тут не використано.
- dataset: розмітка датасету, активна, якщо start <= t < finished_at.
"""
import numpy as np
import pandas as pd

from .config import HORIZON_H


def to_ns(ts: pd.Series) -> np.ndarray:
    """tz-aware серія в UTC -> numpy datetime64[ns] без часової зони."""
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")


def as_known_at(events: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Те, що було відомо в момент t: лише початки <= t; кінець відомий лише якщо відбій спостерігався до t.

    Для naive=True відбій не спостерігався ніколи, тож кінець невідомий (NaT) завжди.
    """
    k = events[events.started_at <= t].copy()
    unknown = k.naive | (k.finished_at > t)
    k["finished_at"] = k["finished_at"].mask(unknown, pd.NaT)
    return k


def build_grid(events: pd.DataFrame, t_last: pd.Timestamp, horizon: int = HORIZON_H) -> pd.DataFrame:
    """Всі години t від першого початку до t_last для кожної області.

    Колонки: oblast, t, z (початок у (t, t+horizon]), active_cons, active_ds, elig_cons, elig_ds.
    z обчислюється з майбутніх початків і є лише ціллю; baselines не мають права його читати для t' > t-horizon.
    Сітка суцільна по годинах: на цьому спираються зсуви й кумулятивні суми.
    """
    hours = pd.date_range(events.started_at.min().ceil("h"), t_last, freq="h", tz="UTC")
    tt = hours.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")
    step = np.timedelta64(horizon, "h")
    parts = []
    for oblast, g in events.groupby("oblast", sort=True):
        s, e = to_ns(g.started_at), g.finished_at.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
        unobserved = (g.naive | g.finished_at.isna()).to_numpy()
        i = np.searchsorted(s, tt, "right") - 1
        has, ii = i >= 0, np.clip(i, 0, None)
        end_after = e[ii] > tt                                  # NaT > tt дає False
        active_ds = has & end_after
        active_cons = has & (unobserved[ii] | end_after)
        z = (np.searchsorted(s, tt + step, "right") - np.searchsorted(s, tt, "right")) > 0
        parts.append(pd.DataFrame({"oblast": oblast, "t": hours, "z": z,
                                   "active_cons": active_cons, "active_ds": active_ds}))
    grid = pd.concat(parts, ignore_index=True)
    grid["oblast"] = grid["oblast"].astype("category")
    grid["elig_cons"], grid["elig_ds"] = ~grid.active_cons, ~grid.active_ds
    return grid
