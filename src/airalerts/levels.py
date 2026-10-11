"""Етап 2: шар рівня тривоги P(червона | початок тривоги в області).

Дані: архів alerts.in.ua (`data/raw/alerts_in_ua.csv`), записи по районах, громадах і містах, рівні `yellow`/`red`
лише з 2026-09-05. Початок тривоги в області (епізод) визначається за тим самим правилом, що й у T1:
новий запис починає епізод, якщо на його початку в області немає жодного активного запису.
Рівень епізоду: червоний, якщо будь-який запис епізоду з початком у [початок епізоду; початок + W] червоний.

Що відомо в момент `t` (щоб не було витоку):
  - належність запису до епізоду визначається лише записами, що почалися раніше (активність = finished_at > початку нового);
  - рівень епізоду відомий з моменту початок + W, тому в ознаки для епізоду `i` потрапляють лише епізоди `j`
    з `start_j + W <= start_i`.
"""
import numpy as np
import pandas as pd

from .config import RAW

OBLAST_EN = {
    "Вінницька область": "Vinnytska oblast", "Волинська область": "Volynska oblast", "Дніпропетровська область": "Dnipropetrovska oblast",
    "Донецька область": "Donetska oblast", "Житомирська область": "Zhytomyrska oblast", "Закарпатська область": "Zakarpatska oblast",
    "Запорізька область": "Zaporizka oblast", "Івано-Франківська область": "Ivano-Frankivska oblast", "Київська область": "Kyivska oblast",
    "Кіровоградська область": "Kirovohradska oblast", "Львівська область": "Lvivska oblast", "Миколаївська область": "Mykolaivska oblast",
    "Одеська область": "Odeska oblast", "Полтавська область": "Poltavska oblast", "Рівненська область": "Rivnenska oblast",
    "Сумська область": "Sumska oblast", "Тернопільська область": "Ternopilska oblast", "Харківська область": "Kharkivska oblast",
    "Херсонська область": "Khersonska oblast", "Хмельницька область": "Khmelnytska oblast", "Черкаська область": "Cherkaska oblast",
    "Чернівецька область": "Chernivetska oblast", "Чернігівська область": "Chernihivska oblast", "м. Київ": "Kyiv City",
}
LEVELS_FROM = pd.Timestamp("2026-09-05", tz="UTC")     # з цієї дати в архіві є рівні
MISSING_END = pd.Timedelta(hours=3)                    # кінець невідомий: запис вважаємо активним 3 год (як в парсері T1)
HOUR_BUCKET = 6                                        # години доби за Києвом групуються по 6


def load_archive(path=None) -> pd.DataFrame:
    """Повітряні тривоги з рівнями: oblast (англ.), started_at, finished_filled (з підстановкою), level."""
    d = pd.read_csv(path or RAW / "alerts_in_ua.csv", parse_dates=["started_at", "finished_at"])
    d = d[(d.alert_type == "air_raid") & (d.started_at >= LEVELS_FROM)].copy()
    d["oblast"] = d.location_oblast.map(OBLAST_EN)
    d = d.dropna(subset=["oblast"])
    d["finished_filled"] = d.finished_at.fillna(d.started_at + MISSING_END)
    d = d.rename(columns={"alert_level": "level"})
    return d[["oblast", "started_at", "finished_filled", "level"]].sort_values(["oblast", "started_at"]).reset_index(drop=True)


def episodes(archive: pd.DataFrame, window_min: float) -> pd.DataFrame:
    """Початки тривог в областях і рівень: red = 1, якщо червоний запис почався протягом window_min від початку епізоду."""
    out = []
    for ob, g in archive.groupby("oblast", sort=False):
        until, cur = None, None
        for s, f, lvl in zip(g.started_at, g.finished_filled, g.level):
            if until is None or s >= until:
                if cur is not None:
                    out.append(cur)
                cur = {"oblast": ob, "started_at": s, "red": 0}
            if lvl == "red" and (s - cur["started_at"]) <= pd.Timedelta(minutes=window_min):
                cur["red"] = 1
            until = f if until is None else max(until, f)
        if cur is not None:
            out.append(cur)
    ep = pd.DataFrame(out).sort_values("started_at").reset_index(drop=True)
    ep["bucket"] = (ep.started_at.dt.tz_convert("Europe/Kyiv").dt.hour // HOUR_BUCKET).astype(int)
    return ep


def predict(ep: pd.DataFrame, window_min: float, win_days: float | None, m_o: float, m_h: float | None = None) -> np.ndarray:
    """P(red) для кожного епізоду з епізодів, рівень яких був відомий на його початку.

    win_days=None: усі відомі епізоди (розширюване вікно), інакше лише за останні win_days діб.
    Глобальна частка → частка області (згладжування m_o) → частка області за годинним блоком (згладжування m_h).
    Для епізодів без жодної історії повертає NaN.
    """
    t = ep.started_at.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
    known_at = t + np.timedelta64(int(window_min * 60), "s")
    red, ob, bk = ep.red.to_numpy(float), ep.oblast.to_numpy(), ep.bucket.to_numpy()
    p = np.full(len(ep), np.nan)
    lo = None if win_days is None else np.timedelta64(int(win_days * 86400), "s")
    for i in range(len(ep)):
        use = known_at <= t[i]
        if lo is not None:
            use &= t > t[i] - lo
        if not use.any():
            continue
        g = red[use].mean()
        so = use & (ob == ob[i])
        po = (red[so].sum() + m_o * g) / (so.sum() + m_o)
        if m_h is not None:
            sh = so & (bk == bk[i])
            po = (red[sh].sum() + m_h * po) / (sh.sum() + m_h)
        p[i] = po
    return p


def combine(p_start, p_red):
    """P(початок червоної тривоги) = P(початок) × P(червона | початок). Припущення: рівень не залежить від факту початку."""
    return np.asarray(p_start) * np.asarray(p_red)
