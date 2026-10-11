"""Тести на витік для шару рівня (етап 2): прогноз для епізоду i не залежить від записів після початку i
і від рівнів епізодів, що почалися пізніше за start_i - W."""
import numpy as np
import pandas as pd
import pytest

from airalerts import levels as L

W = 60


def synthetic_archive(seed=0, days=25):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-09-10", tz="UTC")
    rows = []
    for ob in ("Kyiv City", "Kharkivska oblast", "Lvivska oblast"):
        t = t0
        while t < t0 + pd.Timedelta(days=days):
            t += pd.Timedelta(minutes=float(rng.exponential(300)))
            for k in range(int(rng.integers(1, 4))):                      # кілька районних записів на епізод
                s = t + pd.Timedelta(minutes=float(rng.exponential(15)))
                f = s + pd.Timedelta(minutes=float(rng.exponential(45)) + 1)
                lvl = "red" if rng.random() < (0.6 if ob == "Kharkivska oblast" else 0.15) else "yellow"
                rows.append((ob, s, f, lvl))
    d = pd.DataFrame(rows, columns=["oblast", "started_at", "finished_filled", "level"])
    return d.sort_values(["oblast", "started_at"]).reset_index(drop=True)


PARAMS = [(None, 20, None), (7, 5, 20), (3, 5, 80)]


@pytest.mark.parametrize("win,m_o,m_h", PARAMS)
def test_prediction_ignores_archive_after_start(win, m_o, m_h):
    arch = synthetic_archive()
    ep = L.episodes(arch, W)
    p = L.predict(ep, W, win, m_o, m_h)
    rng = np.random.default_rng(1)
    for i in rng.choice(np.flatnonzero(~np.isnan(p)), 12, replace=False):
        cut = L.episodes(arch[arch.started_at <= ep.started_at[i]], W)
        j = cut.index[(cut.oblast == ep.oblast[i]) & (cut.started_at == ep.started_at[i])][0]
        assert L.predict(cut, W, win, m_o, m_h)[j] == pytest.approx(p[i], abs=1e-12)


@pytest.mark.parametrize("win,m_o,m_h", PARAMS)
def test_prediction_ignores_levels_not_yet_known(win, m_o, m_h):
    """Рівні епізодів, що почалися пізніше за start_i - W, на момент i ще не відомі: їх зміна не рухає p_i."""
    ep = L.episodes(synthetic_archive(), W)
    p = L.predict(ep, W, win, m_o, m_h)
    rng = np.random.default_rng(2)
    for i in rng.choice(np.flatnonzero(~np.isnan(p)), 12, replace=False):
        ep2 = ep.copy()
        late = ep2.started_at > ep.started_at[i] - pd.Timedelta(minutes=W)
        late.iloc[i] = False                                                # рівень самого i не є ознакою для i
        ep2.loc[late, "red"] = 1 - ep2.loc[late, "red"]
        assert L.predict(ep2, W, win, m_o, m_h)[i] == pytest.approx(p[i], abs=1e-12)


def test_level_of_own_episode_is_not_a_feature():
    ep = L.episodes(synthetic_archive(), W)
    p = L.predict(ep, W, 7, 5, 20)
    i = int(np.flatnonzero(~np.isnan(p))[40])
    ep2 = ep.copy()
    ep2.loc[i, "red"] = 1 - ep2.loc[i, "red"]
    assert L.predict(ep2, W, 7, 5, 20)[i] == pytest.approx(p[i], abs=1e-12)


def test_episode_starts_do_not_depend_on_future_records():
    arch = synthetic_archive()
    ep = L.episodes(arch, W)
    for s in ep.started_at.iloc[::25]:
        cut = L.episodes(arch[arch.started_at <= s], W)
        pre = ep[ep.started_at <= s]
        assert list(cut.started_at) == list(pre.started_at)


def test_missing_finish_uses_fixed_fill_not_future():
    arch = pd.DataFrame({"oblast": ["Kyiv City"] * 3,
                         "started_at": pd.to_datetime(["2026-09-10 10:00", "2026-09-10 12:00", "2026-09-10 14:00"], utc=True),
                         "finished_filled": pd.to_datetime(["2026-09-10 13:00", "2026-09-10 12:30", "2026-09-10 14:30"], utc=True),
                         "level": ["yellow", "red", "yellow"]})
    ep = L.episodes(arch, W)
    assert len(ep) == 2                                   # 12:00 всередині активного запису 10:00..13:00; 14:00 новий епізод
    assert ep.red.tolist() == [0, 0]                      # червоний запис о 12:00 поза вікном W=60 хв від початку 10:00


def test_combine():
    assert L.combine(0.5, 0.2) == pytest.approx(0.1)
