"""Межа «минуле/майбутнє» у момент прогнозу t: подія рівно в t потрапляє в ознаки (started_at <= t),
але не в ціль, ціль це події в (t, t + H]: ліва межа виключена, права включена."""
import numpy as np
import pandas as pd
import pytest

from airalerts import features as F
from airalerts.config import HORIZON_H
from airalerts.state import build_grid

T = pd.Timestamp("2026-03-10 12:00", tz="UTC")
H = pd.Timedelta(hours=HORIZON_H)
TOY_NEIGHBORS = {"A oblast": ["B oblast"], "B oblast": ["A oblast", "C oblast"], "C oblast": ["B oblast"]}


@pytest.fixture(autouse=True)
def toy_neighbors(monkeypatch):
    monkeypatch.setattr(F, "NEIGHBORS", TOY_NEIGHBORS)


def make(extra=()):
    """Область A: давня тривога, тривога рівно в T (з відбоєм) і додаткові початки extra. B і C: по одній давній, щоб сітка була спільна."""
    old = T - pd.Timedelta(days=2)
    rows = [("A oblast", old, old + pd.Timedelta(minutes=10), False), ("B oblast", old, old + pd.Timedelta(minutes=10), False),
            ("C oblast", old, old + pd.Timedelta(minutes=10), False),
            ("A oblast", T, T + pd.Timedelta(minutes=20), False)]
    rows += [("A oblast", s, s + pd.Timedelta(minutes=10), False) for s in extra]
    ev = pd.DataFrame(rows, columns=["oblast", "started_at", "finished_at", "naive"])
    return ev.sort_values(["oblast", "started_at"]).reset_index(drop=True)


def row(grid, X, t, oblast="A oblast"):
    m = ((grid.t == t) & (grid.oblast.astype(str) == oblast)).to_numpy()
    assert m.sum() == 1
    return grid[m].iloc[0], X[m].iloc[0]


def build(ev):
    grid = build_grid(ev, T + pd.Timedelta(hours=8))
    return grid, F.build_features(grid, "elig_cons", F.state_features(ev, grid))


def test_event_at_t_is_feature_not_label_and_each_event_counts_once():
    grid, X = build(make())
    for k in range(-5, 3):
        t = T + pd.Timedelta(hours=k)
        g, x = row(grid, X, t)
        in_features = x["n_start_24h"] == 1                      # окрім події в T, у вікні 24 год нічого немає
        in_label = bool(g.z)
        assert in_features == (t >= T)
        assert in_label == (T - H <= t < T)                      # (t, t+H] містить T
        assert not (in_features and in_label)                    # подія не у обох частинах
    # події ліворуч від T-H ще не потрапляють у ціль, рівно T-H вже потрапляють (права межа включена)
    assert not row(grid, X, T - H - pd.Timedelta(hours=1))[0].z
    assert row(grid, X, T - H)[0].z


def test_row_at_event_time_is_not_eligible_but_hour_before_is():
    grid, X = build(make())
    g_t, _ = row(grid, X, T)
    g_before, _ = row(grid, X, T - pd.Timedelta(hours=1))
    assert not g_t.elig_cons and not g_t.elig_ds                 # тривога вже активна в t
    assert g_before.elig_cons and g_before.elig_ds and g_before.z


def test_state_features_at_t_see_event_at_t_but_not_the_hour_before():
    grid, X = build(make())
    _, x_t = row(grid, X, T)
    _, x_before = row(grid, X, T - pd.Timedelta(hours=1))
    assert x_t["n_start_1h"] == 1 and x_before["n_start_1h"] == 0
    assert x_t["since_start"] < x_before["since_start"]          # у t «щойно почалась», за годину до: давня подія
    assert x_before["n_start_3h"] == 0 and x_t["n_start_3h"] == 1


@pytest.mark.parametrize("offset,expected", [(pd.Timedelta(hours=3), True), (pd.Timedelta(hours=3, seconds=1), False),
                                              (pd.Timedelta(seconds=1), True)])
def test_label_right_boundary_is_inclusive_at_t_plus_h(offset, expected):
    """Наступний початок рівно в T+H входить у ціль рядка T, на секунду пізніше ні; початок через секунду після T входить."""
    grid, X = build(make(extra=[T + offset]))
    g, _ = row(grid, X, T)
    assert bool(g.z) == expected


def test_next_event_at_t_plus_h_is_not_in_features_of_row_t():
    grid, X = build(make(extra=[T + H]))
    _, x = row(grid, X, T)
    assert x["n_start_24h"] == 1                                 # лише подія в T; подія в T+H майбутнє для рядка T
    _, x_later = row(grid, X, T + H)
    assert x_later["n_start_24h"] == 2
