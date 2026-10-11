"""Тести відсутності витоку для ознак і основної моделі (ітерація 4)."""
import numpy as np
import pandas as pd
import pytest

from airalerts import features as F
from airalerts import models as M
from airalerts.config import HORIZON_H
from airalerts.splits import train_mask
from airalerts.state import as_known_at, build_grid

TOY_NEIGHBORS = {"A oblast": ["B oblast"], "B oblast": ["A oblast", "C oblast"], "C oblast": ["B oblast"]}


@pytest.fixture(autouse=True)
def toy_neighbors(monkeypatch):
    monkeypatch.setattr(F, "NEIGHBORS", TOY_NEIGHBORS)


def feats(events, t_last, elig_col):
    grid = build_grid(events, t_last)
    return grid, F.build_features(grid, elig_col, F.state_features(events, grid))


def at(grid, X, t):
    """Ознаки в момент t, впорядковані за областю."""
    m = (grid.t == t).to_numpy()
    return X[m].set_index(grid.oblast[m].astype(str).to_numpy()).sort_index()


@pytest.mark.parametrize("elig_col", ["elig_cons", "elig_ds"])
def test_features_ignore_future_by_truncation(events, t_end, probe_times, elig_col):
    """Усі ознаки в t, пораховані на повній історії, збігаються з ознаками, порахованими лише на відомому в t.

    Консервативна схема: усічення за as_known_at (кінці й naive невідомі). Схема датасету: усічення лише початків,
    бо її придатні зразки самі спираються на ретроспективні кінці датасету.
    """
    g_full, X_full = feats(events, t_end, elig_col)
    for t in probe_times:
        known = as_known_at(events, t) if elig_col == "elig_cons" else events[events.started_at <= t]
        g_tr, X_tr = feats(known, t, elig_col)
        pd.testing.assert_frame_equal(at(g_full, X_full, t), at(g_tr, X_tr, t), check_exact=False, atol=1e-5, rtol=0)


def test_features_ignore_future_perturbation(events, t_end, probe_times):
    """Псуємо все після t (кінці тривалих тривог, нові початки, прапорець naive): ознаки в t незмінні."""
    rng = np.random.default_rng(11)
    g0, X0 = feats(events, t_end, "elig_cons")
    for t in probe_times[:3]:
        ev = events.copy()
        ongoing = ev.finished_at > t
        ev.loc[ongoing, "finished_at"] += pd.to_timedelta(rng.integers(1, 5, ongoing.sum()), unit="h")
        future = ev.started_at > t
        ev.loc[future, "naive"] = rng.random(future.sum()) < 0.5
        ev = ev[~(future & (rng.random(len(ev)) < 0.3))]
        g1, X1 = feats(ev, t_end, "elig_cons")
        pd.testing.assert_frame_equal(at(g0, X0, t), at(g1, X1, t), check_exact=False, atol=1e-5, rtol=0)


def test_features_ignore_labels_inside_horizon(events, t_end, probe_times):
    """Перевертання міток z для всіх t' > t - H не змінює жодної ознаки в t."""
    grid = build_grid(events, t_end)
    state = F.state_features(events, grid)
    base = F.build_features(grid, "elig_cons", state)
    for t in probe_times:
        bad = grid.copy()
        late = (bad.t > t - pd.Timedelta(hours=HORIZON_H)).to_numpy()
        bad.loc[late, "z"] = ~bad.loc[late, "z"]
        X_bad = F.build_features(bad, "elig_cons", state)
        pd.testing.assert_frame_equal(at(grid, base, t), at(bad, X_bad, t), check_exact=False, atol=1e-5, rtol=0)


def test_synthetic_end_is_not_used(events, t_end, probe_times):
    """Синтетичний кінець naive-тривог (+30 хв) не впливає на ознаки: зсуваємо їх, ознаки лишаються."""
    g0, X0 = feats(events, t_end, "elig_cons")
    ev = events.copy()
    ev.loc[ev.naive, "finished_at"] += pd.Timedelta(hours=2)
    g1, X1 = feats(ev, t_end, "elig_cons")
    for t in probe_times:
        pd.testing.assert_frame_equal(at(g0, X0, t), at(g1, X1, t), check_exact=False, atol=1e-5, rtol=0)


def test_neighbor_map_is_symmetric_and_complete():
    for o, nbs in TOY_NEIGHBORS.items():
        for n in nbs:
            assert o in TOY_NEIGHBORS[n]


def test_real_neighbor_map_is_symmetric_and_complete():
    real = F._ADJ                                   # справжній словник (фікстура підміняє лише F.NEIGHBORS)
    assert len(real) == 24
    for o, nbs in real.items():
        assert nbs and o not in nbs
        for n in nbs:
            assert n in real and o in real[n], (o, n)


@pytest.mark.parametrize("kind,params", [("logit", {"C": 0.3}), ("hgb", {"max_iter": 30, "leaves": 8, "min_leaf": 50})])
def test_model_ignores_labels_after_purge(events, t_end, kind, params):
    """Прогноз вікна з початком w0 не залежить від міток z для t > w0 - H - 1 год."""
    grid, X = feats(events, t_end, "elig_cons")
    w0 = grid.t.min() + pd.Timedelta(days=120)
    el = grid.elig_cons.to_numpy()
    pred = np.flatnonzero(el & ((grid.t >= w0) & (grid.t < w0 + pd.Timedelta(days=7))).to_numpy())
    train = np.flatnonzero(el & train_mask(grid, w0, 90))
    y = grid.z.to_numpy().copy()
    p_a = M.fit_predict(kind, params, X, y, train, pred)
    y_bad = y.copy()
    y_bad[(grid.t > w0 - pd.Timedelta(hours=HORIZON_H + 1)).to_numpy()] ^= True
    p_b = M.fit_predict(kind, params, X, y_bad, train, pred)
    np.testing.assert_allclose(p_a, p_b, atol=1e-9)


@pytest.mark.parametrize("kind,params", [("logit", {"C": 0.3}), ("hgb", {"max_iter": 30, "leaves": 8, "min_leaf": 50})])
def test_model_ignores_features_outside_train_and_pred(events, t_end, kind, params):
    """Масштаб і дерева оцінюються лише на навчальних рядках: зіпсовані ознаки решти рядків (пізніші за вікно) нічого не змінюють."""
    grid, X = feats(events, t_end, "elig_cons")
    w0 = grid.t.min() + pd.Timedelta(days=120)
    w1 = w0 + pd.Timedelta(days=7)
    el = grid.elig_cons.to_numpy()
    pred = np.flatnonzero(el & ((grid.t >= w0) & (grid.t < w1)).to_numpy())
    train = np.flatnonzero(el & train_mask(grid, w0, 90))
    y = grid.z.to_numpy()
    p_a = M.fit_predict(kind, params, X, y, train, pred)
    X_bad = X.copy()
    later = (grid.t >= w1).to_numpy()
    X_bad.loc[later, :] = X_bad.loc[later, :].to_numpy() * 7.0 + 3.0
    p_b = M.fit_predict(kind, params, X_bad, y, train, pred)
    np.testing.assert_allclose(p_a, p_b, atol=1e-9)
