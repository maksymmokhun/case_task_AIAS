"""Тести відсутності витоку: у момент t прогноз не може залежати від нічого, що сталося після t."""
import numpy as np
import pandas as pd
import pytest

from airalerts import baselines as bl
from airalerts.config import HORIZON_H, RAW, TEST_START, VAL_LAST_T
from airalerts.data import load_events
from airalerts.state import as_known_at, build_grid


def rows_at(grid, t):
    return np.flatnonzero((grid.t == t).to_numpy())


def test_conservative_state_matches_known_at_reference(events, t_end, probe_times):
    """Консервативний стан у сітці збігається з незалежним еталоном, побудованим лише з відомого в t."""
    grid = build_grid(events, t_end)
    for t in probe_times:
        known = as_known_at(events, t)
        for oblast, g in known.groupby("oblast"):
            last = g.iloc[-1]
            expected_active = pd.isna(last.finished_at)       # кінець відомий лише якщо відбій спостерігався
            row = grid[(grid.oblast == oblast) & (grid.t == t)].iloc[0]
            assert bool(row.active_cons) == expected_active, (oblast, t)


@pytest.mark.parametrize("model", ["b2", "b0_28", "b0_90"])
def test_trailing_baselines_ignore_future(events, t_end, probe_times, model):
    """Прогноз у t однаковий на повній історії й на історії, усіченій до t."""
    full = build_grid(events, t_end)
    f = {"b2": lambda g: bl.b2_trailing_rate(g, "elig_cons"),
         "b0_28": lambda g: bl.b0_hour_rate(g, "elig_cons", 28),
         "b0_90": lambda g: bl.b0_hour_rate(g, "elig_cons", 90)}[model]
    p_full = f(full)
    for t in probe_times:
        trunc = build_grid(as_known_at(events, t), t)
        np.testing.assert_allclose(p_full[rows_at(full, t)], f(trunc)[rows_at(trunc, t)], rtol=0, atol=1e-12)


def test_b1_ignores_future(events, t_end, probe_times):
    full = build_grid(events, t_end)
    for t in probe_times:
        trunc = build_grid(as_known_at(events, t), t)
        cut = t - pd.Timedelta(days=30)
        out = []
        for g in (full, trunc):
            out.append(bl.b1_fit_predict(g, "elig_cons", (g.t <= cut).to_numpy(), (g.t == t).to_numpy()))
        np.testing.assert_allclose(out[0][rows_at(full, t)], out[1][rows_at(trunc, t)], atol=1e-9)


@pytest.mark.parametrize("method", ["ets", "arima"])
def test_b3_ignores_future(events, t_end, probe_times, method):
    full = build_grid(events, t_end)
    for t in probe_times[:3]:
        trunc = build_grid(as_known_at(events, t), t)
        a = bl.b3_probabilities(full, events, method, rows_at(full, t))
        b = bl.b3_probabilities(trunc, as_known_at(events, t), method, rows_at(trunc, t))
        np.testing.assert_allclose(a[rows_at(full, t)], b[rows_at(trunc, t)], atol=1e-9)


def test_labels_inside_horizon_are_not_used(events, t_end, probe_times):
    """Якщо перевернути мітки z для всіх t' > t - H, прогнози в t не змінюються (правило «мітки з t' <= t - H»)."""
    full = build_grid(events, t_end)
    for t in probe_times:
        bad = full.copy()
        mask = (bad.t > t - pd.Timedelta(hours=HORIZON_H)).to_numpy()
        bad.loc[mask, "z"] = ~bad.loc[mask, "z"]
        for f in (lambda g: bl.b2_trailing_rate(g, "elig_cons"),
                  lambda g: bl.b0_hour_rate(g, "elig_cons", 28)):
            np.testing.assert_allclose(f(full)[rows_at(full, t)], f(bad)[rows_at(bad, t)], atol=1e-12)


def test_future_perturbation_does_not_change_state_or_predictions(events, t_end, probe_times):
    """Псуємо все після t (кінці тривалих тривог, нові початки, прапорець naive): стан і прогноз у t незмінні."""
    rng = np.random.default_rng(5)
    base = build_grid(events, t_end)
    for t in probe_times[:3]:
        ev = events.copy()
        ongoing = ev.finished_at > t
        ev.loc[ongoing, "finished_at"] += pd.to_timedelta(rng.integers(1, 5, ongoing.sum()), unit="h")
        future = ev.started_at > t
        ev.loc[future, "naive"] = rng.random(future.sum()) < 0.5
        ev = ev[~(future & (rng.random(len(ev)) < 0.3))]
        g = build_grid(ev, t_end)
        r_b, r_g = rows_at(base, t), rows_at(g, t)
        assert (base.active_cons.to_numpy()[r_b] == g.active_cons.to_numpy()[r_g]).all()
        np.testing.assert_allclose(bl.b2_trailing_rate(base, "elig_cons")[r_b],
                                   bl.b2_trailing_rate(g, "elig_cons")[r_g], atol=1e-12)


def test_validation_does_not_touch_test_period():
    assert VAL_LAST_T + pd.Timedelta(hours=HORIZON_H) < TEST_START


@pytest.mark.skipif(not (RAW / "volunteer_data_en.csv").exists(), reason="немає сирих даних")
def test_loader_cuts_test_period():
    assert load_events().started_at.max() < TEST_START


def test_validation_windows_and_training_purge(events, t_end):
    """Навчальні мітки закінчуються до початку кожного вікна; вікна послідовні й не заходять у тест."""
    from airalerts.splits import train_mask, val_windows
    wins = val_windows()
    assert all(a[1] == b[0] for a, b in zip(wins, wins[1:]))
    assert wins[-1][1] <= TEST_START - pd.Timedelta(hours=HORIZON_H)
    grid = build_grid(events, t_end)
    w0 = grid.t.min() + pd.Timedelta(days=100)
    for days in (None, 30):
        tm = train_mask(grid, w0, days)
        assert grid.t[tm].max() + pd.Timedelta(hours=HORIZON_H) < w0


def test_recalibration_ignores_labels_after_purge(events, t_end):
    """Рекалібрація для вікна з початком w0 не залежить від міток z для t > w0 - H - 1 год."""
    from airalerts.splits import train_mask
    grid = build_grid(events, t_end)
    rng = np.random.default_rng(7)
    p_raw = np.clip(rng.random(len(grid)) * 0.8 + 0.1, 0.01, 0.99)
    w0 = grid.t.min() + pd.Timedelta(days=120)
    pred = ((grid.t >= w0) & (grid.t < w0 + pd.Timedelta(days=7))).to_numpy()
    a = bl.recalibrate(grid, "elig_cons", p_raw, train_mask(grid, w0, 90), pred)
    bad = grid.copy()
    late = (bad.t > w0 - pd.Timedelta(hours=HORIZON_H + 1)).to_numpy()
    bad.loc[late, "z"] = ~bad.loc[late, "z"]
    b = bl.recalibrate(bad, "elig_cons", p_raw, train_mask(bad, w0, 90), pred)
    np.testing.assert_allclose(a[pred], b[pred], atol=1e-12)
