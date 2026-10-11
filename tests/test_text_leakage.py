"""Текстові ознаки в момент t не залежать від повідомлень після t."""
import numpy as np
import pandas as pd
import pytest

from airalerts import text as T
from airalerts.state import build_grid

from conftest import make_events

T0 = pd.Timestamp("2026-03-01 00:00", tz="UTC")


def synthetic_messages(seed=0, n=400):
    rng = np.random.default_rng(seed)
    times = T0 + pd.to_timedelta(np.sort(rng.uniform(0, 14 * 24 * 3600, n)), unit="s")
    words = ["Харківщина", "Київщина", "Загроза з півночі", "Відбій", "Одеса", "Київ", "БпЛА на Харківщину", "Ракети на Київщину", "Балістика по Одесі"]
    return pd.DataFrame({"id": np.arange(n), "time": times, "text": [words[i] for i in rng.integers(0, len(words), n)],
                         "edited": rng.random(n) < 0.05})


def grid_for(msgs):
    ev = make_events()
    ev = ev.assign(oblast=ev.oblast.map({"A oblast": "Kharkivska oblast", "B oblast": "Kyivska oblast", "C oblast": "Kyiv City"}))
    g = build_grid(ev, T0 + pd.Timedelta(days=13))
    return g[g.t >= T0 + pd.Timedelta(days=1)].reset_index(drop=True)


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("cut_h", [30, 100, 250])
def test_features_ignore_messages_after_t(cut_h, typed):
    msgs = synthetic_messages()
    grid = grid_for(msgs)
    t = T0 + pd.Timedelta(hours=cut_h)
    row = (grid.t == t).to_numpy()
    full = T.text_features(grid, msgs, typed)[row]
    cut = T.text_features(grid, msgs[msgs.time <= t].reset_index(drop=True), typed)[row]
    pd.testing.assert_frame_equal(full, cut)
    # будь-яка зміна повідомлень після t теж не впливає
    mutated = msgs.copy()
    after = mutated.time > t
    mutated.loc[after, "text"] = "Харківщина Київщина Київ Одеса БпЛА ракети балістика"
    mutated.loc[after, "edited"] = False
    pd.testing.assert_frame_equal(full, T.text_features(grid, mutated, typed)[row])


def test_message_at_exactly_t_is_included_and_window_is_left_open():
    msgs = pd.DataFrame({"id": [1, 2], "time": [T0 + pd.Timedelta(hours=40), T0 + pd.Timedelta(hours=41)],
                         "text": ["Харківщина", "Харківщина"], "edited": [False, False]})
    grid = grid_for(msgs)
    f = T.text_features(grid, msgs)
    row = ((grid.t == T0 + pd.Timedelta(hours=41)) & (grid.oblast == "Kharkivska oblast")).to_numpy()
    assert f.loc[row, "txt_ob_1h"].iloc[0] == pytest.approx(np.log1p(1))     # (t-1 год, t]: лише повідомлення рівно в t
    assert f.loc[row, "txt_ob_3h"].iloc[0] == pytest.approx(np.log1p(2))


def test_edited_messages_excluded_from_mentions_but_counted_overall():
    msgs = pd.DataFrame({"id": [1], "time": [T0 + pd.Timedelta(hours=45)], "text": ["Харківщина"], "edited": [True]})
    grid = grid_for(msgs)
    f = T.text_features(grid, msgs)
    row = ((grid.t == T0 + pd.Timedelta(hours=45)) & (grid.oblast == "Kharkivska oblast")).to_numpy()
    assert f.loc[row, "txt_ob_1h"].iloc[0] == 0
    assert f.loc[row, "txt_all_1h"].iloc[0] == pytest.approx(np.log1p(1))


def test_stems_distinguish_kyiv_city_and_oblast():
    low = lambda s: {o for o, p in T.PATTERNS.items() if p.search(s.lower())}
    assert low("Ракети на Київщину") == {"Kyivska oblast"}
    assert low("Вибухи в Києві") == {"Kyiv City"}
    assert low("Харкова і Харківщини") == {"Kharkivska oblast"}


def test_typed_counts_require_both_oblast_and_type():
    msgs = pd.DataFrame({"id": [1, 2, 3], "time": [T0 + pd.Timedelta(hours=44, minutes=m) for m in (10, 20, 30)],
                         "text": ["БпЛА на Харківщину", "Ракети на Київщину", "Харківщина спокійно"], "edited": [False] * 3})
    grid = grid_for(msgs)
    f = T.text_features(grid, msgs, typed=True)
    row = ((grid.t == T0 + pd.Timedelta(hours=45)) & (grid.oblast == "Kharkivska oblast")).to_numpy()
    assert f.loc[row, "txt_ob_1h"].iloc[0] == pytest.approx(np.log1p(2))            # дві згадки Харківщини
    assert f.loc[row, "txt_ob_drone_1h"].iloc[0] == pytest.approx(np.log1p(1))      # лише одна з БпЛА
    assert f.loc[row, "txt_ob_missile_1h"].iloc[0] == 0                              # ракета була про Київщину
