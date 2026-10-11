"""Прототип: чи додають багатші ознаки сусідів сигналу до основної моделі. Лише валідація (V1–V4), тест не читається.

Гіпотеза (за run_importance.py): із групи «сусіди» працює лише `nb_start_3h` (середнє початків сусідів за 3 год), і то
нелінійно (HGB). Імовірно, середнє по сусідах розбавляє сигнал, а годинна сітка його згладжує. Пробуємо варіанти:
  G1 вікна:      nb_start_1h, nb_start_6h                                   (інші часові масштаби)
  G2 агрегати:   nb_sum_start_3h, nb_max_start_3h, nb_frac_start_3h,
                 nb_since_min (час від останнього початку у будь-якого сусіда), nb_active_max
  G3 друге коло: nb2_start_3h, nb2_start_24h                                (сусіди сусідів, поширення)
Гіперпараметри обох моделей ті самі, що вибрані в run_model.py (без перепідбору), тож приріст не від налаштування.
Порівняння парне: BSS кожного варіанта проти базового набору ознак (V0) з блоковим бутстрепом (блоки по 3 доби).
Усі ознаки в момент t використовують лише початки з started_at <= t і стан активності за t' <= t.
Запуск: cd src && ../.venv/bin/python run_neighbors.py
"""
import time

import numpy as np
import pandas as pd

from airalerts import features as F
from airalerts import models as M
from airalerts.baseline_preds import SCHEMES
from airalerts.config import TAB, TEST_START, VAL_LAST_T
from airalerts.data import load_events
from airalerts.metrics import auc_within_groups, block_bootstrap_bss
from airalerts.splits import train_mask, val_windows
from airalerts.state import build_grid, to_ns

SEL = {"logit": {"C": 0.03, "days": 180},
       "hgb": {"leaves": 16, "min_leaf": 400, "max_iter": 100, "days": 365}}
H_NS = np.timedelta64(1, "h")
GROUPS_NEW = {"G1_windows": ["nb_start_1h", "nb_start_6h"],
              "G2_aggregates": ["nb_sum_start_3h", "nb_max_start_3h", "nb_frac_start_3h", "nb_since_min", "nb_active_max"],
              "G3_ring2": ["nb2_start_3h", "nb2_start_24h"]}
VARIANTS = {"V0_base": [], "V1_windows": GROUPS_NEW["G1_windows"], "V2_aggregates": GROUPS_NEW["G2_aggregates"],
            "V3_ring2": GROUPS_NEW["G3_ring2"], "V4_all": sum(GROUPS_NEW.values(), [])}


def neighbor_features(events, grid):
    """Нові ознаки сусідів: DataFrame з індексом grid і колонками з GROUPS_NEW."""
    idx = grid.groupby("oblast", observed=True).indices
    order = sorted(idx)
    tt = to_ns(grid.t.iloc[idx[order[0]]])
    n_o, n_h = len(order), len(tt)

    cnt = {w: np.zeros((n_o, n_h)) for w in (1, 3, 6, 24)}
    since = np.zeros((n_o, n_h))
    for j, o in enumerate(order):
        s = np.sort(to_ns(events[events.oblast == o].started_at))
        n_started = np.searchsorted(s, tt, "right")
        for w in cnt:
            cnt[w][j] = n_started - np.searchsorted(s, tt - w * H_NS, "right")
        last = n_started - 1
        since[j] = np.where(last >= 0, (tt - s[np.clip(last, 0, None)]) / H_NS, F.CAP_H)
    active = np.vstack([grid.active_cons.to_numpy(float)[idx[o]] for o in order])

    mean_adj = F.adjacency(order)                       # нормована по рядках
    adj = (mean_adj > 0).astype(float)                  # 0/1
    ring2 = ((adj @ adj) > 0) & (adj == 0) & ~np.eye(n_o, dtype=bool)
    ring2 = ring2.astype(float)
    ring2 = ring2 / np.maximum(ring2.sum(axis=1, keepdims=True), 1)
    nbrs = [np.flatnonzero(adj[i]) for i in range(n_o)]

    mats = {"nb_start_1h": mean_adj @ cnt[1], "nb_start_6h": mean_adj @ cnt[6],
            "nb_sum_start_3h": adj @ cnt[3],
            "nb_max_start_3h": np.vstack([cnt[3][nb].max(axis=0) for nb in nbrs]),
            "nb_frac_start_3h": mean_adj @ (cnt[3] > 0).astype(float),
            "nb_since_min": np.log1p(np.vstack([since[nb].min(axis=0) for nb in nbrs]).clip(max=F.CAP_H)),
            "nb_active_max": np.vstack([active[nb].max(axis=0) for nb in nbrs]),
            "nb2_start_3h": ring2 @ cnt[3], "nb2_start_24h": ring2 @ cnt[24]}
    out = {}
    for name, m in mats.items():
        arr = np.empty(len(grid), np.float32)
        for j, o in enumerate(order):
            arr[idx[o]] = m[j]
        out[name] = arr
    return pd.DataFrame(out, index=grid.index)


def main():
    t0 = time.time()
    events = load_events()
    grid = build_grid(events, VAL_LAST_T)
    assert grid.t.max() < TEST_START, "у сітці не має бути тесту"
    y = grid.z.to_numpy()
    wins = val_windows()
    el = grid.elig_cons.to_numpy()
    X = F.build_features(grid, SCHEMES["conservative"], F.state_features(events, grid))
    NB = neighbor_features(events, grid)
    X = pd.concat([X, NB], axis=1)
    print(f"ознаки готові, {time.time() - t0:.0f} с", flush=True)

    preds, rows_by_win = {}, []
    for w0, w1 in wins:
        rows_by_win.append(np.flatnonzero(el & ((grid.t >= w0) & (grid.t < w1)).to_numpy()))
    rows = np.concatenate(rows_by_win)
    yy, og, tv = y[rows], grid.oblast.to_numpy()[rows], grid.t.iloc[rows]
    for kind, par in SEL.items():
        fit_par = {k: v for k, v in par.items() if k != "days"}
        for name, extra in VARIANTS.items():
            out = []
            for (w0, w1), pred in zip(wins, rows_by_win):
                train = np.flatnonzero(el & train_mask(grid, w0, par["days"]))
                out.append(M.fit_predict(kind, fit_par, X, y, train, pred, extra=tuple(extra)))
            preds[(kind, name)] = np.concatenate(out)
            print(f"{kind} {name} готово, {time.time() - t0:.0f} с", flush=True)

    recs = []
    for kind in SEL:
        ref = preds[(kind, "V0_base")]
        b_ref = float(np.mean((ref - yy) ** 2)); a_ref = auc_within_groups(yy, ref, og)
        bounds = np.cumsum([0] + [len(r) for r in rows_by_win])
        for name in VARIANTS:
            p = preds[(kind, name)]
            est, lo, hi = block_bootstrap_bss(yy, p, ref, tv)
            per_win = [float(np.mean((ref[a:b] - yy[a:b]) ** 2) - np.mean((p[a:b] - yy[a:b]) ** 2))
                       for a, b in zip(bounds[:-1], bounds[1:])]
            recs.append({"model": kind, "variant": name, "n_new": len(VARIANTS[name]),
                         "brier": float(np.mean((p - yy) ** 2)), "auc_in_oblast": auc_within_groups(yy, p, og),
                         "dAUC_vs_V0": auc_within_groups(yy, p, og) - a_ref,
                         "bss_vs_V0": est, "bss_lo": lo, "bss_hi": hi,
                         "windows_better": int(np.sum(np.array(per_win) > 0)) if name != "V0_base" else np.nan})
    res = pd.DataFrame(recs)
    res.round(5).to_csv(TAB / "neighbors_val_metrics.csv", index=False)
    print("\n", res.round(5).to_string(index=False))

    # по областях: Δ AUC найкращого варіанта проти V0 для обох моделей
    from sklearn.metrics import roc_auc_score
    best = res[res.variant != "V0_base"].sort_values("bss_vs_V0", ascending=False)
    po = []
    for o in sorted(set(og)):
        m = og == o
        if min(yy[m].sum(), (1 - yy[m]).sum()) < 30:
            continue
        row = {"oblast": o, "positives": int(yy[m].sum())}
        for kind in SEL:
            v = best[best.model == kind].iloc[0].variant
            row[f"{kind}_best_variant"] = v
            row[f"{kind}_dAUC"] = roc_auc_score(yy[m], preds[(kind, v)][m]) - roc_auc_score(yy[m], preds[(kind, "V0_base")][m])
        po.append(row)
    po = pd.DataFrame(po)
    po.round(4).to_csv(TAB / "neighbors_val_by_oblast.csv", index=False)
    print("\nΔ AUC по областях (найкращий варіант проти V0):\n", po.round(4).to_string(index=False))
    print(f"Готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
