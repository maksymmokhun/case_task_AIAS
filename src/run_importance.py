"""Важливість окремих ознак на валідації (V1–V4), без тесту.

Тест не читається: load_events відсікає все з TEST_START, сітка закінчується на VAL_LAST_T.
Для обраних у run_model.py конфігурацій (логістична C=0,03/180 діб, HGB leaves=16, min_leaf=400, max_iter=100/365 діб):
- перестановка (permutation): ознака перемішується у прогнозному вікні, модель не перенавчається. Δ Brier і Δ AUC всередині області;
- виключення (drop-one): ознака замінюється константою і модель перенавчається в кожному вікні. Ловить випадки,
  коли ознаку заміняє інша (пріори, лічильники). Для ознак у парі з дублером перестановка завищує важливість,
  а виключення занижує, тому дивимось на обидва.
Додатково: AUC всередині кожної області для обох моделей і бази B0_90d.
Запуск: cd src && ../.venv/bin/python run_importance.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from airalerts import features as F
from airalerts import models as M
from airalerts.baseline_preds import SCHEMES, baseline_predictions
from airalerts.baselines import P_MAX, P_MIN
from airalerts.config import FIG, TAB, VAL_LAST_T
from airalerts.data import load_events
from airalerts.maps import plot_importance
from airalerts.metrics import auc_within_groups
from airalerts.splits import train_mask, val_windows
from airalerts.state import build_grid

SEL = {"logit": {"C": 0.03, "days": 180},
       "hgb": {"leaves": 16, "min_leaf": 400, "max_iter": 100, "days": 365}}
N_PERM = 5
GROUP_OF = {c: g for g, cols in F.GROUPS.items() for c in cols}


def main():
    t0 = time.time()
    events = load_events()
    grid = build_grid(events, VAL_LAST_T)
    assert grid.t.max() < pd.Timestamp("2026-08-15 04:00", tz="UTC"), "у сітці не має бути тесту"
    y = grid.z.to_numpy()
    wins = val_windows()
    el = grid.elig_cons.to_numpy()
    state = F.state_features(events, grid)
    X = F.build_features(grid, SCHEMES["conservative"], state)
    rng = np.random.default_rng(0)

    rows_all = []                       # прогнозні рядки V1–V4 в порядку вікон
    preds_full = {k: [] for k in SEL}
    perm = {k: {} for k in SEL}         # kind -> feature -> [Δ Brier, Δ AUC] усереднено за повторами (сума Brier за вікнами)
    drop = {k: {} for k in SEL}
    drop_win = {k: {} for k in SEL}     # kind -> feature -> список Δ Brier по вікнах

    pooled = {}
    for kind, par in SEL.items():
        fit_par = {k: v for k, v in par.items() if k != "days"}
        base_p, perm_p, drop_p = [], {}, {}
        win_rows = []
        for w, (w0, w1) in enumerate(wins):
            pred = np.flatnonzero(el & ((grid.t >= w0) & (grid.t < w1)).to_numpy())
            train = np.flatnonzero(el & train_mask(grid, w0, par["days"]))
            p, model, cols = M.fit_predict(kind, fit_par, X, y, train, pred, return_model=True)
            win_rows.append(pred); base_p.append(p)
            dense = M._design(X, pred, cols, kind == "logit")
            for j, c in enumerate(cols):                # у логістичній моделі oblast_code немає в cols: це блок one-hot нижче
                for _ in range(N_PERM):
                    d = dense.copy()
                    d[:, [j]] = d[rng.permutation(len(d))][:, [j]]
                    pp = np.clip(model.predict_proba(d)[:, 1], P_MIN, P_MAX)
                    perm_p.setdefault(c, {}).setdefault(w, []).append(pp)
            if kind == "logit":                      # блок області (one-hot) переставляємо цілим
                span = list(range(len(cols), len(cols) + M.N_OBLASTS))
                for _ in range(N_PERM):
                    d = dense.copy()
                    d[:, span] = d[rng.permutation(len(d))][:, span]
                    pp = np.clip(model.predict_proba(d)[:, 1], P_MIN, P_MAX)
                    perm_p.setdefault("oblast_code", {}).setdefault(w, []).append(pp)
            # drop-one: перенавчання з константною колонкою
            for c in F.ALL_COLS:
                if (kind == "logit" and c in M._LOGIT_DROP - {"oblast_code"}) or (kind == "hgb" and c in M._HGB_DROP):
                    continue
                Xc = X.copy(); Xc[c] = np.float32(0)
                drop_p.setdefault(c, {})[w] = M.fit_predict(kind, fit_par, Xc, y, train, pred)
            print(f"{kind} вікно V{w + 1} готово, {time.time() - t0:.0f} с", flush=True)

        rows = np.concatenate(win_rows)
        yy, og = y[rows], grid.oblast.to_numpy()[rows]
        full = np.concatenate(base_p)
        b_full = float(np.mean((full - yy) ** 2)); a_full = auc_within_groups(yy, full, og)
        pooled[kind] = (rows, full, b_full, a_full)
        preds_full[kind] = full
        print(f"{kind}: Brier {b_full:.5f}, AUC в області {a_full:.4f}")
        for c, byw in perm_p.items():
            db, da = [], []
            for r in range(N_PERM):
                pp = np.concatenate([byw[w][r] for w in range(len(wins))])
                db.append(float(np.mean((pp - yy) ** 2)) - b_full)
                da.append(a_full - auc_within_groups(yy, pp, og))
            perm[kind][c] = (np.mean(db), np.std(db), np.mean(da), np.std(da))
        for c, byw in drop_p.items():
            pp = np.concatenate([byw[w] for w in range(len(wins))])
            drop[kind][c] = (float(np.mean((pp - yy) ** 2)) - b_full, a_full - auc_within_groups(yy, pp, og))
            drop_win[kind][c] = [float(np.mean((byw[w] - y[win_rows[w]]) ** 2) - np.mean((base_p[w] - y[win_rows[w]]) ** 2))
                                 for w in range(len(wins))]

    # ---------- таблиця ----------
    recs = []
    for kind in SEL:
        for c in perm[kind]:
            pb, psd, pa, pasd = perm[kind][c]
            db, da = drop[kind].get(c, (np.nan, np.nan))
            dw = drop_win[kind].get(c, [np.nan] * 4)
            recs.append({"model": kind, "feature": c, "group": GROUP_OF[c],
                         "perm_dBrier": pb, "perm_dBrier_sd": psd, "perm_dAUC": pa, "perm_dAUC_sd": pasd,
                         "drop_dBrier": db, "drop_dAUC": da,
                         "drop_windows_positive": int(np.sum(np.array(dw) > 0)) if not np.isnan(db) else np.nan})
    imp = pd.DataFrame(recs)
    imp.round(6).to_csv(TAB / "model_val_feature_importance.csv", index=False)

    # ---------- AUC по областях ----------
    base = baseline_predictions(events, grid, wins)["conservative"]["B0_90d"]
    rows, _, _, _ = pooled["logit"]
    yy, og = y[rows], grid.oblast.to_numpy()[rows]
    ob = []
    for o in sorted(set(og)):
        m = og == o
        if min(yy[m].sum(), (1 - yy[m]).sum()) < 30:
            ob.append({"oblast": o, "n": int(m.sum()), "positives": int(yy[m].sum())}); continue
        ob.append({"oblast": o, "n": int(m.sum()), "positives": int(yy[m].sum()),
                   "auc_B0_90d": roc_auc_score(yy[m], base[rows][m]),
                   "auc_LOGIT": roc_auc_score(yy[m], preds_full["logit"][m]),
                   "auc_HGB": roc_auc_score(yy[m], preds_full["hgb"][m])})
    pd.DataFrame(ob).round(4).to_csv(TAB / "model_val_oblast_auc.csv", index=False)

    plot_importance(imp, FIG / "21_feature_importance.png")
    print(f"Готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
