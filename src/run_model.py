"""Ітерація 4: основна модель (логістична регресія, HistGradientBoosting) на валідаційних вікнах.

Тест (останні 8 тижнів) не читається: load_events відсікає все з TEST_START.
Гіперпараметри (довжина вікна навчання, регуляризація, розмір дерев) вибираються лише на валідації двома способами:
- «обрані» (sel): найкращий Brier на всіх 4 вікнах разом. Оцінка оптимістична, бо вибір і оцінка на тих самих вікнах;
- «вкладені» (nested): для вікна Vk конфігурацію вибирають за вікнами V0..V(k-1) (V0 лише для вибору), тож
  оцінка на Vk не бачила своїх даних під час вибору. Це чесна оцінка Brier після налаштування.
Запуск: cd src && ../.venv/bin/python run_model.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import features as F
from airalerts import models as M
from airalerts.baseline_preds import MODELS as BASE_MODELS
from airalerts.baseline_preds import SCHEMES, baseline_predictions
from airalerts.config import FIG, HORIZON_H, TAB, VAL_LAST_T
from airalerts.data import load_events
from airalerts.maps import map_val_bss
from airalerts.metrics import auc_within_groups, block_bootstrap_bss, brier, reliability, summary
from airalerts.splits import pre_window, train_mask, val_windows
from airalerts.state import build_grid

DAYS = (90, 180, 365)
CONFIGS = ([("logit", {"C": c, "days": d}) for d in DAYS for c in (0.03, 0.3, 3.0)] +
           [("hgb", {"leaves": lv, "min_leaf": ml, "max_iter": it, "days": d})
            for d in DAYS for lv in (8, 16) for ml in (100, 400) for it in (100, 250)])
PURGE = pd.Timedelta(hours=HORIZON_H + 1)
FAMILIES = {"logit": "LOGIT", "hgb": "HGB"}


def cfg_name(kind, par):
    return kind + "|" + ",".join(f"{k}={v}" for k, v in par.items())


def run_config(kind, par, X, y, grid, elig, windows, drop=()):
    """Прогнози конфігурації для кожного вікна: {k: (рядки, ймовірності)}."""
    out = {}
    fit_par = {k: v for k, v in par.items() if k != "days"}
    for k, (w0, w1) in windows.items():
        pred = np.flatnonzero(elig & ((grid.t >= w0) & (grid.t < w1)).to_numpy())
        train = np.flatnonzero(elig & train_mask(grid, w0, par["days"]))
        out[k] = (pred, M.fit_predict(kind, fit_par, X, y, train, pred, drop_groups=drop))
    return out


def score(grid, results, ks, before=None):
    """Brier конфігурації на вікнах ks; before обрізає рядки пізніші за t <= before (purge перед наступним вікном)."""
    se, n = 0.0, 0
    for k in ks:
        rows, p = results[k]
        keep = np.ones(len(rows), bool) if before is None else (grid.t.iloc[rows] <= before).to_numpy()
        se += float(((p[keep] - grid.z.to_numpy()[rows][keep]) ** 2).sum()); n += int(keep.sum())
    return se / n


def to_full(n, results, ks):
    out = np.full(n, np.nan)
    for k in ks:
        rows, p = results[k]
        out[rows] = p
    return out


def main():
    t_start = time.time()
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    events = load_events()
    grid = build_grid(events, VAL_LAST_T)
    y = grid.z.to_numpy()
    wins = val_windows()
    windows = {0: pre_window(), **{k + 1: w for k, w in enumerate(wins)}}     # V0 лише для вкладеного вибору
    ks_val = [1, 2, 3, 4]
    in_val = ((grid.t >= wins[0][0]) & (grid.t < wins[-1][1])).to_numpy()
    print(f"сітка {len(grid):,} рядків; V0 {windows[0][0]} .. {windows[0][1]}; валідація {wins[0][0]} .. {wins[-1][1]}")

    state = F.state_features(events, grid)
    X = {s: F.build_features(grid, col, state) for s, col in SCHEMES.items()}
    base = baseline_predictions(events, grid, wins)

    # ---------- налаштування на консервативній схемі ----------
    el = grid.elig_cons.to_numpy()
    res, rows_cfg = {}, []
    for kind, par in CONFIGS:
        t0 = time.time()
        name = cfg_name(kind, par)
        res[name] = run_config(kind, par, X["conservative"], y, grid, el, windows)
        rows_cfg.append({"config": name, "family": kind, **par,
                         **{f"brier_V{k}": score(grid, res[name], [k]) for k in range(5)},
                         "brier_V1_V4": score(grid, res[name], ks_val), "sec": round(time.time() - t0, 1)})
    cfg = pd.DataFrame(rows_cfg)
    cfg.round(5).to_csv(TAB / "model_val_configs.csv", index=False)
    print("\nКонфігурації (Brier V1–V4):\n", cfg[["config", "brier_V1_V4", "sec"]].round(5).to_string(index=False))

    chosen, nested = {}, {}
    for kind in FAMILIES:
        names = [cfg_name(k, p) for k, p in CONFIGS if k == kind]
        chosen[kind] = min(names, key=lambda n: score(grid, res[n], ks_val))
        pick = {}
        for k in ks_val:
            before = windows[k][0] - PURGE
            pick[k] = min(names, key=lambda n: score(grid, res[n], range(k), before))
        nested[kind] = pick
        print(f"\n{kind}: обрана конфігурація {chosen[kind]}\n  вкладений вибір по вікнах: {pick}")

    P = {s: {m: base[s][m] for m in BASE_MODELS} for s in SCHEMES}
    for kind, short in FAMILIES.items():
        P["conservative"][f"{short}_sel"] = to_full(len(grid), res[chosen[kind]], ks_val)
        out = np.full(len(grid), np.nan)
        for k in ks_val:
            rows, p = res[nested[kind][k]][k]
            out[rows] = p
        P["conservative"][f"{short}_nested"] = out

    # схема датасету: ті самі обрані конфігурації, перенавчені на придатних зразках цієї схеми
    el_ds = grid.elig_ds.to_numpy()
    for kind, short in FAMILIES.items():
        par = dict(next(p for k, p in CONFIGS if cfg_name(k, p) == chosen[kind]))
        r = run_config(kind, par, X["dataset"], y, grid, el_ds, {k: windows[k] for k in ks_val})
        P["dataset"][f"{short}_sel"] = to_full(len(grid), r, ks_val)

    # ---------- метрики ----------
    new = [f"{s}_{v}" for s in FAMILIES.values() for v in ("sel", "nested")]
    models = BASE_MODELS + new
    recs = []
    for s, col in SCHEMES.items():
        elg = grid[col].to_numpy()
        for m in models:
            if m not in P[s]:
                continue
            for k, (w0, w1) in enumerate(wins):
                mk = elg & ((grid.t >= w0) & (grid.t < w1)).to_numpy()
                recs.append({"scheme": s, "model": m, "window": f"V{k + 1}", **summary(y[mk], P[s][m][mk])})
            mk = elg & in_val
            recs.append({"scheme": s, "model": m, "window": "pooled", **summary(y[mk], P[s][m][mk])})
    met = pd.DataFrame(recs)
    met.to_csv(TAB / "model_val_metrics.csv", index=False)

    mk = el & in_val
    cons = met[(met.scheme == "conservative") & (met.window == "pooled")].set_index("model").loc[models]
    best_base = cons.loc[BASE_MODELS, "brier"].idxmin()
    cons["bss_vs_best_base"] = 1 - cons.brier / cons.loc[best_base, "brier"]
    cons["auc_within_oblast"] = [auc_within_groups(y[mk], P["conservative"][m][mk], grid.oblast[mk]) for m in models]
    ci = {m: block_bootstrap_bss(y[mk], P["conservative"][m][mk], P["conservative"][best_base][mk], grid.t[mk])
          for m in models}
    cons["bss_ci_low"], cons["bss_ci_high"] = [ci[m][1] for m in models], [ci[m][2] for m in models]
    cons.round(4).to_csv(TAB / "model_val_summary_conservative.csv")
    print(f"\nНайкраща база на валідації: {best_base}")
    print(cons[["n", "brier", "log_loss", "pr_auc", "roc_auc", "auc_within_oblast", "bss_vs_best_base", "bss_ci_low", "bss_ci_high"]].round(4).to_string())

    wtab = met[(met.scheme == "conservative") & (met.window != "pooled")].pivot(index="window", columns="model", values="brier")[models]
    print("\nBrier по вікнах:\n", wtab.round(4).T.to_string())

    diff = []
    for m in BASE_MODELS + [f"{s}_sel" for s in FAMILIES.values()]:
        a = met[(met.scheme == "conservative") & (met.window == "pooled") & (met.model == m)].iloc[0]
        b = met[(met.scheme == "dataset") & (met.window == "pooled") & (met.model == m)].iloc[0]
        diff.append({"model": m, "brier_conservative": a.brier, "brier_dataset": b.brier, "delta": b.brier - a.brier})
    diff = pd.DataFrame(diff).set_index("model")
    diff.round(4).to_csv(TAB / "model_val_scheme_difference.csv")
    print("\nРізниця схем:\n", diff.round(4).to_string())

    # ---------- по областях ----------
    top = min(new, key=lambda m: cons.loc[m, "brier"])
    po = []
    for oblast, ix in grid[mk].groupby("oblast", observed=True).indices.items():
        sub = grid[mk].iloc[ix]
        yy = sub.z.to_numpy()
        po.append({"oblast": oblast, "n": len(yy), "positives": int(yy.sum()),
                   f"brier_{best_base}": brier(yy, P["conservative"][best_base][sub.index]),
                   f"brier_{top}": brier(yy, P["conservative"][top][sub.index])})
    po = pd.DataFrame(po).set_index("oblast")
    po["bss_vs_best_base"] = 1 - po[f"brier_{top}"] / po[f"brier_{best_base}"]
    po.round(4).to_csv(TAB / "model_val_by_oblast.csv")
    print(f"\nПо областях ({top} проти {best_base}):\n", po.round(4).to_string())

    # ---------- абляція груп ознак: що дає приріст ----------
    abl = []
    for kind, short in FAMILIES.items():
        par = dict(next(p for k, p in CONFIGS if cfg_name(k, p) == chosen[kind]))
        full = score(grid, res[chosen[kind]], ks_val)
        for g in F.GROUPS:
            r = run_config(kind, par, X["conservative"], y, grid, el, {k: windows[k] for k in ks_val}, drop=(g,))
            abl.append({"family": short, "dropped_group": g, "brier": score(grid, r, ks_val), "brier_full": full})
    abl = pd.DataFrame(abl)
    abl["delta_brier"] = abl.brier - abl.brier_full
    abl.round(5).to_csv(TAB / "model_val_ablation.csv", index=False)
    print("\nАбляція (Δ Brier після виключення групи, + означає, що група корисна):\n", abl.round(5).to_string(index=False))

    # ---------- графіки ----------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ідеал")
    for m in [best_base, "B3_ARIMA_cal", "LOGIT_nested", "HGB_nested"]:
        rel = reliability(y[mk], P["conservative"][m][mk], bins=12)
        ax.plot(rel[:, 0], rel[:, 1], marker="o", ms=3, label=m)
    ax.set_xlabel("прогнозована ймовірність"); ax.set_ylabel("фактична частка початків")
    ax.set_title("Калібрування, валідація (консервативна схема)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "09_model_calibration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    wtab[[best_base, "B3_ARIMA_cal", "LOGIT_nested", "HGB_nested"]].plot(marker="o", ax=ax)
    ax.set_ylabel("Brier"); ax.set_title("Brier по валідаційних вікнах: бази проти основної моделі"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "10_model_brier_by_window.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    abl.pivot(index="dropped_group", columns="family", values="delta_brier").loc[list(F.GROUPS)].plot.barh(ax=ax)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Δ Brier після виключення групи (більше = група корисніша)")
    ax.set_title("Абляція груп ознак, валідація"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG / "11_model_ablation.png"); plt.close(fig)

    map_val_bss(po, top, best_base, FIG / "12_model_bss_by_oblast.png")   # карта замість стовпчастої діаграми
    print(f"\nГотово за {time.time() - t_start:.0f} с")


if __name__ == "__main__":
    main()
