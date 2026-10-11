"""Ітерація 3: базові моделі B0–B3 на валідаційних вікнах за двома схемами розмітки.

Тест (останні 8 тижнів) не читається: load_events відсікає все з TEST_START.
Запуск: .venv/bin/python src/run_baselines.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import baselines as bl
from airalerts.baseline_preds import MODELS, SCHEMES, baseline_predictions
from airalerts.config import FIG, TAB, VAL_LAST_T
from airalerts.data import load_events
from airalerts.metrics import auc_within_groups, brier, reliability, summary
from airalerts.splits import train_mask, val_windows
from airalerts.state import build_grid



def main():
    t_start = time.time()
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    events = load_events()
    grid = build_grid(events, VAL_LAST_T)
    wins = val_windows()
    in_val = ((grid.t >= wins[0][0]) & (grid.t < wins[-1][1])).to_numpy()
    print(f"сітка: {len(grid):,} рядків; валідація {wins[0][0]} .. {wins[-1][1]}; рядків у валідації {in_val.sum():,}")

    preds = baseline_predictions(events, grid, wins)

    # чутливість B1 до довжини вікна навчання (консервативна схема): показує вплив нестаціонарності
    sens = []
    for days in (None, 365, 180, 90, 60):
        p = np.full(len(grid), np.nan)
        for w0, w1 in wins:
            pred = ((grid.t >= w0) & (grid.t < w1)).to_numpy()
            p[pred] = bl.b1_fit_predict(grid, "elig_cons", train_mask(grid, w0, days), pred)[pred]
        mk = grid.elig_cons.to_numpy() & in_val
        sens.append({"train_days": "вся історія" if days is None else days, "brier": brier(grid.z[mk], p[mk]),
                     "mean_pred": float(p[mk].mean()), "base_rate": float(grid.z[mk].mean())})
    pd.DataFrame(sens).round(4).to_csv(TAB / "b1_training_window.csv", index=False)
    print("\nB1: чутливість до вікна навчання\n", pd.DataFrame(sens).round(4).to_string(index=False))

    # ---- метрики ----
    recs = []
    for s, col in SCHEMES.items():
        el = grid[col].to_numpy()
        for m in MODELS:
            p = preds[s][m]
            for k, (w0, w1) in enumerate(wins):
                mk = el & ((grid.t >= w0) & (grid.t < w1)).to_numpy()
                recs.append({"scheme": s, "model": m, "window": f"V{k + 1}", **summary(grid.z[mk], p[mk])})
            mk = el & in_val
            recs.append({"scheme": s, "model": m, "window": "pooled", **summary(grid.z[mk], p[mk])})
    met = pd.DataFrame(recs)
    met.to_csv(TAB / "baselines_val_metrics.csv", index=False)

    pooled = met[met.window == "pooled"].pivot(index="model", columns="scheme", values="brier").loc[MODELS]
    cons = met[(met.scheme == "conservative") & (met.window == "pooled")].set_index("model").loc[MODELS]
    ref = cons.loc["B2_90d", "brier"]
    cons["bss_vs_B2"] = 1 - cons.brier / ref
    elc = grid.elig_cons.to_numpy() & in_val
    cons["auc_within_oblast"] = [auc_within_groups(grid.z[elc], preds["conservative"][m][elc], grid.oblast[elc]) for m in MODELS]
    best = cons.brier.idxmin()
    cons["bss_vs_best"] = 1 - cons.brier / cons.loc[best, "brier"]
    cons.round(4).to_csv(TAB / "baselines_val_summary_conservative.csv")
    print("\nНайкраща база за Brier (консервативна, pooled):", best)
    print(cons[["n", "base_rate", "brier", "log_loss", "pr_auc", "roc_auc", "auc_within_oblast", "bss_vs_B2"]].round(4).to_string())

    diff = pd.DataFrame({"brier_conservative": pooled.conservative, "brier_dataset": pooled.dataset})
    diff["delta"] = diff.brier_dataset - diff.brier_conservative
    n = met[met.window == "pooled"].pivot(index="model", columns="scheme", values="n").loc[MODELS]
    r = met[met.window == "pooled"].pivot(index="model", columns="scheme", values="base_rate").loc[MODELS]
    diff["n_cons"], diff["n_dataset"] = n.conservative, n.dataset
    diff["rate_cons"], diff["rate_dataset"] = r.conservative, r.dataset
    diff.round(4).to_csv(TAB / "baselines_val_scheme_difference.csv")
    print("\nРізниця схем:\n", diff.round(4).to_string())

    # по областях: найкраща база і B2
    el = grid.elig_cons.to_numpy() & in_val
    po = []
    for oblast, ix in grid[el].groupby("oblast", observed=True).indices.items():
        sub = grid[el].iloc[ix]
        y = sub.z.to_numpy()
        po.append({"oblast": oblast, "n": len(y), "positives": int(y.sum()), "brier_B2": brier(y, preds["conservative"]["B2_90d"][sub.index]),
                   f"brier_{best}": brier(y, preds["conservative"][best][sub.index])})
    po = pd.DataFrame(po).set_index("oblast")
    po["bss_best_vs_B2"] = 1 - po[f"brier_{best}"] / po.brier_B2
    po.round(4).to_csv(TAB / "baselines_val_by_oblast.csv"); print("\nПо областях:\n", po.round(4).to_string())

    # ---- графіки ----
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ідеал")
    mk = grid.elig_cons.to_numpy() & in_val
    for m in MODELS:
        rel = reliability(grid.z[mk], preds["conservative"][m][mk], bins=12)
        ax.plot(rel[:, 0], rel[:, 1], marker="o", ms=3, label=m)
    ax.set_xlabel("прогнозована ймовірність"); ax.set_ylabel("фактична частка початків")
    ax.set_title("Калібрування баз, валідація (консервативна схема)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "06_baselines_calibration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    w = met[(met.scheme == "conservative") & (met.window != "pooled")].pivot(index="window", columns="model", values="brier")[MODELS]
    w.plot(marker="o", ax=ax); ax.set_ylabel("Brier"); ax.set_title("Brier баз по валідаційних вікнах (консервативна схема)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "07_baselines_brier_by_window.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    diff[["brier_conservative", "brier_dataset"]].plot.bar(ax=ax, color=["#1f77b4", "#d62728"]); ax.set_ylabel("Brier")
    ax.set_title("Brier баз за двома схемами розмітки (валідація, pooled)"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(FIG / "08_baselines_scheme_difference.png"); plt.close(fig)
    print(f"\nГотово за {time.time() - t_start:.0f} с")


if __name__ == "__main__":
    main()
