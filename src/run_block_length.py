"""Обґрунтування довжини блока для блокового бутстрепу BSS (на валідації, тест не читається).

Різниця квадратичних помилок бази й моделі агрегується по добах (усі області разом). Якщо добові різниці
автокорельовані, блоки мають бути довшими за лаг, на якому кореляція зникає, інакше ІН буде занадто вузьким.
Незалежна перевірка: стандартна похибка за Newey–West.
Запуск: cd src && ../.venv/bin/python run_block_length.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import baselines as bl
from airalerts import features as F
from airalerts.config import FIG, PRIMARY, SECONDARY, TAB, VAL_LAST_T
from airalerts.data import load_events
from airalerts.metrics import block_bootstrap_bss, newey_west_se
from airalerts.splits import val_windows
from airalerts.state import build_grid
from run_model import run_config

LOGIT, HGB = PRIMARY, SECONDARY
BLOCKS = (1, 2, 3, 5, 7, 10, 14)
MAX_LAG = 14


def acf(x, nlag):
    x = np.asarray(x, float) - np.mean(x)
    return np.array([np.sum(x[k:] * x[:len(x) - k]) / np.sum(x * x) for k in range(1, nlag + 1)])


def main():
    events = load_events()
    grid = build_grid(events, VAL_LAST_T)
    y = grid.z.to_numpy()
    wins = val_windows()
    windows = {k + 1: w for k, w in enumerate(wins)}
    el = grid.elig_cons.to_numpy()
    X = F.build_features(grid, "elig_cons", F.state_features(events, grid))
    rows = np.flatnonzero(el & ((grid.t >= wins[0][0]) & (grid.t < wins[-1][1])).to_numpy())

    p_ref = bl.b0_hour_rate(grid, "elig_cons", 90)[rows]
    preds = {}
    for name, (kind, par) in (("LOGIT", LOGIT), ("HGB", HGB)):
        r = run_config(kind, par, X, y, grid, el, windows)
        full = np.full(len(grid), np.nan)
        for rr, p in r.values():
            full[rr] = p
        preds[name] = full[rows]
    yy, tt = y[rows].astype(float), grid.t.iloc[rows].reset_index(drop=True)
    day = tt.dt.floor("D")

    out, acf_rows, nw_rows = [], [], []
    for name, p in preds.items():
        d = pd.Series((p_ref - yy) ** 2 - (p - yy) ** 2).groupby(day).sum()       # виграш моделі за добу, сума по областях
        a = acf(d.to_numpy(), MAX_LAG)
        acf_rows += [{"model": name, "lag_days": k + 1, "acf": v} for k, v in enumerate(a)]
        band = 1.96 / np.sqrt(len(d))
        ref_loss = float(((p_ref - yy) ** 2).sum())
        for L in BLOCKS:
            widths = []
            for off in range(L):                                                   # зсув початку блоків: ІН не має залежати від нього
                _, lo, hi = block_bootstrap_bss(yy, p, p_ref, tt + pd.Timedelta(days=off), block_days=L, n_boot=2000, seed=off)
                widths.append(hi - lo)
            point, lo, hi = block_bootstrap_bss(yy, p, p_ref, tt, block_days=L, n_boot=2000, seed=0)
            out.append({"model": name, "block_days": L, "n_blocks": int(np.ceil(len(d) / L)), "bss": point, "ci_low": lo, "ci_high": hi,
                        "width": hi - lo, "width_min_offset": min(widths), "width_max_offset": max(widths)})
        for q in (0, 1, 3, 7, 14):
            se = newey_west_se(d.to_numpy(), q) * len(d) / ref_loss              # похибка суми різниць -> BSS
            point = d.sum() / ref_loss
            nw_rows.append({"model": name, "nw_lags": q, "bss": point, "se": se, "ci_low": point - 1.96 * se, "ci_high": point + 1.96 * se})
        print(f"{name}: ACF добових виграшів, лаги 1..7: {np.round(a[:7], 2)}; смуга ±{band:.2f}; діб {len(d)}")

    res, nw, ac = pd.DataFrame(out), pd.DataFrame(nw_rows), pd.DataFrame(acf_rows)
    res.round(5).to_csv(TAB / "bootstrap_block_length.csv", index=False)
    nw.round(5).to_csv(TAB / "bootstrap_newey_west.csv", index=False)
    ac.round(4).to_csv(TAB / "bootstrap_acf.csv", index=False)
    print("\nБлоковий бутстреп (n_boot=2000):\n", res.round(4).to_string(index=False))
    print("\nNewey–West:\n", nw.round(4).to_string(index=False))

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in ac.groupby("model"):
        ax[0].plot(g.lag_days, g.acf, marker="o", label=name)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axhline(band, color="gray", ls="--", lw=0.8); ax[0].axhline(-band, color="gray", ls="--", lw=0.8)
    ax[0].set_xlabel("лаг, діб"); ax[0].set_ylabel("автокореляція"); ax[0].set_title("Автокореляція добового виграшу моделі над B0_90d"); ax[0].legend()
    for name, g in res.groupby("model"):
        ax[1].plot(g.block_days, g.width, marker="o", label=name)
        ax[1].fill_between(g.block_days, g.width_min_offset, g.width_max_offset, alpha=0.2)
    ax[1].set_xlabel("довжина блока, діб"); ax[1].set_ylabel("ширина 95% ІН для BSS"); ax[1].set_title("Ширина ІН залежно від довжини блока"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "13_bootstrap_block_length.png"); plt.close(fig)


if __name__ == "__main__":
    main()
