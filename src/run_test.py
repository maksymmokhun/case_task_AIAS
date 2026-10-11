"""Ітерація 5: ЄДИНИЙ запуск на тесті (останні 8 тижнів, від TEST_START).

Протокол зафіксовано до запуску в docs/definition_of_done.md, розділ «Основна модель і протокол тесту»:
- основна модель: логістична регресія (PRIMARY), додаткова: HGB (SECONDARY); конфігурації не змінюються;
- одне перенавчання на даних до TEST_START − (H+1) год, повторного налаштування немає;
- вердикт: BSS основної моделі проти B0_90d, консервативна схема, блоковий бутстреп (3 доби), нижня межа 95% ІН > 0;
- блоки по 7 діб і Newey–West (лаг 3) лише описово;
- решта таблиць описові; нічого з них не повертається в налаштування.
Це перше місце, де читаються дані з TEST_START (load_events(cut_test=False)).
Запуск: cd src && ../.venv/bin/python run_test.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import features as F
from airalerts.baseline_preds import MODELS as BASE_MODELS
from airalerts.baseline_preds import SCHEMES, baseline_predictions
from airalerts.config import (BLOCK_DAYS, BLOCK_DAYS_ROBUST, FIG, HORIZON_H, N_BOOT, PRIMARY, REGIME_CHANGE, SECONDARY,
                              TAB, TEST_START)
from airalerts.data import load_events
from airalerts.metrics import auc_within_groups, block_bootstrap_bss, brier, newey_west_se, reliability, summary
from airalerts.splits import train_mask
from airalerts.state import build_grid
from run_model import run_config

REF = "B0_90d"                  # база для правила успіху (обрана на валідації)
MIN_POS = 30                    # області з меншою кількістю позитивів позначаються й не інтерпретуються
NEW = {"LOGIT": PRIMARY, "HGB": SECONDARY}


def bss_row(y, p, p_ref, t, block_days=BLOCK_DAYS):
    point, lo, hi = block_bootstrap_bss(y, p, p_ref, t, block_days=block_days, n_boot=N_BOOT, seed=0)
    return {"bss": point, "ci_low": lo, "ci_high": hi}


def main():
    t_start = time.time()
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    events = load_events(cut_test=False)
    t_last = max(events.started_at.max(), events.finished_at.max()).floor("h") - pd.Timedelta(hours=HORIZON_H)
    grid = build_grid(events, t_last)                           # права межа обрізана на H год: далі ціль невідома
    y = grid.z.to_numpy()
    t_end = t_last + pd.Timedelta(hours=1)
    window = (TEST_START, t_end)
    test_rows = ((grid.t >= TEST_START) & (grid.t < t_end)).to_numpy()
    print(f"знімок даних: останній початок {events.started_at.max()}, сітка до {t_last}; тест {TEST_START} .. {t_last}")

    # перевірка меж: навчання закінчується до тесту (purge), тест не починається раніше за TEST_START
    for days in (PRIMARY[1]["days"], SECONDARY[1]["days"], 365):
        tm = train_mask(grid, TEST_START, days)
        assert grid.t[tm].max() + pd.Timedelta(hours=HORIZON_H) < TEST_START
    assert grid.t[test_rows].min() >= TEST_START

    state = F.state_features(events, grid)
    X = {s: F.build_features(grid, col, state) for s, col in SCHEMES.items()}
    base = baseline_predictions(events, grid, [window])         # бази: те саме вікно, одне перенавчання/рекалібрація

    P = {s: {m: base[s][m] for m in BASE_MODELS} for s in SCHEMES}
    for s, col in SCHEMES.items():
        el = grid[col].to_numpy()
        for name, (kind, par) in NEW.items():
            rows, p = run_config(kind, par, X[s], y, grid, el, {1: window})[1]
            full = np.full(len(grid), np.nan)
            full[rows] = p
            P[s][name] = full
    models = BASE_MODELS + list(NEW)

    # ---------- метрики ----------
    recs = []
    for s, col in SCHEMES.items():
        mk = grid[col].to_numpy() & test_rows
        for m in models:
            recs.append({"scheme": s, "model": m, **summary(y[mk], P[s][m][mk])})
    met = pd.DataFrame(recs)
    mk = grid.elig_cons.to_numpy() & test_rows
    yc, tc, pc = y[mk].astype(float), grid.t[mk].reset_index(drop=True), {m: P["conservative"][m][mk] for m in models}
    ref_p = pc[REF]
    cons = met[met.scheme == "conservative"].set_index("model").loc[models].copy()
    cons["auc_within_oblast"] = [auc_within_groups(yc, pc[m], grid.oblast[mk]) for m in models]
    cons["bss_vs_B0_90d"] = [1 - brier(yc, pc[m]) / brier(yc, ref_p) for m in models]
    cons.round(4).to_csv(TAB / "test_summary_conservative.csv")
    met.round(4).to_csv(TAB / "test_metrics_both_schemes.csv", index=False)
    best_on_test = min(BASE_MODELS, key=lambda m: brier(yc, pc[m]))
    print(f"\nПридатних зразків (консервативна): {mk.sum():,}, частка позитивів {yc.mean():.3f}")
    print(cons[["n", "base_rate", "brier", "log_loss", "pr_auc", "roc_auc", "auc_within_oblast", "bss_vs_B0_90d"]].round(4).to_string())
    print(f"Найкраща база за Brier на ТЕСТІ (лише інформація, на вердикт не впливає): {best_on_test}")

    # ---------- вердикт за правилом успіху ----------
    verdict = bss_row(yc, pc["LOGIT"], ref_p, tc)
    ok = verdict["bss"] > 0 and verdict["ci_low"] > 0
    print(f"\nВЕРДИКТ: BSS(LOGIT, проти {REF}) = {verdict['bss']:+.4f}, 95% ІН [{verdict['ci_low']:+.4f}; {verdict['ci_high']:+.4f}] "
          f"(блоки {BLOCK_DAYS} доби) -> модель {'ОБІГРАЛА базу' if ok else 'НЕ обіграла базу за правилом успіху'}")

    # ---------- BSS проти всіх баз з ІН (основний блок) ----------
    rows_bss = []
    for name in NEW:
        for ref in BASE_MODELS:
            rows_bss.append({"model": name, "vs": ref, **bss_row(yc, pc[name], pc[ref], tc)})
    bss_all = pd.DataFrame(rows_bss)
    bss_all.round(4).to_csv(TAB / "test_bss_vs_all_baselines.csv", index=False)
    print("\nBSS проти всіх баз (блоки 3 доби):\n", bss_all.round(4).to_string(index=False))

    # ---------- перевірки стійкості (описово) ----------
    rob = []
    for name in NEW:
        d = pd.Series((ref_p - yc) ** 2 - (pc[name] - yc) ** 2).groupby(tc.dt.floor("D")).sum()
        ref_loss = float(((ref_p - yc) ** 2).sum())
        point, se = d.sum() / ref_loss, newey_west_se(d.to_numpy(), 3) * len(d) / ref_loss
        rob.append({"model": name, "method": f"bootstrap_{BLOCK_DAYS}d", **bss_row(yc, pc[name], ref_p, tc, BLOCK_DAYS)})
        rob.append({"model": name, "method": f"bootstrap_{BLOCK_DAYS_ROBUST}d", **bss_row(yc, pc[name], ref_p, tc, BLOCK_DAYS_ROBUST)})
        rob.append({"model": name, "method": "newey_west_lag3", "bss": point, "ci_low": point - 1.96 * se, "ci_high": point + 1.96 * se})
    rob = pd.DataFrame(rob)
    rob.round(4).to_csv(TAB / "test_robustness_ci.csv", index=False)
    print("\nПеревірки стійкості ІН (описово):\n", rob.round(4).to_string(index=False))

    # ---------- схеми розмітки ----------
    diff = met.pivot(index="model", columns="scheme", values="brier").loc[models]
    diff["delta"] = diff.dataset - diff.conservative
    n = met.pivot(index="model", columns="scheme", values="n").loc[models]
    diff["n_cons"], diff["n_dataset"] = n.conservative, n.dataset
    diff.round(4).to_csv(TAB / "test_scheme_difference.csv")
    print("\nРізниця схем (Brier):\n", diff.round(4).to_string())

    # ---------- популяція придатних рядків ----------
    pre = (grid.t < REGIME_CHANGE.tz_convert("UTC")).to_numpy()
    pop = []
    for name, seg in (("весь тест", test_rows), ("до 4 вересня", test_rows & pre), ("після 4 вересня", test_rows & ~pre)):
        sub = grid[seg]
        pop.append({"segment": name, "rows": len(sub), "eligible_cons": int(sub.elig_cons.sum()), "eligible_ds": int(sub.elig_ds.sum()),
                    "share_cons": sub.elig_cons.mean(), "share_ds": sub.elig_ds.mean(),
                    "base_rate_cons": sub.z[sub.elig_cons].mean(), "base_rate_ds": sub.z[sub.elig_ds].mean()})
    ev_t = events[events.started_at >= TEST_START]
    pop = pd.DataFrame(pop)
    pop.round(4).to_csv(TAB / "test_population.csv", index=False)
    print(f"\nПопуляція придатних рядків:\n{pop.round(4).to_string(index=False)}\nnaive-тривог серед початків на тесті: {ev_t.naive.mean():.3f} ({int(ev_t.naive.sum())} з {len(ev_t)})")

    # ---------- по областях ----------
    sub = grid[mk].reset_index(drop=True)
    po = []
    for oblast, ix in sub.groupby("oblast", observed=True).indices.items():
        yy = yc[ix]
        rec = {"oblast": oblast, "n": len(ix), "positives": int(yy.sum()), "flag_lt30_positives": int(yy.sum()) < MIN_POS,
               **{f"brier_{m}": brier(yy, pc[m][ix]) for m in (REF, "LOGIT", "HGB")}}
        rec["bss_LOGIT_vs_B0_90d"] = 1 - rec["brier_LOGIT"] / rec[f"brier_{REF}"]
        rec["bss_HGB_vs_B0_90d"] = 1 - rec["brier_HGB"] / rec[f"brier_{REF}"]
        po.append(rec)
    po = pd.DataFrame(po).set_index("oblast")
    po.round(4).to_csv(TAB / "test_by_oblast.csv")
    print("\nПо областях (позначені flag_lt30_positives не інтерпретуються):\n", po.round(4).to_string())

    # ---------- по годині доби (Europe/Kyiv) ----------
    hk = sub.t.dt.tz_convert("Europe/Kyiv").dt.hour.to_numpy()
    ph = []
    for h in range(24):
        ix = np.flatnonzero(hk == h)
        yy = yc[ix]
        ph.append({"hour_kyiv": h, "n": len(ix), "base_rate": yy.mean(), **{f"brier_{m}": brier(yy, pc[m][ix]) for m in (REF, "LOGIT", "HGB")}})
    ph = pd.DataFrame(ph).set_index("hour_kyiv")
    ph["bss_LOGIT_vs_B0_90d"] = 1 - ph.brier_LOGIT / ph[f"brier_{REF}"]
    ph.round(4).to_csv(TAB / "test_by_hour.csv")
    print("\nПо годині доби (Kyiv):\n", ph.round(4).to_string())

    # ---------- до / після 4 вересня (описово) ----------
    seg_rows = []
    pre_c = (tc < REGIME_CHANGE.tz_convert("UTC")).to_numpy()
    for name, sel in (("до 4 вересня", pre_c), ("після 4 вересня", ~pre_c)):
        for m in ("LOGIT", "HGB"):
            seg_rows.append({"segment": name, "model": m, "n": int(sel.sum()), "base_rate": yc[sel].mean(),
                             "brier_B0_90d": brier(yc[sel], ref_p[sel]), "brier_model": brier(yc[sel], pc[m][sel]),
                             **bss_row(yc[sel], pc[m][sel], ref_p[sel], tc[sel])})
    seg = pd.DataFrame(seg_rows)
    seg.round(4).to_csv(TAB / "test_regime_split.csv", index=False)
    print("\nДо / після 4 вересня (описово, ІН за блоками 3 доби):\n", seg.round(4).to_string(index=False))

    # ---------- графіки ----------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ідеал")
    for m in (REF, "B3_ARIMA_cal", "LOGIT", "HGB"):
        rel = reliability(yc, pc[m], bins=12)
        ax.plot(rel[:, 0], rel[:, 1], marker="o", ms=3, label=m)
    ax.set_xlabel("прогнозована ймовірність"); ax.set_ylabel("фактична частка початків")
    ax.set_title("Калібрування на тесті (консервативна схема)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "14_test_calibration.png"); plt.close(fig)

    day = tc.dt.floor("D")
    daily = pd.DataFrame({"факт": yc, REF: ref_p, "LOGIT": pc["LOGIT"]}).groupby(day.values).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax, color=["k", "#1f77b4", "#d62728"], lw=1.3)
    ax.axvline(REGIME_CHANGE.tz_convert("UTC").tz_localize(None), color="gray", ls="--", lw=1)
    ax.text(REGIME_CHANGE.tz_convert("UTC").tz_localize(None), ax.get_ylim()[1] * 0.98, " 4 вересня", va="top", fontsize=8)
    ax.set_ylabel("середня ймовірність / частка початків за добу"); ax.set_xlabel("")
    ax.set_title("Прогноз проти факту на тесті, добові середні (усі придатні області)")
    fig.tight_layout(); fig.savefig(FIG / "15_test_forecast_vs_actual.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#bbbbbb" if f else "#1f77b4" for f in po.sort_values("bss_LOGIT_vs_B0_90d").flag_lt30_positives]
    po["bss_LOGIT_vs_B0_90d"].sort_values().plot.barh(ax=ax, color=colors)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("BSS проти B0_90d"); ax.set_ylabel("")
    ax.set_title("BSS основної моделі по областях, тест\n(сірі: менше 30 позитивів, не інтерпретуються)")
    fig.tight_layout(); fig.savefig(FIG / "16_test_bss_by_oblast.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ph["bss_LOGIT_vs_B0_90d"].plot.bar(ax=ax, color="#1f77b4")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("година доби (Europe/Kyiv)"); ax.set_ylabel("BSS проти B0_90d")
    ax.set_title("BSS основної моделі по годині доби, тест")
    fig.tight_layout(); fig.savefig(FIG / "17_test_bss_by_hour.png"); plt.close(fig)
    print(f"\nГотово за {time.time() - t_start:.0f} с")


if __name__ == "__main__":
    main()
