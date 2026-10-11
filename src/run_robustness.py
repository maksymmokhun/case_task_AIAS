"""Перевірки стійкості висновку тесту (описові, на вердикт не впливають; зафіксований вердикт у run_test.py).

A. Незалежні мітки: ті самі прогнози (одне перенавчання, як у run_test.py), але ціль береться з архіву alerts.in.ua:
   «в області починається тривога (епізод) у (t, t+3 год]». Період 2026-09-06 .. кінець сітки, бо архів ведеться з 2026-09-05 21:09 UTC.
   Рядки: придатні за консервативною схемою волонтерського ряду й без активної тривоги в архіві. Ознаки лишаються з волонтерського ряду.
B. Без областей, де кількість початків волонтерського ряду сильно розходиться з епізодами архіву (відношення поза [0,5; 1,5])
   і Закарпатської (0 початків у ряді). Критерій задано до перегляду результатів; перелік з reports/tables/levels_vs_volunteer_by_oblast.csv.
Запуск: cd src && ../.venv/bin/python run_robustness.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import features as F
from airalerts import levels as L
from airalerts.baseline_preds import MODELS as BASE_MODELS
from airalerts.baseline_preds import baseline_predictions
from airalerts.config import BLOCK_DAYS, FIG, HORIZON_H, N_BOOT, PRIMARY, SECONDARY, TAB, TEST_START
from airalerts.data import load_events
from airalerts.metrics import auc_within_groups, block_bootstrap_bss, brier
from airalerts.state import build_grid, to_ns
from run_model import run_config

NEW = {"LOGIT": PRIMARY, "HGB": SECONDARY}
REF, BEST_TEST_BASE = "B0_90d", "B3_ARIMA_cal"
ARCHIVE_FROM = pd.Timestamp("2026-09-06", tz="UTC")
RATIO_LO, RATIO_HI = 0.5, 1.5


def archive_arrays(arch, ep, grid):
    """Для кожного рядка сітки: активна тривога в архіві в t і початок епізоду архіву в (t, t+H]."""
    step = np.timedelta64(HORIZON_H, "h")
    active, z = np.zeros(len(grid), bool), np.zeros(len(grid), bool)
    ob = grid.oblast.astype(str).to_numpy()
    tt_all = grid.t.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
    for oblast in np.unique(ob):
        ix = np.flatnonzero(ob == oblast)
        tt = tt_all[ix]
        g = arch[arch.oblast == oblast]
        if len(g):
            s = to_ns(g.started_at)
            cm = np.maximum.accumulate(to_ns(g.finished_filled))   # кінець найпізнішого з уже розпочатих записів
            i = np.searchsorted(s, tt, "right") - 1
            active[ix] = (i >= 0) & (cm[np.clip(i, 0, None)] > tt)
            es = np.sort(to_ns(ep[ep.oblast == oblast].started_at))
            z[ix] = (np.searchsorted(es, tt + step, "right") - np.searchsorted(es, tt, "right")) > 0
    return active, z


def metrics_row(label, y, P, t, oblast, models):
    out = []
    for m in models:
        out.append({"labels": label, "model": m, "n": len(y), "base_rate": y.mean(), "brier": brier(y, P[m]),
                    "auc_within_oblast": auc_within_groups(y, P[m], oblast),
                    **{f"bss_vs_{ref}": 1 - brier(y, P[m]) / brier(y, P[ref]) for ref in (REF, BEST_TEST_BASE)}})
    return out


def ci(y, p, ref, t):
    b, lo, hi = block_bootstrap_bss(y, p, ref, t, block_days=BLOCK_DAYS, n_boot=N_BOOT, seed=0)
    return b, lo, hi


def main():
    t0 = time.time()
    events = load_events(cut_test=False)
    t_last = max(events.started_at.max(), events.finished_at.max()).floor("h") - pd.Timedelta(hours=HORIZON_H)
    grid = build_grid(events, t_last)
    y = grid.z.to_numpy()
    t_end = t_last + pd.Timedelta(hours=1)
    window = (TEST_START, t_end)
    test_rows = ((grid.t >= TEST_START) & (grid.t < t_end)).to_numpy()

    state = F.state_features(events, grid)
    X = F.build_features(grid, "elig_cons", state)
    base = baseline_predictions(events, grid, [window])["conservative"]
    el = grid.elig_cons.to_numpy()
    P = {m: base[m] for m in BASE_MODELS}
    for name, (kind, par) in NEW.items():
        rows, p = run_config(kind, par, X, y, grid, el, {1: window})[1]
        full = np.full(len(grid), np.nan)
        full[rows] = p
        P[name] = full
    models = BASE_MODELS + list(NEW)

    mk = el & test_rows
    chk = {m: brier(y[mk].astype(float), P[m][mk]) for m in (REF, "LOGIT", "HGB")}
    print(f"контроль відтворення прогнозів: Brier {chk} (у run_test.log: B0_90d 0.1743, LOGIT 0.1588, HGB 0.1572)")
    assert abs(chk[REF] - 0.1743) < 5e-4 and abs(chk["LOGIT"] - 0.1588) < 5e-4 and abs(chk["HGB"] - 0.1572) < 5e-4

    # ---------- B: без областей з розбіжним змістом «початку» ----------
    ratio = pd.read_csv(TAB / "levels_vs_volunteer_by_oblast.csv").set_index("oblast").ratio_volunteer_to_archive
    dropped = sorted(ratio.index[(ratio < RATIO_LO) | (ratio > RATIO_HI)])        # Закарпатська: відношення 0 -> потрапляє сюди
    print(f"\nB. виключені області ({len(dropped)}): {dropped}")
    ob = grid.oblast.astype(str)
    keep = ~ob.isin(dropped).to_numpy()
    rows_b = []
    for label, sel in (("усі області", mk), (f"без {len(dropped)} областей", mk & keep)):
        yy, tt, pp = y[sel].astype(float), grid.t[sel].reset_index(drop=True), {m: P[m][sel] for m in models}
        for name in NEW:
            for ref in (REF, BEST_TEST_BASE):
                b, lo, hi = ci(yy, pp[name], pp[ref], tt)
                rows_b.append({"subset": label, "n": len(yy), "positives": int(yy.sum()), "model": name, "vs": ref,
                               "bss": b, "ci_low": lo, "ci_high": hi})
    rb = pd.DataFrame(rows_b)
    rb.round(4).to_csv(TAB / "robust_exclusion.csv", index=False)
    print(rb.round(4).to_string(index=False))

    # ---------- A: незалежні мітки з архіву ----------
    arch = L.load_archive()
    ep = L.episodes(arch, 60)
    act_a, z_a = archive_arrays(arch, ep, grid)
    per = (grid.t >= ARCHIVE_FROM).to_numpy() & test_rows
    base_rows = per & el                                                          # придатні за волонтерським рядом у період архіву
    both = base_rows & ~act_a
    print(f"\nA. період {ARCHIVE_FROM} .. {t_last}: придатних за волонтерським рядом {base_rows.sum():,}, "
          f"з них з активною тривогою в архіві {int((base_rows & act_a).sum()):,} ({(base_rows & act_a).sum() / base_rows.sum():.1%}); "
          f"спільно придатних {both.sum():,}")

    ya, yv = z_a[both].astype(float), y[both].astype(float)
    agree = pd.crosstab(pd.Series(yv, name="мітка волонтерського ряду"), pd.Series(ya, name="мітка архіву"))
    agree.to_csv(TAB / "robust_label_agreement.csv")
    print("Збіг міток на спільних рядках (рядки: ряд T1, стовпці: архів):\n", agree.to_string())
    print(f"частка позитивів: ряд T1 {yv.mean():.3f}, архів {ya.mean():.3f}; збіг міток {np.mean(ya == yv):.3f}")

    tt = grid.t[both].reset_index(drop=True)
    ob_b = ob[both].reset_index(drop=True)
    pb = {m: P[m][both] for m in models}
    recs = metrics_row("волонтерський ряд (ті самі рядки)", yv, pb, tt, ob_b, models) + \
           metrics_row("архів alerts.in.ua (незалежні)", ya, pb, tt, ob_b, models)
    res = pd.DataFrame(recs)
    ci_rows = []
    for label, yy in (("волонтерський ряд (ті самі рядки)", yv), ("архів alerts.in.ua (незалежні)", ya)):
        for name in NEW:
            for ref in (REF, BEST_TEST_BASE):
                b, lo, hi = ci(yy, pb[name], pb[ref], tt)
                ci_rows.append({"labels": label, "model": name, "vs": ref, "bss": b, "ci_low": lo, "ci_high": hi})
    cis = pd.DataFrame(ci_rows)
    res.round(4).to_csv(TAB / "robust_independent_labels.csv", index=False)
    cis.round(4).to_csv(TAB / "robust_independent_labels_ci.csv", index=False)
    print("\nМетрики на спільних рядках:\n", res.round(4).to_string(index=False))
    print("\nBSS з ІН (блоки 3 доби, ~11 блоків, грубо):\n", cis.round(4).to_string(index=False))

    # ---------- графік ----------
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.4))
    items = []
    for _, r in rb[(rb.model == "LOGIT")].iterrows():
        subset = r.subset.replace("усі області", "all\noblasts").replace("без 7 областей", "without 7\noblasts")   # у таблицях підписи українською
        items.append((f"{subset}\nvs {r.vs}", r.bss, r.ci_low, r.ci_high))
    for _, r in cis[(cis.model == "LOGIT")].iterrows():
        items.append((f"{'archive' if 'архів' in r.labels else 'volunteer'}\nlabels\nvs {r.vs}", r.bss, r.ci_low, r.ci_high))
    xs = np.arange(len(items))
    ax.errorbar(xs, [i[1] for i in items], yerr=[[i[1] - i[2] for i in items], [i[3] - i[1] for i in items]], fmt="o", capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([i[0] for i in items], fontsize=7)
    ax.set_ylabel("Skill score (BSS) of logistic regression"); ax.set_title("Robustness of the skill score: oblasts and label source (95% CI, 3-day blocks)")
    fig.tight_layout(); fig.savefig(FIG / "19_robustness.png"); plt.close(fig)
    print(f"\nГотово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
