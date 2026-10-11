"""Опційний експеримент з текстовими даними: повідомлення каналу Повітряних сил (t.me/s/kpszsu), протокол «валідація, потім тест».

Протокол зафіксовано до запуску (docs/definition_of_done.md, reports/text.md):
  1. Текст є від 2026-03-27, тож навчання кожного вікна розширюване: від TEXT_FROM до (початок вікна - 4 год). Модель логістична,
     C з основної конфігурації, без підбору параметрів. Для кожного вікна пара «без тексту / з текстом» на однакових рядках.
  2. Валідація: вікна V1–V4 разом. Варіанти ознак: A (4 лічильники), B (A + типізовані згадки: дрон, ракета, балістика).
     Вибираємо варіант з більшим BSS(з текстом проти без тексту); текст «приймається», лише якщо нижня межа 95% ІН
     (блоковий бутстреп, 3 доби) цього BSS > 0.
  3. Тест: лише якщо текст прийнято, один запуск вибраного варіанта на всьому тесті (навчання до TEST_START - 4 год).
HGB рахується лише для інформації. Попередній 9-добовий запуск (до цього протоколу) описано в reports/text.md.
Запуск: cd src && ../.venv/bin/python run_text.py   (потрібні сторінки з download_telegram.py)
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import features as F
from airalerts import models as M
from airalerts import text as T
from airalerts.baseline_preds import baseline_predictions
from airalerts.config import BLOCK_DAYS, FIG, HORIZON_H, N_BOOT, PRIMARY, SECONDARY, TAB, TEST_START, VAL_LAST_T
from airalerts.data import load_events
from airalerts.metrics import auc_within_groups, block_bootstrap_bss, brier
from airalerts.splits import train_mask, val_windows
from airalerts.state import build_grid

TEXT_FROM = pd.Timestamp("2026-03-28", tz="UTC")      # повідомлення зібрано від 2026-03-27 00:00, тож вікна 1 і 3 год повні
REF, BEST_BASE = "B0_90d", "B3_ARIMA_cal"
SEC = SECONDARY[1]


def fit_window(kind, par, X, y, el, grid, w0, w1, cols, train_days=None):
    """Прогнози для вікна [w0, w1): навчання на придатних рядках від TEXT_FROM до w0 - 4 год (purge)."""
    train = np.flatnonzero(el & train_mask(grid, w0, train_days) & (grid.t >= TEXT_FROM).to_numpy())
    pred = np.flatnonzero(el & ((grid.t >= w0) & (grid.t < w1)).to_numpy())
    fit_par = {k: v for k, v in par.items() if k != "days"}
    p = M.fit_predict(kind, fit_par, X, y, train, pred, extra=cols)
    return pred, p, len(train)


def main():
    t0 = time.time()
    msgs = T.load_messages()
    assert msgs.time.min() <= pd.Timestamp("2026-03-27", tz="UTC") + pd.Timedelta(hours=1), "збережені сторінки не сягають 2026-03-27"
    events = load_events(cut_test=False)
    t_last = max(events.started_at.max(), events.finished_at.max()).floor("h") - pd.Timedelta(hours=HORIZON_H)
    grid = build_grid(events, t_last)
    y, el = grid.z.to_numpy(), grid.elig_cons.to_numpy()
    X = F.build_features(grid, "elig_cons", F.state_features(events, grid))
    tfB = T.text_features(grid, msgs, typed=True)
    X = pd.concat([X, tfB], axis=1)
    VARIANTS = {"A_counts": tuple(c for c in tfB.columns if c.startswith(("txt_ob_1", "txt_ob_3", "txt_all")) and "_drone_" not in c
                                   and "_missile_" not in c and "_ballistic_" not in c),
                "B_counts_typed": tuple(tfB.columns)}
    assert len(VARIANTS["A_counts"]) == 4 and len(VARIANTS["B_counts_typed"]) == 10

    # ---------- опис даних (описово: чи стабільний стиль каналу в часі) ----------
    men = T.mentions(msgs)
    low = msgs.text.str.lower()
    monthly = pd.DataFrame({"month": msgs.time.dt.strftime("%Y-%m"), "edited": msgs.edited,
                            "mentions_oblast": sum(men.values()) > 0,
                            "typed": low.str.contains("|".join(T.TYPES.values()))}).groupby("month").agg(
        messages=("edited", "size"), edited_share=("edited", "mean"), share_mentioning_oblast=("mentions_oblast", "mean"),
        share_with_threat_type=("typed", "mean"))
    monthly.round(3).to_csv(TAB / "text_monthly.csv")
    pd.DataFrame({"oblast": list(men), "messages_mentioning": [int(v.sum()) for v in men.values()]}).to_csv(TAB / "text_mentions_by_oblast.csv", index=False)
    print(f"повідомлень {len(msgs):,} ({msgs.time.min()} .. {msgs.time.max()}), відредагованих {msgs.edited.mean():.1%}")
    print("\nПо місяцях:\n", monthly.round(3).to_string())

    # ---------- валідація V1–V4 ----------
    wins = val_windows()
    par = {k: v for k, v in PRIMARY[1].items() if k != "days"}
    pools = {name: ([], []) for name in ["none", *VARIANTS]}
    rows_win, idx_all = [], []
    for k, (w0, w1) in enumerate(wins, 1):
        for name, cols in [("none", ())] + list(VARIANTS.items()):
            pred, p, n_train = fit_window("logit", PRIMARY[1], X, y, el, grid, w0, w1, cols)
            pools[name][0].append(pred); pools[name][1].append(p)
            rows_win.append({"window": f"V{k}", "model": name, "n_train": n_train, "n": len(pred), "brier": brier(y[pred].astype(float), p)})
        print(f"V{k}: готово", flush=True)
    idx = np.concatenate(pools["none"][0])
    for name in VARIANTS:
        assert np.array_equal(idx, np.concatenate(pools[name][0]))
    P = {n: np.concatenate(v[1]) for n, v in pools.items()}
    yy, tt = y[idx].astype(float), grid.t.iloc[idx].reset_index(drop=True)
    ob = grid.oblast.astype(str).iloc[idx].to_numpy()
    win_tab = pd.DataFrame(rows_win)
    win_tab.round(5).to_csv(TAB / "text_val_windows.csv", index=False)
    print("\nBrier по вікнах:\n", win_tab.pivot(index="window", columns="model", values="brier").round(5).to_string())

    val_rows = []
    for name in VARIANTS:
        b, lo, hi = block_bootstrap_bss(yy, P[name], P["none"], tt, block_days=BLOCK_DAYS, n_boot=N_BOOT, seed=0)
        wins_better = int(sum(win_tab[(win_tab.model == name) & (win_tab.window == f"V{k}")].brier.iloc[0]
                              < win_tab[(win_tab.model == "none") & (win_tab.window == f"V{k}")].brier.iloc[0] for k in range(1, 5)))
        val_rows.append({"variant": name, "n": len(yy), "brier_text": brier(yy, P[name]), "brier_none": brier(yy, P["none"]),
                         "auc_ob_text": auc_within_groups(yy, P[name], ob), "auc_ob_none": auc_within_groups(yy, P["none"], ob),
                         "bss_vs_none": b, "ci_low": lo, "ci_high": hi, "windows_better": wins_better})
    val = pd.DataFrame(val_rows)
    val.round(5).to_csv(TAB / "text_val_bss.csv", index=False)
    print("\nВалідація, BSS з текстом проти без тексту (3 доби):\n", val.round(5).to_string(index=False))

    best = val.sort_values("bss_vs_none").iloc[-1]
    accepted = bool(best.ci_low > 0)
    print(f"\nКращий варіант: {best.variant}; правило (нижня межа ІН > 0): {'ПРИЙНЯТО' if accepted else 'НЕ ПРИЙНЯТО'}")

    # HGB лише для інформації (той самий вибраний варіант і всі варіанти), на валідації
    info = []
    for name, cols in [("none", ())] + list(VARIANTS.items()):
        ps = [fit_window("hgb", SECONDARY[1], X, y, el, grid, w0, w1, cols)[1] for (w0, w1) in wins]
        info.append((name, np.concatenate(ps)))
    hg = dict(info)
    hgb_rows = []
    for name in VARIANTS:
        b, lo, hi = block_bootstrap_bss(yy, hg[name], hg["none"], tt, block_days=BLOCK_DAYS, n_boot=N_BOOT, seed=0)
        hgb_rows.append({"variant": name, "bss_vs_none": b, "ci_low": lo, "ci_high": hi, "brier_text": brier(yy, hg[name]), "brier_none": brier(yy, hg["none"])})
    hgb = pd.DataFrame(hgb_rows)
    hgb.round(5).to_csv(TAB / "text_val_hgb_info.csv", index=False)
    print("\nHGB (лише для інформації):\n", hgb.round(5).to_string(index=False))

    # ---------- тест: один запуск, лише якщо текст прийнято ----------
    test = None
    if accepted:
        name = best.variant
        w0, w1 = TEST_START, t_last + pd.Timedelta(hours=1)
        pred, p_txt, n_train = fit_window("logit", PRIMARY[1], X, y, el, grid, w0, w1, VARIANTS[name])
        _, p_none, _ = fit_window("logit", PRIMARY[1], X, y, el, grid, w0, w1, ())
        train_main = np.flatnonzero(el & train_mask(grid, w0, PRIMARY[1]["days"]))
        p_main = M.fit_predict("logit", par, X, y, train_main, pred)
        base = baseline_predictions(events, grid, [(w0, w1)])["conservative"]
        p_b0, p_arima = base[REF][pred], base[BEST_BASE][pred]
        yt, ttt = y[pred].astype(float), grid.t.iloc[pred].reset_index(drop=True)
        assert abs(brier(yt, p_main) - 0.1588) < 5e-4, "основна модель має відтворити Brier з run_test.log"
        obt = grid.oblast.astype(str).iloc[pred].to_numpy()
        pt = {"LOGIT_no_text": p_none, f"LOGIT_text_{name}": p_txt, "LOGIT_main_180d": p_main, REF: p_b0, BEST_BASE: p_arima}
        test_m = pd.DataFrame([{"model": m, "n": len(yt), "train_rows": n_train if m.startswith("LOGIT_") and m != "LOGIT_main_180d" else np.nan,
                                "brier": brier(yt, p), "auc_within_oblast": auc_within_groups(yt, p, obt)} for m, p in pt.items()])
        test_m.round(5).to_csv(TAB / "text_test_metrics.csv", index=False)
        crows = []
        for m, ref in ((f"LOGIT_text_{name}", "LOGIT_no_text"), (f"LOGIT_text_{name}", REF), (f"LOGIT_text_{name}", "LOGIT_main_180d"),
                        (f"LOGIT_text_{name}", BEST_BASE), ("LOGIT_main_180d", BEST_BASE)):   # дві останні пари додано після першого запуску, не з протоколу
            b, lo, hi = block_bootstrap_bss(yt, pt[m], pt[ref], ttt, block_days=BLOCK_DAYS, n_boot=N_BOOT, seed=0)
            crows.append({"model": m, "vs": ref, "bss": b, "ci_low": lo, "ci_high": hi})
        test = pd.DataFrame(crows)
        test.round(5).to_csv(TAB / "text_test_bss.csv", index=False)
        print("\nТЕСТ (один запуск):\n", test_m.round(5).to_string(index=False), "\n", test.round(5).to_string(index=False))

    # ---------- графік ----------
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    items = [(f"validation\n{r.variant.split('_')[0]}: with vs without text", r.bss_vs_none, r.ci_low, r.ci_high) for r in val.itertuples()]
    if test is not None:
        r = test.iloc[0]
        items.append(("test\nwith vs without text", r.bss, r.ci_low, r.ci_high))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.errorbar(range(len(items)), [i[1] for i in items], yerr=[[i[1] - i[2] for i in items], [i[3] - i[1] for i in items]], fmt="o", capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=8)
    ax.set_ylabel("Skill score (BSS)"); ax.set_title("Text features (Air Force channel): validation, then test; 95% CI, 3-day blocks")
    fig.tight_layout(); fig.savefig(FIG / "20_text_features.png"); plt.close(fig)
    print(f"\nГотово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
