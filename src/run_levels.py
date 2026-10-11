"""Етап 2 (O3): шар рівня P(червона | початок тривоги в області) на даних alerts.in.ua з рівнями (з 2026-09-05).

Протокол (rolling, без вердикту за правилом успіху, бо вибірка закоротка):
  розігрів 2026-09-05..09-19 (лише історія), налаштування 09-19..10-01, оцінка 10-01..10-10 (останній період).
Ознаки епізоду i: лише епізоди j, рівень яких відомий на його початку (start_j + W <= start_i).
Запуск: cd src && ../.venv/bin/python run_levels.py
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import levels as L
from airalerts.config import FIG, TAB
from airalerts.data import load_events
from airalerts.metrics import block_bootstrap_bss, brier, reliability

W_MIN = 60                                   # вікно рівня епізоду, хв (обрано за таблицею охоплення, див. нижче)
TUNE_FROM, EVAL_FROM = pd.Timestamp("2026-09-19", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
GRID = list(itertools.product((3, 7, 14, None), (5, 20, 80), (None, 20, 80)))   # win_days, m_o, m_h


def name(win, m_o, m_h):
    return f"win={'all' if win is None else win}d,m_o={m_o},m_h={m_h}"


def main():
    arch = L.load_archive()
    print(f"архів: {len(arch)} записів повітряних тривог з рівнями, {arch.started_at.min()} .. {arch.started_at.max()}")

    full = L.episodes(arch, 10 ** 6)
    cap = []
    for w in (0, 5, 15, 30, 60, 120):
        e = L.episodes(arch, w)
        cap.append({"window_min": w, "episodes": len(e), "red_share": e.red.mean(), "share_of_red_any": e.red.sum() / full.red.sum()})
    cap = pd.DataFrame(cap)
    cap.round(4).to_csv(TAB / "levels_window_capture.csv", index=False)
    print("\nОхоплення червоних епізодів залежно від вікна W:\n", cap.round(3).to_string(index=False))

    ep = L.episodes(arch, W_MIN)
    wk = ep.groupby(ep.started_at.dt.strftime("%G-W%V")).agg(episodes=("red", "size"), red_share=("red", "mean"))
    print("\nЧастка червоних серед початків по тижнях (нестаціонарність):\n", wk.round(3).to_string())

    # перевірка припущення «ряд T1 і епізоди архіву збігаються»
    v = load_events(cut_test=False)
    v = v[v.started_at >= L.LEVELS_FROM]
    rows = []
    for tol in (10, 30, 60, 180):
        a_m = v_m = 0
        for ob, g in ep.groupby("oblast"):
            vv = v[v.oblast == ob].started_at.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[s]")
            gg = g.started_at.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[s]")
            if len(vv) == 0:
                continue
            a_m += (np.abs(gg[:, None] - vv[None, :]).min(1).astype(float) / 60 <= tol).sum()
            v_m += (np.abs(vv[:, None] - gg[None, :]).min(1).astype(float) / 60 <= tol).sum()
        rows.append({"tolerance_min": tol, "archive_episodes": len(ep), "volunteer_starts": len(v),
                     "episodes_with_volunteer_start": a_m / len(ep), "volunteer_starts_with_episode": v_m / len(v)})
    cross = pd.DataFrame(rows)
    cross.round(4).to_csv(TAB / "levels_cross_source_match.csv", index=False)
    print("\nЗбіг початків архіву (епізоди) і волонтерського ряду T1:\n", cross.round(3).to_string(index=False))

    by_ob = pd.DataFrame({"archive_episodes": ep.groupby("oblast").size(), "volunteer_starts": v.groupby("oblast").size()}).fillna(0).astype(int)
    by_ob["ratio_volunteer_to_archive"] = (by_ob.volunteer_starts / by_ob.archive_episodes).round(2)
    by_ob.sort_values("ratio_volunteer_to_archive").to_csv(TAB / "levels_vs_volunteer_by_oblast.csv")
    print("\nКількість початків за областями (з 2026-09-05): архів (епізоди) і волонтерський ряд T1:\n", by_ob.sort_values("ratio_volunteer_to_archive").to_string())

    # налаштування на 09-19..10-01
    preds = {name(*c): L.predict(ep, W_MIN, *c) for c in GRID}
    tune = ((ep.started_at >= TUNE_FROM) & (ep.started_at < EVAL_FROM)).to_numpy()
    ev = (ep.started_at >= EVAL_FROM).to_numpy()
    y = ep.red.to_numpy(float)
    ref_all = L.predict(ep, W_MIN, None, 1e9)                    # B_exp: частка червоних за всією відомою історією
    ref_w7 = L.predict(ep, W_MIN, 7, 1e9)                        # B_7d: частка червоних за останні 7 діб
    cfg = pd.DataFrame([{"config": k, "brier_tune": brier(y[tune], p[tune]), "brier_eval": brier(y[ev], p[ev])} for k, p in preds.items()])
    cfg = cfg.sort_values("brier_tune").reset_index(drop=True)
    cfg.round(5).to_csv(TAB / "levels_configs.csv", index=False)
    best = cfg.config[0]
    print(f"\nНалаштування (Brier на 09-19..10-01), 5 найкращих:\n", cfg.head(5).round(4).to_string(index=False))
    print(f"Обрана конфігурація: {best}")

    p_best = preds[best]
    cand = {"B_exp": ref_all, "B_7d": ref_w7, "MODEL": p_best}
    t_ev, y_ev = ep.started_at[ev], y[ev]
    res = []
    for k, p in cand.items():
        r = {"model": k, "n": int(ev.sum()), "red_share": y_ev.mean(), "brier": brier(y_ev, p[ev]),
             "log_loss": float(-np.mean(y_ev * np.log(np.clip(p[ev], 1e-6, 1)) + (1 - y_ev) * np.log(np.clip(1 - p[ev], 1e-6, 1))))}
        for ref in ("B_exp", "B_7d"):
            b, lo, hi = block_bootstrap_bss(y_ev, p[ev], cand[ref][ev], t_ev, block_days=1, n_boot=1000)
            r[f"bss_vs_{ref}"], r[f"lo_{ref}"], r[f"hi_{ref}"] = b, lo, hi
        res.append(r)
    res = pd.DataFrame(res)
    res.round(5).to_csv(TAB / "levels_eval.csv", index=False)
    print("\nОцінка на 10-01..10-10 (ІН: блоки по 1 добі, ~10 блоків, грубо):\n", res.round(4).to_string(index=False))

    # по областях і годинних блоках (описово)
    d = pd.DataFrame({"oblast": ep.oblast[ev], "bucket": ep.bucket[ev], "y": y_ev, "p": p_best[ev], "p_exp": ref_all[ev]})
    for col in ("oblast", "bucket"):
        g = d.groupby(col).apply(lambda x: pd.Series({"n": len(x), "red_share": x.y.mean(), "mean_pred": x.p.mean(),
                                                      "brier_model": brier(x.y, x.p), "brier_B_exp": brier(x.y, x.p_exp)}))
        g.round(4).to_csv(TAB / f"levels_by_{col}.csv")
    print("\nПо годинних блоках (за Києвом, блок = години/6):\n", pd.read_csv(TAB / "levels_by_bucket.csv").to_string(index=False))

    # графіки: тижнева динаміка і калібрування
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    day = ep.started_at.dt.floor("D").dt.tz_localize(None)
    m = (ep.started_at >= TUNE_FROM).to_numpy()
    daily = pd.DataFrame({"day": day[m], "y": y[m], "model": p_best[m], "B_exp": ref_all[m], "B_7d": ref_w7[m]}).groupby("day").mean()
    ax[0].plot(daily.index, daily.y, "k-o", ms=3, label="actual")
    for k, c in (("model", "C0"), ("B_exp", "C3"), ("B_7d", "C2")):
        ax[0].plot(daily.index, daily[k], c, label={"model": "level layer", "B_exp": "B_exp (all history)", "B_7d": "B_7d (last 7 days)"}[k])
    ax[0].axvline(EVAL_FROM.tz_localize(None), color="gray", ls="--", lw=0.8)
    ax[0].set_ylabel("share of red among alert starts"); ax[0].set_title("Daily share of red alerts: actual and forecast"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=30)
    for k, c in (("MODEL", "C0"), ("B_exp", "C3")):
        r = reliability(y_ev, cand[k][ev], bins=6)
        ax[1].plot(r[:, 0], r[:, 1], "-o", color=c, label="level layer" if k == "MODEL" else "B_exp (all history)")
    ax[1].plot([0, 0.6], [0, 0.6], "k--", lw=0.8); ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("actual frequency"); ax[1].set_title("Calibration, 1-10 October"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "18_levels.png"); plt.close(fig)


if __name__ == "__main__":
    main()
