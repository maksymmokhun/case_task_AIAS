"""Ітерація 2: аудит даних. Читає data/raw/*.csv, пише reports/tables/*.csv і reports/figures/*.png.

Ознаки та моделі тут не будуються, скрипт лише описує джерела.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, TAB, FIG = ROOT / "data/raw", ROOT / "reports/tables", ROOT / "reports/figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
H = 3  # горизонт прогнозу, години (docs/problem_statement.md)


def load():
    off = pd.read_csv(RAW / "official_data_en.csv", parse_dates=["started_at", "finished_at"]).drop_duplicates()
    vol = pd.read_csv(RAW / "volunteer_data_en.csv", parse_dates=["started_at", "finished_at"]).rename(columns={"region": "oblast"})
    return off, vol


def hours(df):
    return (df.finished_at - df.started_at).dt.total_seconds() / 3600


def match_rate(off_oblast, vol, lo, hi, tol_s=300):
    """Частка офіційних початків рівня області, для яких є волонтерський початок у межах tol_s секунд."""
    a = off_oblast[(off_oblast.started_at >= lo) & (off_oblast.started_at <= hi)]
    b = vol[(vol.started_at >= lo) & (vol.started_at <= hi)]
    rows = []
    for k, ga in a.groupby("oblast"):
        A = ga.started_at.sort_values().values
        B = b[b.oblast == k].started_at.sort_values().values
        i = np.searchsorted(B, A)
        left = np.where(i > 0, np.abs(A - B[np.clip(i - 1, 0, len(B) - 1)]), np.timedelta64(10**9, "s"))
        right = np.where(i < len(B), np.abs(B[np.clip(i, 0, len(B) - 1)] - A), np.timedelta64(10**9, "s"))
        d = np.minimum(left, right) / np.timedelta64(1, "s")
        rows.append((k, len(A), int((b.oblast == k).sum()), float((d <= tol_s).mean())))
    return pd.DataFrame(rows, columns=["oblast", "official_n", "volunteer_n", "match_5min"])


def eligible_grid(vol, horizon=H):
    """Погодинна сітка (область, t) із ціллю з постановки; тільки моменти без активної тривоги.

    Правий край цензурований: останні `horizon` годин відкинуто, бо їхня ціль ще невідома.
    Стан \"активна\" тут за розміткою датасету (start <= t < end).
    """
    t_end = max(vol.finished_at.max(), vol.started_at.max()).floor("h") - pd.Timedelta(hours=horizon)
    grid = pd.date_range(vol.started_at.min().ceil("h"), t_end, freq="h")
    out = []
    for k, g in vol.groupby("oblast"):
        g = g.sort_values("started_at")
        s, e = g.started_at.values, g.finished_at.values
        tt = grid.values
        i = np.searchsorted(s, tt, "right") - 1
        active = (i >= 0) & (e[np.clip(i, 0, None)] > tt)
        n_next = np.searchsorted(s, tt + np.timedelta64(horizon, "h"), "right") - np.searchsorted(s, tt, "right")
        df = pd.DataFrame({"t": grid, "oblast": k, "active": active, "y": n_next > 0})
        out.append(df[~df.active])
    return pd.concat(out, ignore_index=True)


def hindsight_share(vol, grid):
    """Позначає придатні зразки, що лежать після синтетичного (naive) кінця й до наступного початку в області.

    Датасет вважає ці моменти спокійними, а жива система бачила б тривогу активною (відбою ще не було).
    """
    v = vol.sort_values(["oblast", "started_at"]).copy()
    v["next_start"] = v.groupby("oblast").started_at.shift(-1)
    n = v[v.naive].dropna(subset=["next_start"])
    mark = np.zeros(len(grid), bool)
    for k, gr in n.groupby("oblast"):
        idx = np.where(grid.oblast.values == k)[0]
        tt = grid.t.values[idx]
        for f, ns in zip(gr.finished_at.values, gr.next_start.values):
            mark[idx[(tt > f) & (tt < ns)]] = True
    return grid.assign(hindsight=mark)


def main():
    off, vol = load()
    off["level"] = off["level"].astype("string")
    t = {}

    # 1. структура
    t["structure"] = pd.DataFrame({
        "official": [len(off), off.started_at.min(), off.started_at.max(), off.level.value_counts().to_dict()],
        "volunteer": [len(vol), vol.started_at.min(), vol.started_at.max(), {"naive_share": round(vol.naive.mean(), 4)}],
    }, index=["rows_after_dedup", "first_start", "last_start", "extra"])
    t["structure"].to_csv(TAB / "structure.csv")

    # 2. рівень адмін. одиниці в офіційних даних по місяцях
    lv = off.pivot_table(index=off.started_at.dt.strftime("%Y-%m"), columns="level", values="oblast", aggfunc="size", fill_value=0)
    lv.to_csv(TAB / "official_level_by_month.csv")

    # 3. місячні початки: волонтерські проти офіційних (лише level=oblast)
    m_v = vol.groupby(vol.started_at.dt.strftime("%Y-%m")).size().rename("volunteer")
    m_o = off[off.level == "oblast"].groupby(off.started_at.dt.strftime("%Y-%m")).size().rename("official_oblast_level")
    monthly = pd.concat([m_v, m_o], axis=1).fillna(0).astype(int)
    monthly.to_csv(TAB / "monthly_onsets.csv")

    # 4. збіг джерел там, де офіційні дані ще обласні
    lo, hi = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2025-06-30", tz="UTC")
    mr = match_rate(off[off.level == "oblast"], vol, lo, hi).sort_values("match_5min")
    mr.to_csv(TAB / "cross_source_match.csv", index=False)
    print("overall match (5 min):", round(np.average(mr.match_5min, weights=mr.official_n), 3))

    # 5. тривалості
    dur = hours(vol)
    t["duration"] = pd.DataFrame({"volunteer_h": dur.describe(percentiles=[.5, .9, .99, .999]),
                                  "official_h": hours(off).describe(percentiles=[.5, .9, .99, .999])})
    t["duration"].to_csv(TAB / "duration_summary.csv")

    # 6. добові початки та повнота
    days = pd.date_range(vol.started_at.min().floor("D"), vol.started_at.max().floor("D"), freq="D")
    daily = vol.groupby(vol.started_at.dt.floor("D")).size().reindex(days, fill_value=0)
    print("days:", len(daily), "zero-onset days:", int((daily == 0).sum()), "days<10:", int((daily < 10).sum()))

    # 7. статистика цілі на придатній погодинній сітці
    vol = vol[vol.oblast != "Luhanska oblast"]  # постійну тривогу вилучено з датасету, тож стан хибний
    g = hindsight_share(vol, eligible_grid(vol))
    g["q"] = g.t.dt.tz_localize(None).dt.to_period("Q").astype(str)
    q = g.groupby("q").agg(eligible=("y", "size"), positives=("y", "sum"), rate=("y", "mean")).round(4)
    q.to_csv(TAB / "target_by_quarter.csv")
    po = g.groupby("oblast").agg(eligible=("y", "size"), positives=("y", "sum"), rate=("y", "mean")).round(4)
    po.to_csv(TAB / "target_by_oblast_all.csv")
    t_max = g.t.max()
    hs = []
    for name, gw in (("all", g), ("last_8w", g[g.t > t_max - pd.Timedelta(weeks=8)])):
        hs.append({"period": name, "eligible": len(gw), "hindsight_share": round(gw.hindsight.mean(), 4),
                   "rate_hindsight": round(gw[gw.hindsight].y.mean(), 3), "rate_other": round(gw[~gw.hindsight].y.mean(), 3)})
    pd.DataFrame(hs).to_csv(TAB / "hindsight_naive_ends.csv", index=False); print(pd.DataFrame(hs).to_string(index=False))
    rows = []
    for w in (4, 8, 12, 26):
        gw = g[g.t > t_max - pd.Timedelta(weeks=w)]
        pw = gw.groupby("oblast").y.agg(["size", "sum"])
        rows.append({"weeks": w, "eligible": len(gw), "positives": int(gw.y.sum()), "rate": round(gw.y.mean(), 4),
                     "oblasts_with_<30_pos": int((pw["sum"] < 30).sum()), "min_pos": int(pw["sum"].min())})
    win = pd.DataFrame(rows); win.to_csv(TAB / "target_by_test_window.csv", index=False)
    print(win.to_string(index=False)); print("grid ends:", t_max)

    # ---------- figures ----------
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    lv.plot.area(ax=ax[0], linewidth=0, color=["#8da0cb", "#66c2a5", "#fc8d62"]); ax[0].set_title("Офіційний датасет: рядки за рівнем адмін. одиниці")
    ax[0].set_ylabel("рядків / міс.")
    monthly.plot(ax=ax[1], color=["#1f77b4", "#d62728"]); ax[1].set_title("Початки тривог на рівні області: волонтерські vs офіційні (level=oblast)")
    ax[1].set_ylabel("початків / міс."); ax[1].set_xlabel("")
    fig.tight_layout(); fig.savefig(FIG / "01_sources_granularity.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    daily.rolling(7).mean().plot(ax=ax, color="#1f77b4", label="7 днів"); daily.rolling(30).mean().plot(ax=ax, color="#d62728", label="30 днів")
    ax.set_title("Початки тривог по країні за добу (волонтерські), ковзне середнє"); ax.set_ylabel("початків / добу"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "02_daily_onsets_nonstationarity.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    v = vol.assign(h=vol.started_at.dt.tz_convert("Europe/Kyiv").dt.hour, y=vol.started_at.dt.tz_convert("Europe/Kyiv").dt.year)
    prof = v.groupby(["y", "h"]).size().unstack(0); (prof / prof.sum()).plot(ax=ax, marker="o", ms=3)
    ax.set_title("Частка початків за годиною доби (Europe/Kyiv) по роках"); ax.set_xlabel("година"); ax.set_ylabel("частка")
    fig.tight_layout(); fig.savefig(FIG / "03_hour_profile_by_year.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    po.sort_values("rate").rate.plot.barh(ax=ax[0], color="#1f77b4"); ax[0].set_title("Частка позитивів цілі (3 год),\nвся історія", fontsize=10)
    mr.set_index("oblast").match_5min.plot.barh(ax=ax[1], color="#2ca02c"); ax[1].set_title("Офіц. початки, що збігаються з волонтерськими ±5 хв\n(level=oblast, 2023-01…2025-06)", fontsize=10)
    ax[1].set_ylabel(""); ax[0].set_ylabel("")
    fig.tight_layout(); fig.savefig(FIG / "04_target_rate_and_source_agreement.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    q.rate.plot(ax=ax, marker="o", color="#d62728"); ax.set_title("Частка позитивів цілі по кварталах (усі області, погодинна сітка, без активної тривоги)")
    ax.set_ylabel("P(y=1)"); ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(FIG / "05_target_rate_by_quarter.png"); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
