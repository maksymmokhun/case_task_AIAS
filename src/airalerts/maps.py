"""Карти областей України (matplotlib, без geopandas): межі з data/raw/ua_oblasts.geojson (download_data.py).

У GeoJSON немає Києва як окремої одиниці й Севастополя: Київ малюється кружком на місці міста,
Крим і Луганська сірі («немає даних»). Значення області, позначені `flag`, теж сірі (замало подій).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

from .config import RAW

GEOJSON = RAW / "ua_oblasts.geojson"
KYIV_CITY = (30.52, 50.45)
# ISO 3166-2 -> назва області в наших таблицях
ISO = {"UA-71": "Cherkaska oblast", "UA-74": "Chernihivska oblast", "UA-77": "Chernivetska oblast",
       "UA-12": "Dnipropetrovska oblast", "UA-14": "Donetska oblast", "UA-26": "Ivano-Frankivska oblast",
       "UA-63": "Kharkivska oblast", "UA-65": "Khersonska oblast", "UA-68": "Khmelnytska oblast",
       "UA-32": "Kyivska oblast", "UA-35": "Kirovohradska oblast", "UA-09": "Luhanska oblast",
       "UA-46": "Lvivska oblast", "UA-48": "Mykolaivska oblast", "UA-51": "Odeska oblast",
       "UA-53": "Poltavska oblast", "UA-56": "Rivnenska oblast", "UA-59": "Sumska oblast",
       "UA-61": "Ternopilska oblast", "UA-05": "Vinnytska oblast", "UA-07": "Volynska oblast",
       "UA-21": "Zakarpatska oblast", "UA-23": "Zaporizka oblast", "UA-18": "Zhytomyrska oblast",
       "UA-43": "Crimea"}
UK = {"Cherkaska oblast": "Черкаська", "Chernihivska oblast": "Чернігівська", "Chernivetska oblast": "Чернівецька",
      "Dnipropetrovska oblast": "Дніпропетровська", "Donetska oblast": "Донецька",
      "Ivano-Frankivska oblast": "Івано-Франківська", "Kharkivska oblast": "Харківська",
      "Khersonska oblast": "Херсонська", "Khmelnytska oblast": "Хмельницька", "Kyivska oblast": "Київська",
      "Kirovohradska oblast": "Кіровоградська", "Luhanska oblast": "Луганська", "Lvivska oblast": "Львівська",
      "Mykolaivska oblast": "Миколаївська", "Odeska oblast": "Одеська", "Poltavska oblast": "Полтавська",
      "Rivnenska oblast": "Рівненська", "Sumska oblast": "Сумська", "Ternopilska oblast": "Тернопільська",
      "Vinnytska oblast": "Вінницька", "Volynska oblast": "Волинська", "Zakarpatska oblast": "Закарпатська",
      "Zaporizka oblast": "Запорізька", "Zhytomyrska oblast": "Житомирська", "Kyiv City": "Київ",
      "Crimea": "Крим"}
LABELS = {"en": {"nodata": "no data", "small": "fewer than 30 alert starts"},
          "uk": {"nodata": "немає даних", "small": "менше 30 початків"}}

# діверджентна пара палітри dataviz: синій (краще за еталон) <-> нейтральний сірий <-> червоний (гірше)
CMAP = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
GREY, GREY_EDGE, INK, INK2 = "#d9d8d3", "#ffffff", "#1a1a19", "#52514e"


def map_test_bss(po, path):
    """Рис. 16: BSS основної моделі проти B0_90d на тесті. po: таблиця test_by_oblast (індекс або стовпець oblast)."""
    po = po.set_index("oblast") if "oblast" in po.columns else po
    oblast_map(po["bss_LOGIT_vs_B0_90d"].to_dict(), set(po.index[po.flag_lt30_positives.astype(bool)]), path,
               "Skill of the primary model by oblast, test (8 weeks)", "BSS vs the 90-day baseline B0_90d\n(blue: better, red: worse)",
               fmt="{:+.0%}", lang="en")


def map_val_bss(po, top, best_base, path, min_pos=30):
    """Рис. 12: BSS кращої моделі проти найкращої бази на валідації. po: таблиця model_val_by_oblast."""
    po = po.set_index("oblast") if "oblast" in po.columns else po
    oblast_map(po["bss_vs_best_base"].to_dict(), set(po.index[po.positives < min_pos]), path,
               f"BSS по областях, валідація: {top} проти {best_base} (4 вікна по 4 тижні)",
               "BSS (синій: краще за базу, червоний: гірше)", fmt="{:+.0%}", lang="uk")


def _area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return abs(float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())) / 2


def load_shapes():
    """{назва: (зовнішні кільця, отвори)}. У наборі зовнішнє кільце не завжди перше (Одеська, Миколаївська, Херсонська),
    тому зовнішнім беремо кільце найбільшої площі, а решту: отвір, якщо лежить усередині нього, інакше окрема частина."""
    shapes = {}
    for f in json.loads(GEOJSON.read_text())["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        outer, holes = [], []
        for poly in polys:
            rings = sorted((np.asarray(r) for r in poly), key=_area, reverse=True)
            outer.append(rings[0])
            for r in rings[1:]:
                (holes if MPath(rings[0]).contains_point(r.mean(axis=0)) else outer).append(r)
        shapes[ISO[f["properties"]["iso3166-2"]]] = (outer, holes)
    return shapes


def _project(xy):
    """Рівнопромінна проєкція з поправкою cos(широти), щоб країна не була витягнута."""
    return np.column_stack([xy[:, 0] * np.cos(np.radians(48.5)), xy[:, 1]])


LABEL_SHIFT = {"Kyivska oblast": (0.0, -0.5), "Ternopilska oblast": (-0.3, 0.1), "Cherkaska oblast": (0.15, -0.1),
               "Zakarpatska oblast": (0.3, -0.3), "Ivano-Frankivska oblast": (0.2, 0.15), "Khmelnytska oblast": (0.25, 0.0),
               "Odeska oblast": (0.1, 0.65)}


def _fmt(fmt, v):
    """Формат значення; те, що округлюється до нуля, пишемо як «0%», без «-0%» і «+0%»."""
    out = fmt.format(v)
    return "0%" if out.strip("+-\u2212%") == "0" else out


def _anchor(rings):
    r = max(rings, key=lambda a: len(a))
    p = _project(r)
    x, y = p[:, 0], p[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cr.sum() / 2
    return np.array([((x + np.roll(x, -1)) * cr).sum() / (6 * a), ((y + np.roll(y, -1)) * cr).sum() / (6 * a)])


def _norm(ok, center, vmax):
    if center == 0:
        lim = vmax or float(np.ceil(max(abs(min(ok.values())), abs(max(ok.values()))) * 20) / 20)
        return TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    return TwoSlopeNorm(vmin=vmax[0], vcenter=center, vmax=vmax[1])


def _draw(ax, shapes, values, flags, norm, fmt, lang, mark=()):
    """Малює області на осях ax. mark: області, до значення яких додається «*» (мало подій, шумно)."""
    ok = {k: v for k, v in values.items() if k not in flags and np.isfinite(v)}
    star = lambda n: "*" if n in mark else ""
    for name, (rings, holes) in shapes.items():
        has = name in ok
        for ring in rings:
            ax.add_patch(PathPatch(MPath(_project(ring), closed=True), facecolor=CMAP(norm(ok[name])) if has else GREY,
                                   edgecolor=GREY_EDGE, lw=1.0, hatch=None if has else "////", zorder=2))
        c = _anchor(rings) + np.array(LABEL_SHIFT.get(name, (0.0, 0.0)))
        text = f"{(UK if lang == 'uk' else {}).get(name, name.replace(' oblast', ''))}"
        if has:
            tone = "#ffffff" if abs(norm(ok[name]) - 0.5) > 0.36 else INK
            ax.text(*c, f"{text}\n{_fmt(fmt, ok[name])}{star(name)}", ha="center", va="center", fontsize=6.6, color=tone, zorder=4)
        else:
            ax.text(*c, text, ha="center", va="center", fontsize=6.2, color=INK2, zorder=4)
    kx, ky = _project(np.array([KYIV_CITY]))[0]
    if "Kyiv City" in ok:
        ax.scatter([kx], [ky], s=170, color=CMAP(norm(ok["Kyiv City"])), edgecolor=INK, lw=1.0, zorder=5)
        ax.annotate(f"{UK['Kyiv City'] if lang == 'uk' else 'Kyiv City'}\n{_fmt(fmt, ok['Kyiv City'])}{star('Kyiv City')}",
                    (kx, ky), xytext=(kx + 1.0, ky + 1.35), fontsize=6.6, color=INK, zorder=6, ha="left",
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.7})
    xs = np.concatenate([_project(r)[:, 0] for rs, _ in shapes.values() for r in rs])
    ys = np.concatenate([_project(r)[:, 1] for rs, _ in shapes.values() for r in rs])
    ax.set_xlim(xs.min() - 0.3, xs.max() + 0.3); ax.set_ylim(ys.min() - 0.3, ys.max() + 0.3)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor("#fcfcfb")


def _colorbar(fig, axes, norm, fmt, label, shrink=0.5, cax=None):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=None if cax is not None else axes, cax=cax,
                      shrink=shrink, pad=0.01, aspect=22, format=lambda v, _: fmt.format(v))
    cb.set_label(label, fontsize=8, color=INK2); cb.ax.tick_params(labelsize=7, colors=INK2); cb.outline.set_visible(False)


def _foot(lang, notes):
    lab = LABELS[lang]
    foot = f"hatched grey: {lab['nodata']} or {lab['small']}" if lang == "en" else f"сірі зі штрихуванням: {lab['nodata']} або {lab['small']}"
    return foot + (f". {notes}" if notes else "")


def oblast_map(values, flags, path, title, cbar_label, fmt="{:+.0%}", lang="en", vmax=None, notes=None,
               shapes=None, center=0, mark=()):
    """Хороплет по областях. values: {назва області: число}, flags: множина областей без інтерпретації (сірі).
    center: значення, від якого колір міняє знак (0 для BSS, 0,5 для AUC); для center != 0 vmax = (мін, макс) шкали."""
    shapes = shapes or load_shapes()
    ok = {k: v for k, v in values.items() if k not in flags and np.isfinite(v)}
    norm = _norm(ok, center, vmax)
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    fig.patch.set_facecolor("#fcfcfb")
    _draw(ax, shapes, values, flags, norm, fmt, lang, mark)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    _colorbar(fig, ax, norm, fmt, cbar_label)
    fig.text(0.02, 0.02, _foot(lang, notes), fontsize=7, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.97, top=0.93, bottom=0.06)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


def map_val_auc(tab, path, min_pos=30, mark_below=100, lang="en"):
    """Рис. 22: AUC всередині області на валідації, база B0_90d проти основної моделі (одна шкала, центр 0,5 = випадковість).
    tab: model_val_oblast_auc (стовпці oblast, positives, auc_B0_90d, auc_LOGIT)."""
    tab = tab.set_index("oblast")
    shapes = load_shapes()
    flags = set(tab.index[tab.positives < min_pos]) | set(tab.index[tab.auc_LOGIT.isna()])
    mark = set(tab.index[tab.positives < mark_below])
    ok = tab[~tab.index.isin(flags)]
    norm = TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=0.75)
    T = {"en": ("Baseline B0_90d: alert frequency by oblast and hour", "Primary model (logistic regression)",
                "AUC within each oblast, validation (4 windows of 4 weeks): how well hours with an alert start are told apart from hours without",
                "AUC (0.50 = chance; blue: better, red: worse)",
                f"* fewer than {mark_below} alert starts in the period: noisy estimate. Hatched grey: no data."),
         "uk": ("База B0_90d: частота за областю та годиною", "Основна модель (логістична регресія)",
                "AUC всередині області, валідація (4 вікна по 4 тижні): наскільки добре години з тривогою відрізняються від годин без неї",
                "AUC (0,50 = випадково; синій: краще, червоний: гірше)",
                f"* менше {mark_below} початків тривог за період: оцінка шумна. Сірі зі штрихуванням: немає даних.")}[lang]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, col, ttl in zip(axes, ("auc_B0_90d", "auc_LOGIT"), T[:2]):
        _draw(ax, shapes, ok[col].to_dict(), flags, norm, "{:.2f}", lang, mark)
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=6)
    fig.suptitle(T[2], x=0.01, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.88, bottom=0.07, wspace=0.02)
    _colorbar(fig, None, norm, "{:.2f}", T[3], cax=fig.add_axes([0.948, 0.2, 0.009, 0.5]))
    fig.text(0.01, 0.02, T[4], fontsize=7.5, color=INK2)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


def plot_importance(imp, path, top=12):
    """Рис. 21: важливість окремих ознак на валідації (перестановка, Δ AUC всередині області), обидві моделі."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (kind, ttl) in zip(axes, (("logit", "Logistic regression (primary)"), ("hgb", "Gradient boosting (secondary)"))):
        d = imp[imp.model == kind].sort_values("perm_dAUC", ascending=False).head(top)[::-1]
        ax.barh(d.feature, d.perm_dAUC, xerr=d.perm_dAUC_sd, color="#2a78d6", error_kw={"ecolor": INK2, "lw": 0.8})
        ax.set_facecolor("#fcfcfb"); ax.set_title(ttl, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("AUC lost when the feature is shuffled (within-oblast AUC)", fontsize=8, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Which features the models rely on (validation, descriptive; bars: mean of 5 shuffles, whiskers: spread)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
