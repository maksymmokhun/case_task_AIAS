"""Опційний міні-експеримент: ознаки з текстових повідомлень публічного каналу Повітряних сил (t.me/s/kpszsu).

Правило без витоку: ознака в момент t залежить лише від повідомлень з часом публікації <= t.
Відредаговані повідомлення (на сторінці позначка «edited») у підрахунок згадок областей не входять: ми бачимо лише
останню редакцію, а час правки невідомий, тож текст міг змінитись після t. У загальну кількість вони входять
(час первісної публікації відомий). Видалені повідомлення не видно взагалі, цей зсув усунути не можна.
Згадка області шукається за основами слів (назва області, обласний центр), міста поза цим переліком не враховуються.
"""
import html
import re

import numpy as np
import pandas as pd

from .config import RAW
from .state import to_ns

PAGES_DIR = RAW / "telegram_kpszsu"
WINDOWS_H = (1, 3)

# Основи слів (нижній регістр). Формат: область (як у сітці) -> регулярний вираз; (?<!\w) = початок слова.
_STEMS = {
    "Vinnytska oblast": r"вінниччин|вінниц",
    "Volynska oblast": r"волин|луцьк|луцьку|луцька",
    "Dnipropetrovska oblast": r"дніпропетровщ|дніпропетровськ|дніпр",
    "Donetska oblast": r"донеччин|донецьк",
    "Zhytomyrska oblast": r"житомирщ|житомир",
    "Zakarpatska oblast": r"закарпат|ужгород",
    "Zaporizka oblast": r"запоріж|запорізьк",
    "Ivano-Frankivska oblast": r"івано-франківщ|івано-франківськ|прикарпат",
    "Kyivska oblast": r"київщ|київськ",
    "Kirovohradska oblast": r"кіровоградщ|кіровоградськ|кропивницьк",
    "Lvivska oblast": r"львівщ|львівськ|львів|львов",
    "Mykolaivska oblast": r"миколаївщ|миколаївськ|миколаїв|миколає",
    "Odeska oblast": r"одещ|одеськ|одес",
    "Poltavska oblast": r"полтавщ|полтавськ|полтав",
    "Rivnenska oblast": r"рівненщ|рівненськ|рівне|рівного",
    "Sumska oblast": r"сумщин|сумськ|сум(?:и|ах|ам|ами)(?!\w)",
    "Ternopilska oblast": r"тернопільщ|тернопільськ|тернопіл|тернопол",
    "Kharkivska oblast": r"харківщ|харківськ|харків|харков",
    "Khersonska oblast": r"херсонщ|херсонськ|херсон",
    "Khmelnytska oblast": r"хмельниччин|хмельницьк",
    "Cherkaska oblast": r"черкащ|черкаськ|черкас",
    "Chernivetska oblast": r"чернівецьк|чернівц|буковин",
    "Chernihivska oblast": r"чернігівщ|чернігівськ|чернігів|чернігов",
    "Kyiv City": r"київ(?!щ|ськ)|києв",
}
PATTERNS = {o: re.compile(r"(?<!\w)(?:" + s + ")") for o, s in _STEMS.items()}
# Тип загрози за ключовими словами (задано до перегляду результатів, не підбиралось).
TYPES = {"drone": r"бпла|шахед|дрон|герань", "missile": r"ракет|крилат|калібр", "ballistic": r"балістик"}
TYPE_PATTERNS = {k: re.compile(v) for k, v in TYPES.items()}

_BLOCK = re.compile(r'<div class="tgme_widget_message_wrap')
_TEXT = re.compile(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", " ", fragment)
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def parse_page(raw: str) -> list:
    """Повідомлення зі сторінки вебперегляду: id, time (UTC), text, edited."""
    out = []
    for block in _BLOCK.split(raw)[1:]:
        post = re.search(r'data-post="kpszsu/(\d+)"', block)
        when = re.search(r'<time[^>]*datetime="([^"]+)"', block)
        if not post or not when:
            continue
        txt = _TEXT.search(block)
        meta = re.search(r'tgme_widget_message_meta">(.*?)<a class="tgme_widget_message_date"', block, re.S)
        out.append({"id": int(post.group(1)), "time": pd.Timestamp(when.group(1)).tz_convert("UTC"),
                    "text": _clean(txt.group(1)) if txt else "", "edited": bool(meta and "edited" in meta.group(1))})
    return out


def load_messages(pages_dir=None) -> pd.DataFrame:
    """Усі збережені сторінки, без дублікатів за id, за часом."""
    rows = []
    for f in sorted((pages_dir or PAGES_DIR).glob("page_*.html")):
        rows += parse_page(f.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows).drop_duplicates("id")
    return df.sort_values("time").reset_index(drop=True)


def mentions(df: pd.DataFrame) -> dict:
    """Для кожної області: булева маска повідомлень, що її згадують (відредаговані не враховуються)."""
    low = df.text.str.lower()
    ok = ~df.edited.to_numpy()
    return {o: low.str.contains(p).to_numpy() & ok for o, p in PATTERNS.items()}


def _count(times, tt, span):
    """Скільки значень times (відсортовані) потрапляє в (tt - span, tt] для кожного tt."""
    return np.searchsorted(times, tt, "right") - np.searchsorted(times, tt - span, "right")


def text_features(grid: pd.DataFrame, msgs: pd.DataFrame, typed: bool = False) -> pd.DataFrame:
    """txt_ob_{w}h: повідомлення, що згадують область, у (t - w, t]; txt_all_{w}h: усі повідомлення каналу; log1p.

    typed=True додає txt_ob_{тип}_{w}h: згадують область і містять ключове слово типу загрози (TYPES)."""
    tm = to_ns(msgs.time)
    men = mentions(msgs)
    low = msgs.text.str.lower()
    kinds = {k: low.str.contains(p).to_numpy() for k, p in TYPE_PATTERNS.items()} if typed else {}
    ob = grid.oblast.astype(str).to_numpy()
    tt_all = to_ns(grid.t)
    names = [f"txt_{k}_{w}h" for k in ("ob", "all") for w in WINDOWS_H]
    names += [f"txt_ob_{k}_{w}h" for k in kinds for w in WINDOWS_H]
    cols = {n: np.zeros(len(grid), np.float32) for n in names}
    for w in WINDOWS_H:
        span = np.timedelta64(w, "h")
        cols[f"txt_all_{w}h"][:] = np.log1p(_count(tm, tt_all, span))
        for o in np.unique(ob):
            ix = np.flatnonzero(ob == o)
            tt = tt_all[ix]
            sel = men.get(o, np.zeros(len(tm), bool))
            cols[f"txt_ob_{w}h"][ix] = np.log1p(_count(tm[sel], tt, span))
            for k, km in kinds.items():
                cols[f"txt_ob_{k}_{w}h"][ix] = np.log1p(_count(tm[sel & km], tt, span))
    return pd.DataFrame(cols, index=grid.index)
