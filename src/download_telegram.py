"""Завантаження сторінок публічного вебперегляду каналу Повітряних сил (https://t.me/s/kpszsu) для описового
міні-експерименту з текстовими даними (опційний пункт завдання). Сторінки йдуть від найновіших до старших
(?before=<id>), зупинка, коли найстаріше повідомлення на сторінці раніше за межу. Файли пишуться в data/raw/telegram_kpszsu/
(у .gitignore). На відміну від основних даних, ці не закріплені за хешем: канал живий, тому відтворити точний набір можна лише
з архіву цих сторінок, який у репозиторії не зберігається.
Якщо каталог уже містить сторінки, завантаження продовжується від найстарішої. Запуск: cd src && ../.venv/bin/python download_telegram.py
"""
import re
import time
import urllib.request

from airalerts.config import RAW

OUT = RAW / "telegram_kpszsu"
CHANNEL = "https://t.me/s/kpszsu"
SINCE = "2026-03-27T00:00:00+00:00"   # межа вибірки, UTC (V0 валідації починається 2026-03-28)
PAUSE_S = 1.0


def fetch(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "air-alerts-forecast research (selection test task)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"спроба {k + 1} не вдалась: {e}", flush=True)
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"не вдалось завантажити {url}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have = sorted(OUT.glob("page_*.html"))
    n, total = len(have), 0
    url = f"{CHANNEL}?before={min(int(f.stem.split('_')[2]) for f in have)}" if have else CHANNEL   # дозавантаження глибше в історію
    while True:
        html = fetch(url)
        ids = [int(i) for i in re.findall(r'data-post="kpszsu/(\d+)"', html)]
        times = re.findall(r'<time[^>]*datetime="([^"]+)"', html)
        if not ids or not times:
            print("порожня сторінка, зупинка", flush=True)
            break
        (OUT / f"page_{n:04d}_{min(ids)}_{max(ids)}.html").write_text(html, encoding="utf-8")
        total += len(html.encode("utf-8"))
        n += 1
        oldest = min(times)
        if n % 50 == 0 or oldest < SINCE:
            print(f"сторінок {n}, {total / 1e6:.1f} МБ, найстаріше повідомлення {oldest}", flush=True)
        if oldest < SINCE:
            break
        url = f"{CHANNEL}?before={min(ids)}"
        time.sleep(PAUSE_S)
    print(f"Готово: {n} сторінок, {total / 1e6:.1f} МБ у {OUT}", flush=True)


if __name__ == "__main__":
    main()
