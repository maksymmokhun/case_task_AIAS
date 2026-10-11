"""Константи проєкту. Рішення задокументовано в docs/problem_statement.md і docs/definition_of_done.md."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, TAB, FIG = ROOT / "data/raw", ROOT / "reports/tables", ROOT / "reports/figures"

HORIZON_H = 3                      # горизонт прогнозу, години
EXCLUDED_OBLASTS = ("Luhanska oblast",)  # постійну тривогу вилучено з датасету, стан «немає тривоги» хибний

# Фінальний тест: 8 тижнів до знімка даних (2026-10-10 04:00 UTC). Зафіксовано константою, щоб межа
# не зсувалась при повторному завантаженні. У ітераціях до фінальної оцінки дані з цієї межі не читаються.
TEST_START = pd.Timestamp("2026-08-15 04:00", tz="UTC")

# Останній валідаційний момент t: вікно цілі (t, t+H] закінчується до TEST_START (purge H+1 год).
VAL_LAST_T = TEST_START - pd.Timedelta(hours=HORIZON_H + 1)
VAL_WINDOWS, VAL_WEEKS = 4, 4      # rolling-origin: 4 послідовні вікна по 4 тижні безпосередньо перед тестом

# Конфігурації моделей, зафіксовані до запуску на тесті (docs/definition_of_done.md, «Основна модель і протокол тесту»).
PRIMARY = ("logit", {"C": 0.03, "days": 180})                                    # основна (правило успіху)
SECONDARY = ("hgb", {"leaves": 16, "min_leaf": 400, "max_iter": 100, "days": 365})  # додаткова
BLOCK_DAYS, BLOCK_DAYS_ROBUST, N_BOOT = 3, 7, 1000    # бутстреп: основний блок, перевірка стійкості, повторів
REGIME_CHANGE = pd.Timestamp("2026-09-04", tz="Europe/Kyiv")   # постанова КМУ №1092: поділ на жовтий і червоний рівні
