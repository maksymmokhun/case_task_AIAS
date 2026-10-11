"""Рис. 12, 16, 21 і 22 (карти й важливість ознак) з уже порахованих таблиць, без нових обчислень моделей і без читання тесту.

Ті самі карти малюють run_model.py (рис. 12) і run_test.py (рис. 16) під час повного запуску; цей скрипт
перемальовує їх із CSV, не перезапускаючи тест.
Потребує data/raw/ua_oblasts.geojson (python src/download_data.py).
Запуск: cd src && ../.venv/bin/python make_maps.py
"""
import pandas as pd

from airalerts.config import FIG, TAB
from airalerts.maps import map_test_bss, map_val_auc, map_val_bss, plot_importance


def main():
    map_test_bss(pd.read_csv(TAB / "test_by_oblast.csv"), FIG / "16_test_bss_by_oblast.png")
    summ = pd.read_csv(TAB / "model_val_summary_conservative.csv", index_col=0)
    new = [m for m in summ.index if m.endswith("_nested") or m.endswith("_sel")]
    top = min(new, key=lambda m: summ.loc[m, "brier"])
    base = [c for c in summ.index if c.startswith("B")]
    best_base = min(base, key=lambda m: summ.loc[m, "brier"])
    map_val_bss(pd.read_csv(TAB / "model_val_by_oblast.csv"), top, best_base, FIG / "12_model_bss_by_oblast.png")
    auc = TAB / "model_val_oblast_auc.csv"                    # лише якщо вже порахований run_importance.py
    if auc.exists():
        map_val_auc(pd.read_csv(auc), FIG / "22_val_auc_by_oblast.png")
    imp = TAB / "model_val_feature_importance.csv"
    if imp.exists():
        plot_importance(pd.read_csv(imp), FIG / "21_feature_importance.png")
    print(f"готово: рис. 16, 12, 21 і 22 (модель {top}, база {best_base})")


if __name__ == "__main__":
    main()
