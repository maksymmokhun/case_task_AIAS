"""Прогнози баз B0–B3 на валідаційних вікнах за обома схемами розмітки (спільне для run_baselines і run_model)."""
import time

import numpy as np
import pandas as pd

from . import baselines as bl
from .splits import train_mask

SCHEMES = {"conservative": "elig_cons", "dataset": "elig_ds"}
B3_CAL_DAYS = 90    # вікно навчання рекалібрації B3 (на валідації не підбиралось, взято те саме, що для B1)
B1_TRAIN_DAYS = 90   # вікно навчання B1 обрано на валідації (див. таблицю b1_training_window.csv)
MODELS = ["B2_90d", "B0_28d", "B0_90d", "B1_seasonal_logit", "B3_ETS", "B3_ARIMA", "B3_ETS_cal", "B3_ARIMA_cal"]


def baseline_predictions(events, grid, wins):
    """{схема: {модель: масив прогнозів по рядках grid (NaN поза валідаційними вікнами для B1, B3)}}."""
    preds = {s: {} for s in SCHEMES}
    for s, col in SCHEMES.items():
        preds[s]["B2_90d"] = bl.b2_trailing_rate(grid, col, 90)
        preds[s]["B0_28d"] = bl.b0_hour_rate(grid, col, 28)
        preds[s]["B0_90d"] = bl.b0_hour_rate(grid, col, 90)
        p1 = np.full(len(grid), np.nan)
        for w0, w1 in wins:
            pred = ((grid.t >= w0) & (grid.t < w1)).to_numpy()
            p1[pred] = bl.b1_fit_predict(grid, col, train_mask(grid, w0, B1_TRAIN_DAYS), pred)[pred]
        preds[s]["B1_seasonal_logit"] = p1
    # B3 не залежить від схеми (не використовує стан): рахуємо один раз на об'єднанні придатних зразків.
    # Діапазон розширено назад на B3_CAL_DAYS, щоб навчити рекалібрацію.
    in_b3 = ((grid.t >= wins[0][0] - pd.Timedelta(days=B3_CAL_DAYS + 5)) & (grid.t < wins[-1][1])).to_numpy()
    rows = np.flatnonzero(in_b3 & (grid.elig_cons | grid.elig_ds).to_numpy())
    for name, method in (("B3_ETS", "ets"), ("B3_ARIMA", "arima")):
        t0 = time.time()
        p3 = bl.b3_probabilities(grid, events, method, rows)
        for s, col in SCHEMES.items():
            preds[s][name] = p3
            cal = np.full(len(grid), np.nan)
            for w0, w1 in wins:
                pred = ((grid.t >= w0) & (grid.t < w1)).to_numpy()
                cal[pred] = bl.recalibrate(grid, col, p3, train_mask(grid, w0, B3_CAL_DAYS), pred)[pred]
            preds[s][name + "_cal"] = cal
        print(f"{name}: {len(rows):,} рядків, {time.time() - t0:.0f} с")
    return preds
