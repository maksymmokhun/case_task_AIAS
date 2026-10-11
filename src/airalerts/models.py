"""Основна модель (ітерація 4): логістична регресія і HistGradientBoosting на ознаках з features.py.

Усі статистики (масштаб, дерева) оцінюються лише на навчальних рядках. Навчальна маска будується через
splits.train_mask: purge H+1 год перед початком прогнозного вікна, опційно лише останні `days` діб.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import P_MAX, P_MIN
from .features import GROUPS

# Для логістичної регресії година подається гармоніками, день тижня прапорцем вихідного, область дамі-колонками.
_LOGIT_DROP = {"hour_local", "dow", "oblast_code"}
_HGB_DROP = {"hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2"}
N_OBLASTS = 24


def feature_columns(kind: str, drop_groups: tuple = ()) -> list:
    """Колонки моделі. drop_groups виключає групи ознак (для абляції)."""
    skip = _LOGIT_DROP if kind == "logit" else _HGB_DROP
    return [c for g, cols in GROUPS.items() if g not in drop_groups for c in cols if c not in skip]


def _design(X: pd.DataFrame, rows: np.ndarray, cols: list, dummies: bool) -> np.ndarray:
    dense = X[cols].iloc[rows].to_numpy(np.float32)
    if dummies:
        codes = X["oblast_code"].iloc[rows].to_numpy().astype(int)
        dense = np.hstack([dense, np.eye(N_OBLASTS, dtype=np.float32)[codes]])
    return dense


def fit_predict(kind: str, params: dict, X: pd.DataFrame, y: np.ndarray, train_rows: np.ndarray,
                pred_rows: np.ndarray, drop_groups: tuple = (), return_model: bool = False):
    """Навчає модель на train_rows (позиції в X) і повертає ймовірності для pred_rows.

    kind='logit': StandardScaler + LogisticRegression(C). kind='hgb': HistGradientBoostingClassifier.
    """
    cols = feature_columns(kind, drop_groups)
    if kind == "logit":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=500))
        dummies = "oblast" not in drop_groups
    else:
        cat = [cols.index("oblast_code")] if "oblast_code" in cols else None
        model = HistGradientBoostingClassifier(
            learning_rate=params.get("lr", 0.05), max_iter=params["max_iter"], max_leaf_nodes=params["leaves"],
            min_samples_leaf=params["min_leaf"], l2_regularization=1.0, categorical_features=cat,
            early_stopping=False, random_state=0)
        dummies = False
    model.fit(_design(X, train_rows, cols, dummies), y[train_rows])
    p = np.clip(model.predict_proba(_design(X, pred_rows, cols, dummies))[:, 1], P_MIN, P_MAX)
    return (p, model, cols) if return_model else p
