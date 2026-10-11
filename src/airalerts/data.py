"""Завантаження волонтерського файлу тривог."""
import pandas as pd

from .config import EXCLUDED_OBLASTS, RAW, TEST_START


def load_events(path=None, cut_test=True):
    """Повертає тривоги: oblast, started_at, finished_at, naive. Відсортовано за (oblast, started_at).

    cut_test=True відкидає все, що починається з TEST_START, щоб тест не потрапив у розробку.
    """
    df = pd.read_csv(path or RAW / "volunteer_data_en.csv", parse_dates=["started_at", "finished_at"])
    df = df.rename(columns={"region": "oblast"})
    df = df[~df.oblast.isin(EXCLUDED_OBLASTS)]
    if cut_test:
        df = df[df.started_at < TEST_START]
    return df.sort_values(["oblast", "started_at"]).reset_index(drop=True)
