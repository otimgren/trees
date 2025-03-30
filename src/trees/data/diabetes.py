"""Helper to load diabetes dataset."""

from pathlib import Path

import polars as pl

from trees.config import DATA_PATH


def load_raw_data() -> pl.DataFrame:
    """Load the raw diabetes dataset."""
    return pl.read_csv(DATA_PATH / Path("diabetes/diabetes.csv"))
