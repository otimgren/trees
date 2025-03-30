"""Helpers for loading the Titanic dataset."""

import polars as pl


def load_raw_data() -> pl.DataFrame:
    """Load the raw Titanic dataset."""
    return pl.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
