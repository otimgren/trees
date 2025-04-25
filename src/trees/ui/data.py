"""Helpers for specifying the dataset to use."""

import streamlit as st

from trees.data import diabetes
from trees.df import DataFrame


def load_data(dataset_name: str) -> DataFrame:
    """Load the specified dataset."""
    if dataset_name == "diabetes":
        return DataFrame.from_polars(
            df=diabetes.load_raw_data().with_row_index(),
            id_col_name="index",
            label_col_name="Outcome",
        )

    msg = f"Dataset {dataset_name} not recognized."
    raise ValueError(msg)
