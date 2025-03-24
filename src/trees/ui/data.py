"""Helpers for specifying the dataset to use."""

import streamlit as st

from trees.data import diabetes
from trees.data.dataset import Dataset


def load_data(dataset_name: str) -> Dataset:
    """Load the specified dataset."""
    if dataset_name == "diabetes":
        return Dataset(df=diabetes.load_raw_data(), target_col="Outcome")

    msg = f"Dataset {dataset_name} not recognized."
    raise ValueError(msg)
