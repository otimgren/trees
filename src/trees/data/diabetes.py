"""Helper to load diabetes dataset."""
import polars as pl


def load_raw_data() -> pl.DataFrame:
    """Load the raw diabetes dataset."""
    return pl.read_csv("/home/oskari/data_science/trees/data/diabetes/diabetes.csv")
