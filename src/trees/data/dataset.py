"""Helper class for datasets."""

from dataclasses import dataclass

import polars as pl


@dataclass
class Dataset:
    """Helper class for datasets."""

    df: pl.DataFrame
    target_col: str
    _id_col: str = "__id_col"

    def __post_init__(self) -> None:
        """Initialize the dataset."""
        if self._id_col not in self.df.columns:
            self.df = self.df.with_columns(pl.arange(0, self.df.height).alias(self._id_col))

    @property
    def feature_names(self) -> list[str]:
        """Get the names of the features in the dataset."""
        return [col for col in self.df.columns if col not in (self.target_col, self._id_col)]

    def filter_to_below_threshold(self, feature_name: str, threshold: float) -> "Dataset":
        """Filter the dataset to only include rows where the feature is below the threshold."""
        filtered_df = self.df.filter(pl.col(feature_name) < threshold)
        return Dataset(df=filtered_df, target_col=self.target_col)

    def filter_to_above_or_at_threshold(self, feature_name: str, threshold: float) -> "Dataset":
        """Filter dataset to only include rows where the feature is at or above the threshold."""
        filtered_df = self.df.filter(pl.col(feature_name) >= threshold)
        return Dataset(filtered_df, target_col=self.target_col)

    def filter_to_nulls(self, feature_name: str) -> "Dataset":
        """Filter dataset to only include rows where the feature is null."""
        filtered_df = self.df.filter(pl.col(feature_name).is_null())
        return Dataset(filtered_df, target_col=self.target_col)

    def get_rows_by_ids(self, ids: list[str]) -> "Dataset":
        """Get rows by their ids."""
        filtered_df = self.df.filter(pl.col(self._id_col).is_in(ids))
        return Dataset(filtered_df, target_col=self.target_col)
