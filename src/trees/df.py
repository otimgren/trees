"""My custom dataframe class."""

from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
import polars as pl
from jax.typing import ArrayLike


@dataclass
class DataFrame:
    """Custom dataframe class."""

    ids: ArrayLike
    features: ArrayLike
    feature_names: list[str]
    labels: ArrayLike
    id_col_name: str
    label_col_name: str

    @classmethod
    def from_polars(cls, df: pl.DataFrame, id_col_name: str, label_col_name: str) -> Self:
        """Convert a polars dataframe to a custom dataframe."""
        feature_names = [col for col in df.columns if col not in {id_col_name, label_col_name}]
        ids = jnp.array(df[id_col_name].to_numpy())
        features = jnp.array(df[feature_names].cast(pl.Float32).to_numpy())
        labels = jnp.array(df[label_col_name].cast(pl.Float32).to_numpy())

        return cls(
            ids=ids,
            features=features,
            feature_names=feature_names,
            labels=labels,
            id_col_name=id_col_name,
            label_col_name=label_col_name,
        )

    def __getitem__(self, key: str) -> ArrayLike:
        """Get a column by its name."""
        if key in self.feature_names:
            return self.features[:, self.feature_names.index(key)]
        if key == self.id_col_name:
            return self.ids
        if key == self.label_col_name:
            return self.labels
        raise KeyError(f"Column {key} not found in dataframe.")

    def get_rows_by_ids(self, ids: ArrayLike) -> "DataFrame":
        """Get rows by their ids."""
        mask = jnp.isin(self.ids, ids)
        return DataFrame(
            ids=self.ids[mask],
            features=self.features[mask],
            feature_names=self.feature_names,
            labels=self.labels[mask],
            id_col_name=self.id_col_name,
            label_col_name=self.label_col_name,
        )

    def __add__(self, other: object) -> "DataFrame":
        """Add two datasets together by stacking the dataframes."""
        if not isinstance(other, DataFrame):
            msg = "Can only add DataFrame objects."
            raise TypeError(msg)
        if self.feature_names != other.feature_names:
            msg = "Feature names must match to concatenate DataFrames."
            raise ValueError(msg)
        if self.id_col_name != other.id_col_name:
            msg = "ID column names must match to concatenate DataFrames."
            raise ValueError(msg)
        if self.label_col_name != other.label_col_name:
            msg = "Label column names must match to concatenate DataFrames."
            raise ValueError(msg)
        return DataFrame(
            ids=jnp.concatenate([self.ids, other.ids]),
            features=jnp.concatenate([self.features, other.features]),
            feature_names=self.feature_names,
            labels=jnp.concatenate([self.labels, other.labels]),
            id_col_name=self.id_col_name,
            label_col_name=self.label_col_name,
        )

    def filter_to_below_threshold(self, feature_name: str, threshold: float) -> "DataFrame":
        """Filter the dataset to only include rows where the feature is below the threshold."""
        mask = self.features[:, self.feature_names.index(feature_name)] < threshold
        return DataFrame(
            ids=self.ids[mask],
            features=self.features[mask],
            feature_names=self.feature_names,
            labels=self.labels[mask],
            id_col_name=self.id_col_name,
            label_col_name=self.label_col_name,
        )

    def filter_to_above_or_at_threshold(self, feature_name: str, threshold: float) -> "DataFrame":
        """Filter dataset to only include rows where the feature is at or above the threshold."""
        mask = self.features[:, self.feature_names.index(feature_name)] >= threshold
        return DataFrame(
            ids=self.ids[mask],
            features=self.features[mask],
            feature_names=self.feature_names,
            labels=self.labels[mask],
            id_col_name=self.id_col_name,
            label_col_name=self.label_col_name,
        )

    def filter_to_nulls(self, feature_name: str) -> "DataFrame":
        """Filter dataset to only include rows where the feature is null."""
        mask = jnp.isnan(self.features[:, self.feature_names.index(feature_name)])
        return DataFrame(
            ids=self.ids[mask],
            features=self.features[mask],
            feature_names=self.feature_names,
            labels=self.labels[mask],
            id_col_name=self.id_col_name,
            label_col_name=self.label_col_name,
        )

    def get_logodds(self) -> float:
        """Get the log odds of the target variable."""
        if self.labels.shape[0] == 0:
            return float("nan")

        if self.labels.mean() == 0:
            return -100
        if self.labels.mean() == 1:
            return 100
        return jnp.log(self.labels.mean() / (1 - self.labels.mean()))
