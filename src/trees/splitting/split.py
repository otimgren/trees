"""Code for splitting nodes based on the data available to them."""

from collections.abc import Callable

import jax.numpy as np
from jax.typing import ArrayLike

from trees.df import DataFrame
from trees.splitting.gini import calculate_increase_in_gini_impurity


def find_threshold_candidates(feature_values: ArrayLike) -> ArrayLike:
    """Get candidates for the threshold based on the provided feature values."""
    sorted_unique_values = np.sort(np.unique(feature_values))
    return (sorted_unique_values[1:] + sorted_unique_values[:-1]) / 2.0


def suggest_split_threshold(
    df: DataFrame,
    feature: str,
    metric_function: Callable[
        [ArrayLike, ArrayLike, float], float
    ] = calculate_increase_in_gini_impurity,
) -> tuple[float, float]:
    """Suggest the threshold for a split. Returns the suggested threshold and the metric."""
    feature_values = df[feature]
    thresholds = find_threshold_candidates(feature_values)
    return min(
        (metric_function(feature_values, df.labels, threshold), threshold)
        for threshold in thresholds
    )
