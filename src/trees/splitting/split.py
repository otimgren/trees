"""Code for splitting nodes based on the data available to them."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from trees.df import DataFrame
from trees.splitting.gini import calculate_increase_in_gini_impurity


def find_threshold_candidates(feature_values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Get candidates for the threshold based on the provided feature values."""
    sorted_unique_values = np.sort(np.unique(feature_values))
    return (sorted_unique_values[1:] + sorted_unique_values[:-1]) / 2.0


def suggest_split_threshold(
    df: DataFrame,
    feature: str,
    metric_function: Callable[
        [NDArray[np.float32], NDArray[np.float32], float], float
    ] = calculate_increase_in_gini_impurity,
) -> tuple[float, float]:
    """Suggest the threshold for a split. Returns the suggested threshold and the metric."""
    feature_values = df[feature]
    thresholds = find_threshold_candidates(feature_values)
    return min(
        (metric_function(feature_values, df.labels, threshold), threshold)
        for threshold in thresholds
    )
