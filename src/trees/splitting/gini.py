"""Splitting based on Gini impurity."""

import numpy as np
from numpy.typing import NDArray


# @jit
def calculate_gini_impurity(
    feature_values: NDArray[np.float32], labels: NDArray[np.float32], threshold: float
) -> float:
    """Calculate the Gini coefficient if the feature is split to two categories at threshold."""
    mask = feature_values > threshold
    if mask.sum() == 0 or mask.sum() == len(mask):
        return 1
    p = (labels * mask).sum() / mask.sum()
    return p * (1 - p)


def calculate_increase_in_gini_impurity(
    feature_values: NDArray[np.float32], labels: NDArray[np.float32], threshold: float
) -> float:
    """Calculate the increase in overall Gini impurity when splitting node at threshold."""
    parent_gini_impurity = calculate_gini_impurity(feature_values, labels, threshold)
    mask = feature_values <= threshold
    left_gini_impurity = calculate_gini_impurity(feature_values[mask], labels[mask], threshold)
    right_gini_impurity = calculate_gini_impurity(feature_values[~mask], labels[~mask], threshold)
    return (left_gini_impurity + right_gini_impurity) - parent_gini_impurity
