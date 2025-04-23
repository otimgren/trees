"""Code for splitting nodes based on the data available to them."""

from collections.abc import Callable

import polars as pl

from trees.data.dataset import Dataset


def find_threshold_candidates(feature_values: pl.Series) -> pl.Series:
    """Get candidates for the threshold based on the provided feature values."""
    return feature_values.unique().sort().rolling_median(window_size=2).drop_nulls()


def calculate_gini_impurity(feature_values: pl.Series[float], threshold: float) -> float:
    """Calculate the Gini coefficient if the feature is split to two categories at threshold."""
    classes = feature_values > threshold
    return classes.mean() * (1 - classes.mean())


def calculate_increase_in_gini_impurity(feature_values: pl.Series, threshold: float) -> float:
    """Calculate the increase in overall Gini impurity when splitting node at threshold."""
    parent_gini_impurity = calculate_gini_impurity(feature_values, threshold)
    left_gini_impurity = calculate_gini_impurity(
        feature_values.filter(feature_values <= threshold), threshold
    )
    right_gini_impurity = calculate_gini_impurity(
        feature_values.filter(feature_values > threshold), threshold
    )
    return (left_gini_impurity + right_gini_impurity) - parent_gini_impurity


def suggest_split_threshold(
    dataset: Dataset,
    feature: str,
    metric_function: Callable[[pl.Series, float], float] = calculate_increase_in_gini_impurity,
) -> tuple[float, float]:
    """Suggest the threshold for a split. Returns the suggested threshold and the metric."""
    feature_values = dataset.df[feature]
    thresholds = find_threshold_candidates(feature_values)
    return min((metric_function(feature_values, threshold), threshold) for threshold in thresholds)


def calculate_entropy(feature_values: pl.Series, threshold: float) -> float:
    """Calculate the entropy if the feature is split to two categories at threshold."""
    classes = feature_values > threshold
    p = classes.mean()
    if p == 0 or p == 1:
        return 0.0
    return -(p * pl.log(p) + (1 - p) * pl.log(1 - p))


def calculate_information_gain(feature_values: pl.Series, threshold: float) -> float:
    """Calculate the information gain when splitting node at threshold."""
    parent_entropy = calculate_entropy(feature_values, threshold)
    left_entropy = calculate_entropy(feature_values.filter(feature_values <= threshold), threshold)
    right_entropy = calculate_entropy(feature_values.filter(feature_values > threshold), threshold)
    left_size = feature_values.filter(feature_values <= threshold).len()
    right_size = feature_values.filter(feature_values > threshold).len()
    total_size = feature_values.len()

    weighted_entropy = (left_size / total_size) * left_entropy + (
        right_size / total_size
    ) * right_entropy
    return parent_entropy - weighted_entropy
