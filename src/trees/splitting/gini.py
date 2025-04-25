"""Splitting based on Gini impurity."""

from jax.typing import ArrayLike


def calculate_gini_impurity(feature_values: ArrayLike, threshold: float) -> float:
    """Calculate the Gini coefficient if the feature is split to two categories at threshold."""
    classes = feature_values > threshold
    return (classes.mean() * (1 - classes.mean())).tolist()


def calculate_increase_in_gini_impurity(feature_values: ArrayLike, threshold: float) -> float:
    """Calculate the increase in overall Gini impurity when splitting node at threshold."""
    parent_gini_impurity = calculate_gini_impurity(feature_values, threshold)
    left_gini_impurity = calculate_gini_impurity(
        feature_values[feature_values <= threshold], threshold
    )
    right_gini_impurity = calculate_gini_impurity(
        feature_values[feature_values > threshold], threshold
    )
    return (left_gini_impurity + right_gini_impurity) - parent_gini_impurity
