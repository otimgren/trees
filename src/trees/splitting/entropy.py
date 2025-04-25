"""Spltting based on information gain."""

import jax.numpy as jnp
from jax.typing import ArrayLike


def calculate_entropy(feature_values: ArrayLike, threshold: float) -> float:
    """Calculate the entropy if the feature is split to two categories at threshold."""
    classes = feature_values > threshold
    p = classes.mean()
    if p in {0, 1}:
        return 0.0
    return (-(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))).tolist()


def calculate_information_gain(feature_values: ArrayLike, threshold: float) -> float:
    """Calculate the information gain when splitting node at threshold."""
    parent_entropy = calculate_entropy(feature_values, threshold)
    left_entropy = calculate_entropy(feature_values[feature_values <= threshold], threshold)
    right_entropy = calculate_entropy(feature_values[feature_values > threshold], threshold)
    total_size = len(feature_values)
    left_size = (feature_values <= threshold).sum()
    right_size = total_size - left_size

    weighted_entropy = (left_size / total_size) * left_entropy + (
        right_size / total_size
    ) * right_entropy
    return (parent_entropy - weighted_entropy).tolist()
