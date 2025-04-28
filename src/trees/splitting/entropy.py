"""Spltting based on information gain."""

import math

# import jax.numpy as np
import numpy as np
from jax import jit
from numpy.typing import NDArray


# @jit
def calculate_entropy(
    feature_values: NDArray[np.float32], labels: NDArray[np.float32], threshold: float
) -> float:
    """Calculate the entropy if the feature is split to two categories at threshold."""
    mask = feature_values > threshold
    p: float = labels[mask].mean()
    if p in {0, 1}:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def calculate_information_gain(
    feature_values: NDArray[np.float32], labels: NDArray[np.float32], threshold: float
) -> float:
    """Calculate the information gain when splitting node at threshold."""
    parent_entropy = calculate_entropy(feature_values, labels, threshold)
    mask: NDArray[np.bool] = feature_values <= threshold
    left_entropy = calculate_entropy(feature_values[mask], labels[mask], threshold)
    right_entropy = calculate_entropy(feature_values[~mask], labels[~mask], threshold)
    total_size = len(feature_values)
    left_size = len(feature_values[mask])
    right_size = total_size - left_size

    weighted_entropy = (left_size / total_size) * left_entropy + (
        right_size / total_size
    ) * right_entropy
    return parent_entropy - weighted_entropy
