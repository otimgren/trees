"""Splitting based on XGBoost gain."""

import jax.numpy as np
from jax import jit

def calculate_loss_