"""Tree nodes."""

import uuid
from typing import Any, TypeVar

import numpy as np
from bigtree.node.binarynode import BinaryNode
from numpy.typing import NDArray
from PIL.ImageChops import offset

from trees.df import DataFrame


class Node(BinaryNode):
    """Tree node."""

    def __init__(
        self,
        name: str | int,
        parent: "Node | None" = None,
        left: "Node | None" = None,
        right: "Node | None" = None,
        feature_name: str = "[feature_name]",
        threshold: float | None = None,
        data_ids: NDArray[np.str_ | np.int_] | None = None,
        logodds: float | None = None,
    ):
        super().__init__(name=name, parent=parent, left=left, right=right)
        self.feature_name: str = feature_name
        self.threshold: float = threshold or float("nan")
        self.data_ids: NDArray[np.str_ | np.int_] = (
            np.array([], dtype=np.str_) if data_ids is None else data_ids
        )
        self.logodds: float = logodds or float("nan")

    @property
    def id(self) -> str:
        """Get the node id."""
        return self.name

    @property
    def is_left_child(self) -> bool:
        """Check if the node is a left child."""
        return self.parent is not None and self.parent.left is self

    @property
    def is_right_child(self) -> bool:
        """Check if the node is a right child."""
        return self.parent is not None and self.parent.right is self

    @property
    def is_split(self) -> bool:
        """Check if the node is split."""
        return self.left is not None or self.right is not None

    @property
    def n_obs(self) -> int:
        """Get the number of observations in the node."""
        return len(self.data_ids)

    def split(
        self,
        feature_name: str,
        threshold: float,
        df: DataFrame,
        *,
        left_id: str | None = None,
        right_id: str | None = None,
    ) -> None:
        """Split the node based on feature and threshold."""
        if self.is_split:
            msg = "Node already split."
            raise ValueError(msg)

        if left_id is None:
            left_id = uuid.uuid4().hex
        if right_id is None:
            right_id = uuid.uuid4().hex

        self.feature_name = feature_name
        self.threshold = threshold
        left_dataset = df.filter_to_below_threshold(feature_name, threshold) + df.filter_to_nulls(
            feature_name
        )
        right_dataset = df.filter_to_above_or_at_threshold(feature_name, threshold)
        self.left: Node = Node(
            name=left_id,
            parent=self,
            data_ids=left_dataset.ids,
            logodds=left_dataset.get_logodds(),
        )
        self.right: Node = Node(
            name=right_id,
            parent=self,
            data_ids=right_dataset.ids,
            logodds=right_dataset.get_logodds(),
        )

    def predict(self) -> float:
        """Predict the output for given features."""
        if not self.is_leaf:
            msg = "Cannot predict on a non-leaf node."
            raise ValueError(msg)
        if self.logodds is float("nan"):
            msg = "Log odds not set for leaf node."
            raise ValueError(msg)
        return self.logodds
