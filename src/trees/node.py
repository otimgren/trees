"""Tree nodes."""

import uuid
from dataclasses import dataclass, field
from typing import Self

from trees.df import DataFrame


@dataclass
class Node:
    """Tree node."""

    id: str
    parent: "Node | None"
    left_child: "Node | None" = None
    right_child: "Node | None" = None
    feature_name: str = ""
    threshold: float = float("nan")
    data_ids: list[str] = field(default_factory=list)
    logodds: float = float("nan")

    @property
    def is_root(self) -> bool:
        """Check if the node is the root."""
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf."""
        return self.left_child is None and self.right_child is None

    @property
    def is_left_child(self) -> bool:
        """Check if the node is a left child."""
        return self.parent is not None and self.parent.left_child == self

    @property
    def is_right_child(self) -> bool:
        """Check if the node is a right child."""
        return self.parent is not None and self.parent.right_child == self

    @property
    def is_split(self) -> bool:
        """Check if the node is split."""
        return self.left_child is not None or self.right_child is not None

    @property
    def level(self) -> int:
        """Get the level of the node in the tree."""
        level = 0
        current_node = self
        while current_node.parent is not None:
            level += 1
            current_node = current_node.parent
        return level

    def split(
        self,
        feature_name: str,
        threshold: float,
        left_id: str | None = None,
        right_id: str | None = None,
        df: DataFrame | None = None,
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
        self.left_child = Node(
            id=left_id,
            parent=self,
            data_ids=left_dataset.ids,
            logodds=left_dataset.get_logodds(),
        )
        self.right_child = Node(
            id=right_id,
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

    def find_next(self, features: dict[str, float]) -> "Node | None":
        """Find the next node to traverse based on features."""
        if self.is_leaf:
            return None
        if self.threshold is None or self.feature_name is None:
            msg = "Node not split yet."
            raise ValueError(msg)
        if self.left_child is None and self.right_child is not None:
            return self.right_child
        if self.left_child is not None and self.right_child is None:
            return self.left_child
        if features[self.feature_name] < self.threshold:
            return self.left_child
        return self.right_child
