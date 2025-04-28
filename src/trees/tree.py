"""Class for individual decision trees."""

from dataclasses import dataclass

import numpy as np
from bigtree.tree.search import find_name

from trees.df import DataFrame
from trees.node import Node


@dataclass
class Tree:
    """Individual decision tree."""

    root: Node
    df: DataFrame

    def predict(self, features: list[float]) -> float:
        """Predict the output for given features."""
        # TODO: Implement the predict method
        return 0.0

    def get_node_by_id(self, node_id: str) -> Node:
        """Get the node corresponding to the given id."""
        node = find_name(self.root, node_id)
        if not node:
            msg = f"Node id {node_id} not found in the tree"
            raise KeyError(msg)
        return node

    def split_node(self, node_id: str, feature_name: str, threshold: float) -> None:
        """Split the node based on feature and threshold."""
        node = self.get_node_by_id(node_id)
        node.split(
            feature_name,
            threshold,
            df=self.df.get_rows_by_ids(node.data_ids),
        )
        if node.left is None or node.right is None:
            msg = f"Failed to split node: number of children is {len(node.children)}"
            raise ValueError(msg)

    def delete_node(self, node_id: str, make_new_leaf: bool = True) -> None:
        """Delete the node corresponding to the given id."""
        node = self.get_node_by_id(node_id)
        if node is None:
            msg = f"Node with id {node_id} not found."
            raise ValueError(msg)
        if node.is_root:
            msg = "Can't delete root node."
            raise ValueError(msg)
        if not node.is_leaf and make_new_leaf:
            combined_ids = np.concat([node.left.data_ids, node.right.data_ids])
            new_leaf = Node(
                name=node.id,
                data_ids=combined_ids,
                logodds=self.df.get_rows_by_ids(combined_ids).get_logodds(),
            )
        else:
            new_leaf = None
        if node.is_left_child:
            node.parent.left = new_leaf
        elif node.is_right_child:
            node.parent.right = new_leaf
        del node.children
