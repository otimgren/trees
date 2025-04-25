"""Class for individual decision trees."""

from dataclasses import dataclass

from trees.df import DataFrame
from trees.node import Node


@dataclass
class Tree:
    """Individual decision tree."""

    nodes: list[Node]
    root: Node
    df: DataFrame

    def predict(self, features: list[float]) -> float:
        """Predict the output for given features."""
        # TODO: Implement the predict method
        return 0.0

    def get_node_by_id(self, node_id: str) -> Node:
        """Get the node corresponding to the given id."""
        node = next((node for node in self.nodes if node.id == node_id), None)
        if node is None:
            msg = f"Node id {node_id} not found in the tree"
            raise KeyError(msg)
        return node

    @property
    def n_nodes(self) -> int:
        """Get the number of nodes in the tree."""
        return len(self.nodes)

    def split_node(self, node_id: str, feature_name: str, threshold: float) -> None:
        """Split the node based on feature and threshold."""
        node = self.get_node_by_id(node_id)
        node.split(
            feature_name,
            threshold,
            df=self.df.get_rows_by_ids(node.data_ids),
        )
        if node.left_child is None or node.right_child is None:
            msg = "Failed to split node."
            raise ValueError(msg)
        self.nodes.extend([node.left_child, node.right_child])

    def delete_node(self, node_id: str, make_new_leaf: bool = True) -> None:
        """Delete the node corresponding to the given id."""
        node = self.get_node_by_id(node_id)
        if node is None:
            msg = f"Node with id {node_id} not found."
            raise ValueError(msg)
        if node.left_child is not None:
            self.delete_node(node.left_child.id, make_new_leaf=False)
        if node.right_child is not None:
            self.delete_node(node.right_child.id, make_new_leaf=False)
        if node.is_root:
            msg = "Can't delete root node."
            raise ValueError(msg)
        self.nodes.remove(node)
        if not node.is_leaf and make_new_leaf:
            combined_ids = (node.left_child.data_ids) + (node.right_child.data_ids)
            new_leaf = Node(
                id=node.id,
                parent=node.parent,
                data_ids=combined_ids,
                logodds=self.df.get_rows_by_ids(combined_ids).get_logodds(),
            )
            if node.parent.left_child == node:
                node.parent.left_child = new_leaf
            elif node.parent.right_child == node:
                node.parent.right_child = new_leaf
            self.nodes.append(new_leaf)
