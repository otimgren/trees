"""Class for individual decision trees."""

from dataclasses import dataclass

from trees.node import Node


@dataclass
class Tree:
    """Individual decision tree."""

    nodes: list[Node]
    root: Node

    def predict(self, features: list[float]) -> float:
        """Predict the output for given features."""
        # TODO: Implement the predict method
        return 0.0

    def get_node_by_id(self, node_id: str) -> Node | None:
        """Get the node corresponding to the given id."""
        return next((node for node in self.nodes if node.id == node_id), None)

    @property
    def n_nodes(self) -> int:
        """Get the number of nodes in the tree."""
        return len(self.nodes)

    def split_node(self, node_id: str, feature_name: str, threshold: float) -> None:
        """Split the node based on feature and threshold."""
        node = self.get_node_by_id(node_id)
        if node is None:
            msg = f"Node with id {node_id} not found."
            raise ValueError(msg)
        node.split(feature_name, threshold)
        if node.left_child is None or node.right_child is None:
            msg = "Failed to split node."
            raise ValueError(msg)
        self.nodes.extend([node.left_child, node.right_child])

    def delete_node(self, node_id: str) -> None:
        """Delete the node corresponding to the given id."""
        node = self.get_node_by_id(node_id)
        if node is None:
            msg = f"Node with id {node_id} not found."
            raise ValueError(msg)
        if not node.is_leaf:
            msg = "Can only delete leaf nodes."
            raise ValueError(msg)
        if node.parent is None:
            msg = "Parent node for node not found."
            raise ValueError(msg)
        if node.parent.left_child == node:
            node.parent.left_child = None
        elif node.parent.right_child == node:
            node.parent.right_child = None
        self.nodes.remove(node)
