"""Script for creating a new tree."""

from trees.data import diabetes
from trees.df import DataFrame
from trees.node import Node
from trees.splitting.split import suggest_split_threshold
from trees.tree import Tree


def initialize_tree() -> Tree:
    """Load data and create a single tree."""
    dataset = DataFrame.from_polars(
        diabetes.load_raw_data().with_row_index(), id_col_name="index", label_col_name="Outcome"
    )
    root_node = Node(
        name="root",
        parent=None,
        logodds=dataset.get_logodds(),
        data_ids=dataset.ids,
    )
    return Tree(
        root=root_node,
        df=dataset,
    )


def split_on_feature(node: Node, tree: Tree, feature_name: str) -> None:
    """Split the root node on the specified feature."""
    _, threshold = suggest_split_threshold(tree.df.get_rows_by_ids(node.data_ids), feature_name)
    tree.split_node(
        node_id=node.name,
        threshold=threshold,
        feature_name=feature_name,
    )


def main() -> None:
    """Create and split a tree."""
    tree = initialize_tree()
    # tree.root.show(attr_list=["n_obs", "logodds"])
    split_on_feature(tree.root, tree, "Pregnancies")
    left_child = tree.root.left
    split_on_feature(left_child, tree, "BMI")
    tree.root.show(attr_list=["n_obs", "feature_name", "threshold", "logodds"])

    left_child_id = left_child.id
    tree.delete_node(left_child_id)
    print(f"Deleted node {left_child_id}")
    tree.root.show(attr_list=["n_obs", "feature_name", "threshold", "logodds"])


if __name__ == "__main__":
    main()
