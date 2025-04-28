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
        id="root",
        parent=None,
        logodds=dataset.get_logodds(),
        data_ids=dataset.ids,
    )
    return Tree(
        nodes=[
            root_node,
        ],
        root=root_node,
        df=dataset,
    )


def split_on_feature(tree: Tree, feature_name: str) -> None:
    """Split the root node on the specified feature."""
    threshold, _ = suggest_split_threshold(tree.df, feature_name)
    tree.split_node(
        node_id=tree.root.id,
        threshold=threshold,
        feature_name=feature_name,
    )


def main() -> None:
    """Main function to create and split a tree."""
    tree = initialize_tree()
    split_on_feature(tree, "Glucose")
    print(tree)


if __name__ == "__main__":
    main()
