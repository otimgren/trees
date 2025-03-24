"""Helpers for displaying trees in streamlit."""

import streamlit as st

from trees.node import Node
from trees.tree import Tree
from trees.ui.session_state import update_session_state


def initialize_tree() -> None:
    """Initialize a default tree structure in the session state."""
    if "tree" not in st.session_state:
        root_node = Node(id="root", parent=None)
        tree = Tree(
            nodes=[
                root_node,
            ],
            root=root_node,
        )
        # tree.split_node("root", "feature_name", 0.5)
        update_session_state(tree)
