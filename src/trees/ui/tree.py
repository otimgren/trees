"""Helpers for displaying trees in streamlit."""

import streamlit as st

from trees.node import Node
from trees.tree import Tree
from trees.ui.data import load_data
from trees.ui.session_state import SessionState, update_session_state


def initialize_tree() -> None:
    """Initialize a default tree structure in the session state."""
    should_reset = st.button("Reset")
    if not SessionState().is_initialized or should_reset:
        dataset = load_data("diabetes")
        root_node = Node(
            id="root",
            parent=None,
            logodds=dataset.get_logodds(),
            train_ids=dataset.ids,
        )
        tree = Tree(
            nodes=[
                root_node,
            ],
            root=root_node,
            dataset=dataset,
        )
        update_session_state(tree)
