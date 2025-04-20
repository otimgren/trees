"""Helpers for displaying nodes in streamlit."""

import streamlit as st

from trees.split import suggest_split_threshold
from trees.ui.session_state import SessionState, update_session_state


def split_selected_node() -> None:
    """Split the selected node into two new nodes when the button is pressed."""
    st.write("Split Node")
    selected_id = SessionState().curr_state.selected_id
    feature_name = st.selectbox(
        "Feature Name",
        options=SessionState().tree.dataset.feature_names,
    )

    selected_node = SessionState().tree.get_node_by_id(selected_id) if selected_id else None
    metric, suggested_threshold = (
        suggest_split_threshold(
            dataset=SessionState().tree.dataset.get_rows_by_ids(selected_node.train_ids),
            feature=feature_name,
        )
        if selected_node
        else (None, None)
    )
    threshold = st.slider(
        "Threshold",
        value=suggested_threshold,
        min_value=float(
            SessionState()
            .tree.dataset.get_rows_by_ids(selected_node.train_ids)
            .df[feature_name]
            .min()
        )
        if selected_node
        else None,
        max_value=float(
            SessionState()
            .tree.dataset.get_rows_by_ids(selected_node.train_ids)
            .df[feature_name]
            .max()
        )
        if selected_node
        else None,
    )
    st.markdown(f"*metric* = {metric or 0:.3f}")
    submitted = st.button("Split selected node")
    if submitted:
        if selected_id is None:
            return
        SessionState().tree.split_node(
            selected_id,
            feature_name,
            threshold,
        )
        update_session_state(SessionState().tree)
        st.rerun()


def delete_selected_node() -> None:
    """Delete the selected node when the button is pressed."""
    if st.button("Delete Selected Node"):
        id_to_delete = SessionState().curr_state.selected_id
        if id_to_delete is None:
            return
        SessionState().tree.delete_node(id_to_delete)
        update_session_state(SessionState().tree)
        st.rerun()
