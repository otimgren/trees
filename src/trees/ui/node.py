"""Helpers for displaying nodes in streamlit."""

import streamlit as st

from trees.ui.session_state import update_session_state


def split_selected_node() -> None:
    """Split the selected node into two new nodes when the button is pressed."""

    st.write("Split Node")
    selected_id = st.session_state.curr_state.selected_id
    feature_name = st.selectbox(
        "Feature Name",
        options=st.session_state.dataset.feature_names,
    )
    threshold = st.slider(
        "Threshold",
        min_value=st.session_state.dataset.df[feature_name].min(),
        max_value=st.session_state.dataset.df[feature_name].max(),
    )
    submitted = st.button("Split selected node")
    if submitted:
        print("submitted")
        if selected_id is None:
            return
        st.session_state.tree.split_node(selected_id, feature_name, threshold)
        update_session_state(st.session_state.tree)
        st.rerun()


def delete_selected_node() -> None:
    """Delete the selected node when the button is pressed."""
    if st.button("Delete Selected Node"):
        id_to_delete = st.session_state.curr_state.selected_id
        st.session_state.tree.delete_node(id_to_delete)
        update_session_state(st.session_state.tree)
        st.rerun()
