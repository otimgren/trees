import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.layouts import TreeLayout

from trees.ui.node import delete_selected_node, split_selected_node
from trees.ui.session_state import SessionState
from trees.ui.tree import initialize_tree


def main():
    st.set_page_config("Streamlit Flow Example", layout="wide")
    st.title("Streamlit Flow Example")
    initialize_tree()

    SessionState().curr_state = streamlit_flow(
        key="streamlit_flow_tree",
        state=SessionState().flow_state,
        layout=TreeLayout(direction="down"),
        fit_view=True,
        height=500,
        enable_node_menu=True,
        enable_edge_menu=True,
        enable_pane_menu=True,
        get_edge_on_click=True,
        get_node_on_click=True,
        show_minimap=True,
        hide_watermark=True,
        allow_new_edges=True,
        min_zoom=0.1,
    )

    col1, col2 = st.columns(2)

    with col1:
        split_selected_node()

    with col2:
        delete_selected_node()

    col1, col2, col3 = st.columns(3)

    with col1:
        for node in SessionState().curr_state.nodes:
            st.write(node)

    with col2:
        for edge in SessionState().curr_state.edges:
            st.write(edge)

    with col3:
        st.write(SessionState().curr_state.selected_id)


if __name__ == "__main__":
    main()
