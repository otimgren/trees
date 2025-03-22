from uuid import uuid4

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import RadialLayout, TreeLayout
from streamlit_flow.state import StreamlitFlowState


def initialize_tree() -> None:
    """Initialize a default tree structure in the session state."""
    if "curr_state" not in st.session_state:
        nodes = [
            StreamlitFlowNode(
                id="0",
                pos=(0, 0),
                data={"content": "Root", "level": 0},
                node_type="input",
                source_position="bottom",
            ),
            StreamlitFlowNode(
                id="1",
                pos=(1, 0),
                data={"content": "Left-1", "level": 1},
                node_type="default",
                source_position="bottom",
                target_position="top",
            ),
            StreamlitFlowNode(
                id="2",
                pos=(2, 0),
                data={"content": "Right-1", "level": 1},
                node_type="default",
                source_position="bottom",
                target_position="top",
            ),
        ]

        edges = [
            StreamlitFlowEdge(
                "0-1",
                "0",
                "1",
                animated=True,
                marker_start={},
                marker_end={"type": "arrow"},
            ),
            StreamlitFlowEdge("0-2", "0", "2", animated=True),
        ]

        st.session_state.curr_state = StreamlitFlowState(nodes, edges)


def split_selected_node() -> None:
    """Split the selected node into two new nodes when the button is pressed."""
    if st.button("Split selected node"):
        selected_node = st.session_state.curr_state.get_node_by_id(
            st.session_state.curr_state.selected_id
        )
        if selected_node is None:
            return

        left_node = StreamlitFlowNode(
            str(f"st-flow-node_{uuid4()}"),
            (0, 0),
            {"content": "Left node"},
            node_type="default",
            source_position="bottom",
            target_position="top",
        )
        right_node = StreamlitFlowNode(
            str(f"st-flow-node_{uuid4()}"),
            (0, 0),
            {"content": "Right node"},
            node_type="default",
            source_position="bottom",
            target_position="top",
        )

        st.session_state.curr_state.nodes.extend([left_node, right_node])
        st.session_state.curr_state.edges.extend(
            [
                StreamlitFlowEdge(
                    f"{selected_node.id}-{left_node.id}",
                    selected_node.id,
                    left_node.id,
                    animated=True,
                ),
                StreamlitFlowEdge(
                    f"{selected_node.id}-{right_node.id}",
                    selected_node.id,
                    right_node.id,
                    animated=True,
                ),
            ],
        )
        st.rerun()


def delete_selected_node() -> None:
    """Delete the selected node when the button is pressed."""
    if st.button("Delete Selected Node"):
        id_to_delete = st.session_state.curr_state.selected_id
        st.session_state.curr_state.nodes = [
            node for node in st.session_state.curr_state.nodes if node.id != id_to_delete
        ]
        st.session_state.curr_state.edges = [
            edge
            for edge in st.session_state.curr_state.edges
            if id_to_delete not in (edge.source, edge.target)
        ]
        st.rerun()


def main():
    st.set_page_config("Streamlit Flow Example", layout="wide")
    st.title("Streamlit Flow Example")
    initialize_tree()

    st.session_state.curr_state = streamlit_flow(
        "example_flow",
        st.session_state.curr_state,
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

    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     for node in st.session_state.curr_state.nodes:
    #         st.write(node)

    # with col2:
    #     for edge in st.session_state.curr_state.edges:
    #         st.write(edge)

    # with col3:
    #     st.write(st.session_state.curr_state.selected_id)


if __name__ == "__main__":
    main()
