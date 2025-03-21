import random
from uuid import uuid4

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import RadialLayout, TreeLayout
from streamlit_flow.state import StreamlitFlowState


def main():
    st.set_page_config("Streamlit Flow Example", layout="wide")
    st.title("Streamlit Flow Example")

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

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Split selected node"):
            selected_node = st.session_state.curr_state.get_node_by_id(
                st.session_state.curr_state.selected_id
            )

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

    with col2:
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

    with col3:
        if st.button("Add Random Edge") and len(st.session_state.curr_state.nodes) > 1:
            source_candidates = [
                streamlit_node
                for streamlit_node in st.session_state.curr_state.nodes
                if streamlit_node.type in ["input", "default"]
            ]
            target_candidates = [
                streamlit_node
                for streamlit_node in st.session_state.curr_state.nodes
                if streamlit_node.type in ["default", "output"]
            ]
            source = random.choice(source_candidates)
            target = random.choice(target_candidates)
            new_edge = StreamlitFlowEdge(
                f"{source.id}-{target.id}", source.id, target.id, animated=True
            )
            if not any(edge.id == new_edge.id for edge in st.session_state.curr_state.edges):
                st.session_state.curr_state.edges.append(new_edge)
            st.rerun()

    with col4:
        if st.button("Delete Random Edge") and len(st.session_state.curr_state.edges) > 0:
            edge_to_delete = random.choice(st.session_state.curr_state.edges)
            st.session_state.curr_state.edges = [
                edge for edge in st.session_state.curr_state.edges if edge.id != edge_to_delete.id
            ]
            st.rerun()

    with col5:
        if st.button("Random Flow"):
            nodes = [
                StreamlitFlowNode(
                    str(f"st-flow-node_{uuid4()}"),
                    (0, 0),
                    {"content": f"Node {i}"},
                    "default",
                    "right",
                    "left",
                )
                for i in range(5)
            ]
            edges = []
            for _ in range(5):
                source = random.choice(nodes)
                target = random.choice(nodes)
                if source.id != target.id:
                    new_edge = StreamlitFlowEdge(
                        f"{source.id}-{target.id}", source.id, target.id, animated=True
                    )
                    if not any(edge.id == new_edge.id for edge in edges):
                        edges.append(new_edge)
            st.session_state.curr_state = StreamlitFlowState(nodes=nodes, edges=edges)
            st.rerun()

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

    col1, col2, col3 = st.columns(3)

    with col1:
        for node in st.session_state.curr_state.nodes:
            st.write(node)

    with col2:
        for edge in st.session_state.curr_state.edges:
            st.write(edge)

    with col3:
        st.write(st.session_state.curr_state.selected_id)


if __name__ == "__main__":
    main()
