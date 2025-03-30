"""Helpers for keeping track of session state in streamlit."""

import time

import streamlit as st
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

from trees.node import Node
from trees.tree import Tree


def get_edges_from_node(node: Node) -> list[StreamlitFlowEdge]:
    """Get edges from a list of nodes."""
    edges = []
    if node.left_child:
        edges.append(
            StreamlitFlowEdge(
                f"{node.id}-{node.left_child.id}",
                node.id,
                node.left_child.id,
                animated=True,
            )
        )
    if node.right_child:
        edges.append(
            StreamlitFlowEdge(
                f"{node.id}-{node.right_child.id}",
                node.id,
                node.right_child.id,
                animated=True,
            )
        )
    return edges


def get_edges_from_nodes(nodes: list[Node]) -> list[StreamlitFlowEdge]:
    """Get edges from a list of nodes."""
    edges = []
    for node in nodes:
        edges.extend(get_edges_from_node(node))
    return edges


def get_flownode_from_node(node: Node) -> StreamlitFlowNode:
    """Convert a Node to a StreamlitFlowNode."""
    return StreamlitFlowNode(
        node.id,
        (-1 if node.is_left_child else 1, -node.level),
        {"content": f"{node.feature_name} <= {node.threshold}" if node.feature_name else "Leaf"},
        node_type="input" if node.is_root else "default",
        source_position="bottom",
        target_position="top",
    )


def get_flownodes_from_nodes(nodes: list[Node]) -> list[StreamlitFlowNode]:
    """Convert a list of Nodes to a list of StreamlitFlowNodes."""
    return [get_flownode_from_node(node) for node in nodes]


def update_session_state(tree: Tree) -> None:
    """Update the session state with the current tree and flow state."""
    st.session_state.tree = tree

    flow_nodes = get_flownodes_from_nodes(st.session_state.tree.nodes)
    flow_edges = get_edges_from_nodes(st.session_state.tree.nodes)
    st.session_state.curr_state = StreamlitFlowState(flow_nodes, flow_edges)
    time.sleep(0.5)
