"""Helpers for keeping track of session state in streamlit."""

from dataclasses import dataclass

from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

from trees.node import Node
from trees.tree import Tree


@dataclass
class _SessionState:
    """Keeps track of the session state."""

    _tree: Tree | None = None
    _flow_state: StreamlitFlowState | None = None
    _curr_state: StreamlitFlowState | None = None

    @property
    def is_initialized(self) -> bool:
        """Whether or not the session has been initialized."""
        return self._tree is not None

    @property
    def tree(self) -> Tree:
        """Get the tree for the session."""
        if self._tree is None:
            msg = "Session tree not set."
            raise ValueError(msg)

        return self._tree

    @tree.setter
    def tree(self, tree: Tree) -> None:
        """Set the tree for the session."""
        self._tree = tree

    @property
    def flow_state(self) -> StreamlitFlowState:
        """Get the flow state for the session."""
        if self._flow_state is None:
            msg = "Session flow state not set."
            raise ValueError(msg)

        return self._flow_state

    @flow_state.setter
    def flow_state(self, flow_state: StreamlitFlowState) -> None:
        self._flow_state = flow_state

    @property
    def curr_state(self) -> StreamlitFlowState:
        """Get the current state for the session.

        TODO: just flow state should be enough, don't know why the dashboard doesn't work with just
        one of these
        """
        if self._curr_state is None:
            msg = "Session flow state not set."
            raise ValueError(msg)

        return self._curr_state

    @curr_state.setter
    def curr_state(self, curr_state: StreamlitFlowState) -> None:
        self._curr_state = curr_state


class SessionState:
    """Replacement for streamlit's session_state with type hinting."""

    _session_state: _SessionState = _SessionState()

    def __new__(cls) -> _SessionState:
        return cls._session_state


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
        (-10 if node.is_left_child else 10, node.level * 10),
        {"content": _get_node_content(node)},
        node_type="input" if node.is_root else "default",
        source_position="bottom",
        target_position="top",
    )


def _get_node_content(node: Node) -> str:
    """Get the content of a node."""
    feature_part = f"{node.feature_name} <= {node.threshold:.1f}" if node.feature_name else "Leaf"
    log_odds_part = f" (logp={node.logodds:.2f}, n={len(node.data_ids):,})"
    return f"{feature_part}{log_odds_part}"


def get_flownodes_from_nodes(nodes: list[Node]) -> list[StreamlitFlowNode]:
    """Convert a list of Nodes to a list of StreamlitFlowNodes."""
    return [get_flownode_from_node(node) for node in nodes]


def update_session_state(tree: Tree) -> None:
    """Update the session state with the current tree and flow state."""
    SessionState().tree = tree
    flow_nodes = get_flownodes_from_nodes(SessionState().tree.nodes)
    flow_edges = get_edges_from_nodes(SessionState().tree.nodes)
    SessionState().flow_state = StreamlitFlowState(flow_nodes, flow_edges)
