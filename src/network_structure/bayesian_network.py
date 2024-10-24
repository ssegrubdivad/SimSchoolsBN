# src/network_structure/bayesian_network.py

from typing import List, Dict, Tuple, Set, Any
from .node import Node
from .edge import Edge

class BayesianNetwork:
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.cpds: Dict[str, Any] = {}

    def add_node(self, node: Node) -> None:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        else:
            raise ValueError(f"Node with id {node.id} already exists in the network.")

    def add_edge(self, parent_id: str, child_id: str) -> None:
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise ValueError("Both parent and child nodes must exist in the network.")
        
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        
        edge = Edge(parent, child)
        self.edges.append(edge)
        child.add_parent(parent)

        if self.has_cycle():
            self.remove_edge(parent_id, child_id)
            raise ValueError("Adding this edge would create a cycle in the graph.")

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} does not exist in the network.")
        
        node = self.nodes[node_id]
        
        # Remove all edges connected to this node
        self.edges = [edge for edge in self.edges if edge.parent != node and edge.child != node]
        
        # Remove the node from its parents' children lists and its children's parent lists
        for parent in node.parents:
            parent.children.remove(node)
        for child in node.children:
            child.parents.remove(node)
        
        del self.nodes[node_id]

    def remove_edge(self, parent_id: str, child_id: str) -> None:
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        self.edges = [edge for edge in self.edges if edge.parent != parent or edge.child != child]
        child.remove_parent(parent)

    def get_nodes(self):
        return list(self.nodes.values())

    def get_edges(self):
        return self.edges

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def is_cyclic(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for edge in self.edges:
                if edge.parent.id == node_id:
                    neighbor = edge.child.id
                    if neighbor not in visited:
                        if is_cyclic(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if is_cyclic(node_id):
                    return True
        return False

    def get_ancestors(self, node_id: str) -> Set[str]:
        ancestors = set()
        def dfs(current_id: str):
            for edge in self.edges:
                if edge.child.id == current_id and edge.parent.id not in ancestors:
                    ancestors.add(edge.parent.id)
                    dfs(edge.parent.id)
        dfs(node_id)
        return ancestors

    def get_descendants(self, node_id: str) -> Set[str]:
        descendants = set()
        def dfs(current_id: str):
            for edge in self.edges:
                if edge.parent.id == current_id and edge.child.id not in descendants:
                    descendants.add(edge.child.id)
                    dfs(edge.child.id)
        dfs(node_id)
        return descendants

    def check_model(self) -> bool:
        # Implement basic model checking logic
        # For now, we'll just check if the graph is acyclic
        return not self.has_cycle()

    def add_cpds(self, *cpds):
        for cpd in cpds:
            node_id = cpd.variable
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} does not exist in the network")
            self.cpds[node_id] = cpd
            self.nodes[node_id].set_distribution(cpd)

    def get_cpds(self, node_id: str = None):
        if node_id is None:
            return list(self.cpds.values())
        return self.cpds.get(node_id)

    def remove_cpds(self, *node_ids):
        for node_id in node_ids:
            if node_id in self.cpds:
                del self.cpds[node_id]
                self.nodes[node_id].set_distribution(None)
        
    def __str__(self) -> str:
        return f"BayesianNetwork(name={self.name}, nodes={len(self.nodes)}, edges={len(self.edges)})"

    def __repr__(self) -> str:
        return self.__str__()