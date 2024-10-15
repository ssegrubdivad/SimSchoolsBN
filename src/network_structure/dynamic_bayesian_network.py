# src/network_structure/dynamic_bayesian_network.py

from typing import List, Dict, Tuple
from .bayesian_network import BayesianNetwork
from .node import Node
from .edge import Edge

class DynamicBayesianNetwork(BayesianNetwork):
    def __init__(self, name: str):
        super().__init__(name)
        self.timeslices: List[Dict[str, List[Tuple[str, str]]]] = []
        self.inter_slice_edges: List[Tuple[str, str]] = []

    def add_timeslice(self, intra_slice_edges: List[Tuple[str, str]], inter_slice_edges: List[Tuple[str, str]]):
        """
        Add a new time slice to the Dynamic Bayesian Network.
        
        :param intra_slice_edges: List of edges within the new time slice
        :param inter_slice_edges: List of edges connecting the new time slice to the previous one
        """
        new_timeslice = {
            'intra_slice_edges': intra_slice_edges,
            'inter_slice_edges': inter_slice_edges
        }
        self.timeslices.append(new_timeslice)
        
        # Add nodes and edges for the new time slice
        for edge in intra_slice_edges:
            self._add_node_if_not_exists(edge[0], f"{edge[0]}_{len(self.timeslices)}")
            self._add_node_if_not_exists(edge[1], f"{edge[1]}_{len(self.timeslices)}")
            self.add_edge(f"{edge[0]}_{len(self.timeslices)}", f"{edge[1]}_{len(self.timeslices)}")
        
        # Add inter-slice edges
        for edge in inter_slice_edges:
            self._add_node_if_not_exists(edge[0], f"{edge[0]}_{len(self.timeslices)-1}")
            self._add_node_if_not_exists(edge[1], f"{edge[1]}_{len(self.timeslices)}")
            self.add_edge(f"{edge[0]}_{len(self.timeslices)-1}", f"{edge[1]}_{len(self.timeslices)}")
            self.inter_slice_edges.append((f"{edge[0]}_{len(self.timeslices)-1}", f"{edge[1]}_{len(self.timeslices)}"))

    def _add_node_if_not_exists(self, node_id: str, new_node_id: str):
        if new_node_id not in self.nodes:
            new_node = Node(new_node_id, f"{node_id} (t={len(self.timeslices)})", self.nodes[node_id].variable_type)
            self.add_node(new_node)

    def get_node_at_timeslice(self, node_id: str, timeslice: int) -> Node:
        """
        Get a node at a specific time slice.
        
        :param node_id: The ID of the node
        :param timeslice: The time slice to retrieve the node from
        :return: The Node object at the specified time slice
        """
        timeslice_node_id = f"{node_id}_{timeslice}"
        return self.get_node(timeslice_node_id)

    def get_inter_slice_edges(self) -> List[Tuple[str, str]]:
        """
        Get all inter-slice edges in the Dynamic Bayesian Network.
        
        :return: A list of tuples representing inter-slice edges
        """
        return self.inter_slice_edges

    def get_intra_slice_edges(self, timeslice: int) -> List[Tuple[str, str]]:
        """
        Get all intra-slice edges for a specific time slice.
        
        :param timeslice: The time slice to get edges for
        :return: A list of tuples representing intra-slice edges
        """
        if timeslice < 0 or timeslice >= len(self.timeslices):
            raise ValueError(f"Invalid time slice: {timeslice}")
        
        return [(f"{edge[0]}_{timeslice}", f"{edge[1]}_{timeslice}") 
                for edge in self.timeslices[timeslice]['intra_slice_edges']]

    def get_num_timeslices(self) -> int:
        """
        Get the number of time slices in the Dynamic Bayesian Network.
        
        :return: The number of time slices
        """
        return len(self.timeslices)

    def __str__(self) -> str:
        return (f"DynamicBayesianNetwork(name={self.name}, "
                f"nodes={len(self.nodes)}, edges={len(self.edges)}, "
                f"timeslices={len(self.timeslices)})")

    def __repr__(self) -> str:
        return self.__str__()