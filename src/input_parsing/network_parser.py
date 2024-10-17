# src/input_parsing/network_parser.py

import re
import logging
from typing import Dict, List, Tuple
from src.network_structure import BayesianNetwork, Node, DynamicBayesianNetwork

class NetworkParser:
    def __init__(self):
        self.network = None
        self.metadata = {}
        self.nodes = {}
        self.edges = []
        self.dbn_timeslices = []
        self.logger = logging.getLogger(__name__)

    def parse_file(self, file_path: str) -> BayesianNetwork:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except IOError as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

        try:
            self._parse_content(lines)
            self._create_network()
        except ValueError as e:
            self.logger.error(f"Error parsing network structure: {str(e)}")
            raise

        return self.network

    def _parse_content(self, lines: List[str]) -> None:
        current_node = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('META'):
                self._parse_metadata(line)
            elif line.startswith('NODE'):
                current_node = self._parse_node(line)
            elif line.startswith('EDGE'):
                self._parse_edge(line)
            elif current_node and line not in ['END_NETWORK', 'END_CPT', 'END_CPT_FILE']:
                self._parse_node_details(current_node, line)

    def _parse_metadata(self, line: str) -> None:
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            _, key, value = parts
            self.metadata[key] = value
        self.logger.info(f"Parsed metadata: {self.metadata}")

    def _parse_node(self, line: str) -> str:
        parts = line.split()
        if len(parts) == 3:  # NODE name type
            _, node_id, variable_type = parts
            self.logger.info(f"Parsing node: id='{node_id}', type='{variable_type}'")
            try:
                node = Node(node_id, node_id, variable_type)
                self.nodes[node_id] = {
                    'node': node,
                    'details': {}
                }
                self.logger.info(f"Successfully created node: {node_id}")
                return node_id
            except ValueError as e:
                self.logger.error(f"Error creating node {node_id}: {str(e)}")
                raise
        else:
            self.logger.error(f"Invalid NODE line: {line}")
            raise ValueError(f"Invalid NODE line: {line}")

    def _parse_node_details(self, node_id: str, line: str) -> None:
        if node_id in self.nodes:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                if key == 'STATES':
                    states = [s.strip() for s in value.split(',')]
                    self.nodes[node_id]['node'].add_states(states)
                self.nodes[node_id]['details'][key] = value
            else:
                self.logger.warning(f"Skipping invalid node detail line for {node_id}: {line}")

    def _parse_edge(self, line: str) -> None:
        parts = line.split()
        if len(parts) == 3:
            _, parent, child = parts
            self.edges.append((parent, child))
        else:
            self.logger.warning(f"Skipping invalid EDGE line: {line}")

    def _parse_dbn_timeslices(self, content: str) -> None:
        dbn_pattern = r'DBN_TIMESLICE(.*?)(?=DBN_TIMESLICE|\Z)'
        for match in re.finditer(dbn_pattern, content, re.DOTALL):
            timeslice = match.group(1).strip()
            self.dbn_timeslices.append(self._parse_timeslice(timeslice))
        self.logger.info(f"Parsed {len(self.dbn_timeslices)} DBN timeslices")

    def _parse_timeslice(self, timeslice: str) -> Dict[str, List[Tuple[str, str]]]:
        intra_edges = []
        inter_edges = []
        for line in timeslice.split('\n'):
            if line.startswith('INTRA_SLICE_EDGES'):
                intra_edges = self._parse_edge_list(line)
            elif line.startswith('INTER_SLICE_EDGES'):
                inter_edges = self._parse_edge_list(line)
        return {'intra_slice_edges': intra_edges, 'inter_slice_edges': inter_edges}

    def _parse_edge_list(self, line: str) -> List[Tuple[str, str]]:
        edge_pattern = r'EDGE\s+(\w+)\s+(\w+)'
        return re.findall(edge_pattern, line)

    def _create_network(self) -> None:
        self.network = BayesianNetwork(self.metadata.get('network_name', 'Unnamed Network'))
        
        for node_id, node_info in self.nodes.items():
            self.network.add_node(node_info['node'])

        for parent_id, child_id in self.edges:
            try:
                self.network.add_edge(parent_id, child_id)
            except ValueError as e:
                self.logger.error(f"Error adding edge {parent_id} -> {child_id}: {str(e)}")

        self.logger.info(f"Created Bayesian Network with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def get_metadata(self) -> Dict[str, str]:
        return self.metadata

    def get_nodes(self) -> Dict[str, Dict[str, str]]:
        return self.nodes

    def get_edges(self) -> List[Tuple[str, str]]:
        return self.edges