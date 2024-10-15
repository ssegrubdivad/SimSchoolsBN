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
                content = file.read()
        except IOError as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

        try:
            self._parse_metadata(content)
            self._parse_nodes(content)
            self._parse_edges(content)
            self._parse_dbn_timeslices(content)
            self._create_network()
        except ValueError as e:
            self.logger.error(f"Error parsing network structure: {str(e)}")
            raise

        return self.network

    def _parse_metadata(self, content: str) -> None:
        metadata_pattern = r'META\s+(\w+)\s+(.+)'
        for match in re.finditer(metadata_pattern, content):
            key, value = match.groups()
            self.metadata[key] = value
        self.logger.info(f"Parsed metadata: {self.metadata}")

    def _parse_nodes(self, content: str) -> None:
        node_pattern = r'NODE\s+(\w+)\s+(\w+)\s+(\w+)(.*?)(?=NODE|\Z)'
        for match in re.finditer(node_pattern, content, re.DOTALL):
            node_id, node_name, variable_type, node_details = match.groups()
            self.nodes[node_id] = {
                'name': node_name,
                'type': variable_type,
                'details': self._parse_node_details(node_details)
            }
        self.logger.info(f"Parsed {len(self.nodes)} nodes")

    def _parse_node_details(self, details: str) -> Dict[str, str]:
        parsed_details = {}
        for line in details.strip().split('\n'):
            if line.strip():
                try:
                    key, value = line.strip().split(None, 1)
                    parsed_details[key] = value
                except ValueError:
                    self.logger.warning(f"Skipping invalid node detail line: {line}")
        return parsed_details

    def _parse_edges(self, content: str) -> None:
        edge_pattern = r'EDGE\s+(\w+)\s+(\w+)'
        self.edges = re.findall(edge_pattern, content)
        self.logger.info(f"Parsed {len(self.edges)} edges")

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
        if self.dbn_timeslices:
            self.network = DynamicBayesianNetwork(self.metadata.get('network_name', 'Unnamed DBN'))
        else:
            self.network = BayesianNetwork(self.metadata.get('network_name', 'Unnamed Network'))
        
        for node_id, node_info in self.nodes.items():
            node = Node(node_id, node_info['name'], node_info['type'])
            self.network.add_node(node)

        for parent_id, child_id in self.edges:
            try:
                self.network.add_edge(parent_id, child_id)
            except ValueError as e:
                self.logger.error(f"Error adding edge {parent_id} -> {child_id}: {str(e)}")

        if isinstance(self.network, DynamicBayesianNetwork):
            for timeslice in self.dbn_timeslices:
                self.network.add_timeslice(timeslice['intra_slice_edges'], timeslice['inter_slice_edges'])

        self.logger.info(f"Created {'Dynamic ' if self.dbn_timeslices else ''}Bayesian Network with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def get_metadata(self) -> Dict[str, str]:
        return self.metadata

    def get_nodes(self) -> Dict[str, Dict[str, str]]:
        return self.nodes

    def get_edges(self) -> List[Tuple[str, str]]:
        return self.edges