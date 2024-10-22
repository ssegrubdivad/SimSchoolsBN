# src/input_parsing/cpt_parser.py

import re
import logging
import numpy as np
from typing import Dict, List, Any
from src.probability_distribution import DiscreteDistribution, ContinuousDistribution, CLGDistribution

class DistributionFactory:
    @staticmethod
    def create_distribution(node_name: str, cpt_type: str, parents: List[str], params: Dict[str, Any]) -> Any:
        if cpt_type == 'DISCRETE':
            return DiscreteDistribution(node_name, parents, params['states'], params['table'])
        elif cpt_type == 'CONTINUOUS':
            return ContinuousDistribution(node_name, parents, params['distribution']['type'])
        elif cpt_type == 'CLG':
            continuous_parents = params['distribution'].get('continuous_parents', [])
            discrete_parents = [p for p in parents if p not in continuous_parents]
            clg = CLGDistribution(node_name, continuous_parents, discrete_parents)
            clg.set_parameters({"parameters": params['distribution']})
            return clg
        else:
            raise ValueError(f"Unknown CPT type: {cpt_type}")

class CPTParser:
    def __init__(self, network):
        self.cpts = {}
        self.metadata = {}
        self.logger = logging.getLogger(__name__)
        self.node_states = {node.id: node.states for node in network.nodes.values()}
        self.logger.debug(f"Initialized CPTParser with node states: {self.node_states}")
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            self.logger.info(f"Successfully read file: {file_path}")
        except IOError as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

        try:
            self._parse_metadata(content)
            self.logger.info(f"Successfully parsed metadata: {self.metadata}")
            self._parse_cpts(content)
        except ValueError as e:
            self.logger.error(f"Error parsing CPT file: {str(e)}")
            raise

        if not self.cpts:
            raise ValueError("No valid CPTs were parsed from the file.")

        self.logger.info(f"Successfully parsed {len(self.cpts)} CPTs")
        return self.cpts

    def _parse_cpts(self, content: str) -> None:
        cpt_pattern = r'^[ \t]*CPT\s+([^\s]+)(.*?)(?=^[ \t]*CPT\s+[^\s]+|\Z)'
        matches = list(re.finditer(cpt_pattern, content, re.DOTALL | re.MULTILINE))
        self.logger.info(f"Found {len(matches)} CPT definitions in the file")
        
        for match in matches:
            node_id, cpt_content = match.groups()
            if node_id.lower() == 'definitions':
                self.logger.warning(f"Skipping invalid node name: {node_id}")
                continue
            
            self.logger.info(f"Parsing CPT for node: {node_id}")
            try:
                parsed_cpt = self._parse_cpt_content(node_id, cpt_content)
                if parsed_cpt:
                    self.cpts[node_id] = parsed_cpt
                    self.logger.info(f"Successfully parsed CPT for node: {node_id}")
            except ValueError as e:
                self.logger.error(f"Error parsing CPT for node {node_id}: {str(e)}")
                self.logger.error(f"CPT content for node {node_id}: {cpt_content}")
                raise  # Stop parsing immediately when an error is encountered

        if not self.cpts:
            raise ValueError("No valid CPTs were parsed from the file.")

        self.logger.info(f"Successfully parsed CPTs for {len(self.cpts)} nodes")

    def _parse_metadata(self, content: str) -> None:
        metadata_pattern = r'META\s+(\w+)\s+(.+)'
        for match in re.finditer(metadata_pattern, content):
            key, value = match.groups()
            self.metadata[key] = value
        self.logger.info(f"Parsed metadata: {self.metadata}")

    def _parse_cpt_content(self, node_id: str, content: str) -> DiscreteDistribution:
        lines = content.strip().split('\n')
        cpt_type = None
        parents = []
        states = []
        table_content = []
        in_table_section = False
        self.logger.debug(f"BEGIN _parse_cpt_content for node: {node_id}")
        self.logger.debug(f"Total lines in content: {len(lines)}")
        for line in lines:
            line = line.strip()
            self.logger.debug(f"Processing line in _parse_cpt_content: '{line}'")
            if line.startswith('TYPE'):
                cpt_type = line.split()[1]
                self.logger.debug(f"CPT type: {cpt_type}")
            elif line.startswith('PARENTS'):
                parent_line = line[len('PARENTS'):].strip()
                parents = parent_line.split()
                self.logger.debug(f"Parents: {parents}")
            elif line.startswith('STATES'):
                state_line = line[len('STATES'):].strip()
                states = [state.strip() for state in state_line.split(',') if state.strip()]
                self.logger.debug(f"States: {states}")
            elif line.startswith('TABLE'):
                in_table_section = True
                self.logger.debug("Start of TABLE section")
            elif line == 'END_TABLE':
                in_table_section = False
                self.logger.debug("End of TABLE section")
            elif in_table_section:
                table_content.append(line)
                self.logger.debug(f"Added line to table_content: '{line}'")
        
        self.logger.debug(f"Final values before calling _parse_table:")
        self.logger.debug(f"Node {node_id}: Type={cpt_type}, Parents={parents}, States={states}")
        self.logger.debug(f"table_content has {len(table_content)} lines:")
        for idx, line in enumerate(table_content):
            self.logger.debug(f"  table_content[{idx}]: '{line}'")
        
        if cpt_type != 'DISCRETE':
            raise ValueError(f"Unsupported CPT type for node {node_id}: {cpt_type}")
        if not states:
            raise ValueError(f"No states defined for node {node_id}")
        
        self.node_states[node_id] = states
        probabilities = self._parse_table(table_content, node_id, parents, states)

        return DiscreteDistribution(node_id, parents, states, probabilities, self.node_states)

    def _parse_table(self, table_lines: List[str], node_id: str, parents: List[str], states: List[str]) -> Dict[tuple, List[float]]:
        self.logger.debug(f"\nBEGIN _parse_table for node {node_id}")
        self.logger.debug(f"Received {len(table_lines)} table_lines")
        for idx, line in enumerate(table_lines):
            self.logger.debug(f"  table_lines[{idx}]: '{line}'")
        
        table = {}
        expected_entries = 1
        valid_entries = 0
        self.logger.debug(f"Parents: {parents}")
        self.logger.debug(f"States: {states}")
        self.logger.debug(f"Node states dictionary: {self.node_states}")
        
        for parent in parents:
            if parent not in self.node_states:
                self.logger.error(f"Parent node {parent} not found in self.node_states")
                raise ValueError(f"Parent node {parent} not found for node {node_id}")
            parent_states = self.node_states[parent]
            self.logger.debug(f"Parent {parent} has states: {parent_states}")
            expected_entries *= len(parent_states)

        self.logger.debug(f"Node {node_id}: Expected {expected_entries} entries")
        
        for line_num, line in enumerate(table_lines, 1):
            line = line.strip()
            self.logger.debug(f"Processing line {line_num}: '{line}'")
            if line.startswith('#') or line == '':
                self.logger.debug(f"Skipping line {line_num} (comment or empty)")
                continue  # Skip comment lines and empty lines
            try:
                parts = line.split('|')
                self.logger.debug(f"Split line into {len(parts)} parts: {parts}")
                if len(parents) == 0:  # No parents
                    probabilities = [float(p.strip()) for p in line.split(',') if p.strip()]
                    if len(probabilities) != len(states):
                        raise ValueError(f"Line {line_num}: Expected {len(states)} probabilities, got {len(probabilities)}")
                    table[()] = probabilities
                    valid_entries += 1
                    self.logger.debug(f"Parsed entry for node with no parents: {probabilities}")
                elif len(parts) == 2:  # Parents and probabilities separated by |
                    parent_values = tuple(v.strip() for v in parts[0].split(',') if v.strip())
                    probabilities = [float(p.strip()) for p in parts[1].split(',') if p.strip()]
                    self.logger.debug(f"Parent values: {parent_values}")
                    self.logger.debug(f"Probabilities: {probabilities}")
                    if len(parent_values) != len(parents):
                        raise ValueError(f"Line {line_num}: Mismatch in number of parent values. Expected {len(parents)}, got {len(parent_values)}.")
                    if len(probabilities) != len(states):
                        raise ValueError(f"Line {line_num}: Expected {len(states)} probabilities, got {len(probabilities)}")
                    table[parent_values] = probabilities
                    valid_entries += 1
                    self.logger.debug(f"Added entry to table with parent_values={parent_values}, probabilities={probabilities}")
                    self.logger.debug(f"Current valid_entries count: {valid_entries}")
                else:
                    raise ValueError(f"Line {line_num}: Invalid format. Expected either a single probability list or parent values | probability list.")
            except ValueError as e:
                self.logger.error(f"Error parsing line {line_num}: {str(e)}")
                raise ValueError(f"Error parsing values on line {line_num}: {str(e)}")

        self.logger.debug(f"Finished processing all lines")
        self.logger.debug(f"Final table contents: {table}")
        self.logger.debug(f"Final valid_entries count: {valid_entries}")
        
        if valid_entries != expected_entries:
            noParents = ""
            if len(parents) == 1:
                noParents = "parent"
            else:
                noParents = "parents"
            error_message = (
                f"An incorrect number of combination entries were found in the CPT for node {node_id}. Because this node has {len(parents)} {noParents}, we calculate that it should contain {expected_entries} combination entries.\n\n"
                f"We expected to find exactly {expected_entries} combination entries, but your file provided {valid_entries} for this node.\n\n"
                f"Please provide all {expected_entries} combination entries. Remember that simplified or incomplete CPTs are not supported."
            )
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        return table
        
    def _parse_distribution(self, distribution_lines: List[str]) -> Dict[str, Any]:
        distribution = {}
        for line in distribution_lines:
            if line.strip() == '':
                continue
            try:
                key, value = line.split('=')
                distribution[key.strip()] = eval(value.strip())
            except (ValueError, SyntaxError) as e:
                self.logger.error(f"Error parsing distribution line '{line}': {str(e)}")
                raise ValueError(f"Invalid distribution format in line: {line}")
        return distribution

    def get_metadata(self) -> Dict[str, str]:
        return self.metadata

