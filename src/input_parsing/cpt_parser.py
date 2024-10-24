# src/input_parsing/cpt_parser.py

import re
import logging
import numpy as np
from typing import Dict, List, Any
from src.probability_distribution.distribution import Distribution
from src.probability_distribution import (
    DiscreteDistribution,
    ContinuousDistribution,
    CLGDistribution
)

class DistributionFactory:
    @staticmethod
    def create_distribution(node_name: str, cpt_type: str, parents: List[str], params: Dict[str, Any]) -> Distribution:
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
    def __init__(self, model):
        """
        Initialize CPT Parser with a Bayesian Network model.
        
        Args:
            model: BayesianNetwork instance containing node and structure information
        """
        self.cpts = {}
        self.metadata = {}
        self.logger = logging.getLogger(__name__)
        self.model = model  # Store the model reference
        self.node_states = {node.id: node.states for node in model.nodes.values()
                           if node.variable_type == 'discrete'}  # Only store states for discrete nodes
        self.parent_types = {}  # Track parent variable types
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

    def _parse_cpt_content(self, node_id: str, content: str) -> Distribution:
        """Parse the content of a CPT definition."""
        lines = content.strip().split('\n')
        cpt_type = None
        parents = []
        states = []
        table_content = []
        distribution_content = []
        in_table_section = False
        in_distribution_section = False
        
        self.logger.debug(f"BEGIN _parse_cpt_content for node: {node_id}")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('TYPE'):
                cpt_type = line.split()[1]
                self.logger.debug(f"CPT type: {cpt_type}")
            elif line.startswith('PARENTS'):
                parent_line = line[len('PARENTS'):].strip()
                parents = parent_line.split() if parent_line else []
                self.logger.debug(f"Parents: {parents}")
            elif line.startswith('STATES'):
                state_line = line[len('STATES'):].strip()
                states = [state.strip() for state in state_line.split(',') if state.strip()]
                self.logger.debug(f"States: {states}")
            elif line.startswith('TABLE'):
                in_table_section = True
                in_distribution_section = False
                self.logger.debug("Start of TABLE section")
            elif line == 'END_TABLE':
                in_table_section = False
            elif line.startswith('DISTRIBUTION'):
                in_table_section = False
                in_distribution_section = True
                self.logger.debug("Start of DISTRIBUTION section")
            elif in_table_section:
                table_content.append(line)
            elif in_distribution_section:
                distribution_content.append(line)
        
        if cpt_type == 'DISCRETE':
            if not states:
                raise ValueError(f"No states defined for discrete node {node_id}")
            self.node_states[node_id] = states
            probabilities = self._parse_table(table_content, node_id, parents, states)
            return DiscreteDistribution(node_id, parents, states, probabilities, self.node_states)
        elif cpt_type == 'CONTINUOUS':
            return self._parse_continuous_distribution(node_id, parents, distribution_content)
        elif cpt_type == 'CLG':
            return self._parse_clg_distribution(node_id, parents, distribution_content)
        else:
            raise ValueError(f"Unsupported CPT type for node {node_id}: {cpt_type}")

    def _parse_continuous_distribution(self, node_id: str, parents: List[str], distribution_lines: List[str]) -> ContinuousDistribution:
        """Parse a continuous distribution specification."""
        distribution_params = {}
        
        # Parse the distribution parameters
        for line in distribution_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                key, value = [part.strip() for part in line.split('=')]
                # Handle different parameter types
                try:
                    distribution_params[key] = eval(value)
                except:
                    distribution_params[key] = value
            except ValueError as e:
                raise ValueError(f"Invalid distribution parameter format in line: {line}")
        
        # Create and initialize the distribution
        dist = ContinuousDistribution(node_id, parents, 
                                    distribution_params.get('type', 'gaussian'))
        
        # Set the parameters in the format expected by ContinuousDistribution
        dist.set_parameters({"parameters": {(): distribution_params}})
        
        return dist

    def _parse_clg_distribution(self, node_id: str, parents: List[str], distribution_lines: List[str]) -> CLGDistribution:
        """Parse a CLG distribution specification."""
        distribution_params = {}
        
        # Parse the distribution parameters
        for line in distribution_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                key, value = [part.strip() for part in line.split('=')]
                # Handle different parameter types
                if key == 'continuous_parents':
                    # Parse list of continuous parents
                    value = [p.strip() for p in value.split(',')]
                elif key == 'coefficients':
                    # Parse list of coefficients
                    value = eval(value)  # Safely evaluate list literal
                else:
                    # Parse numeric values
                    try:
                        value = float(value)
                    except ValueError:
                        value = value.strip()
                distribution_params[key] = value
            except ValueError as e:
                raise ValueError(f"Invalid CLG parameter format in line: {line}")
        
        # Validate required parameters
        required_params = ['continuous_parents', 'mean_base', 'coefficients', 'variance']
        missing_params = [param for param in required_params if param not in distribution_params]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for CLG distribution in node {node_id}: "
                f"{', '.join(missing_params)}"
            )
        
        # Create CLG distribution
        continuous_parents = distribution_params['continuous_parents']
        discrete_parents = [p for p in parents if p not in continuous_parents]
        clg = CLGDistribution(node_id, continuous_parents, discrete_parents)
        
        # Set parameters
        clg.set_parameters({"parameters": distribution_params})
        
        return clg

    def _parse_table(self, table_lines: List[str], node_id: str, parents: List[str], states: List[str]) -> Dict[tuple, List[float]]:
        self.logger.debug(f"\nBEGIN _parse_table for node {node_id}")
        self.logger.debug(f"Received {len(table_lines)} table_lines")
        for idx, line in enumerate(table_lines):
            self.logger.debug(f"  table_lines[{idx}]: '{line}'")
        
        table = {}
        expected_entries = self._calculate_expected_entries(parents, node_id)  # New method call
        valid_entries = 0
        self.logger.debug(f"Parents: {parents}")
        self.logger.debug(f"States: {states}")
        self.logger.debug(f"Node states dictionary: {self.node_states}")

        for line_num, line in enumerate(table_lines, 1):
            line = line.strip()
            self.logger.debug(f"Processing line {line_num}: '{line}'")
            if line.startswith('#') or line == '':
                self.logger.debug(f"Skipping line {line_num} (comment or empty)")
                continue

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
                    # Parse parent values with range support immediately
                    parent_values_raw = parts[0].strip()
                    
                    # Handle the case where there are no parents but a '|' in the line
                    if not parent_values_raw and len(parents) == 0:
                        probabilities = [float(p.strip()) for p in parts[1].split(',') if p.strip()]
                        table[()] = probabilities
                        valid_entries += 1
                        continue

                    # Split and process parent values, preserving ranges
                    parsed_parent_values = []
                    in_range = False
                    current_range = []
                    current_value = ''
                    
                    for char in parent_values_raw:
                        if char == '(':
                            in_range = True
                        elif char == ')':
                            in_range = False
                            if current_value:
                                try:
                                    range_parts = [float(x.strip()) for x in current_value.split(',')]
                                    if len(range_parts) != 2:
                                        raise ValueError(
                                            f"Error in CPT for node '{node_id}' on line {line_num}:\n"
                                            f"Invalid range specification '({current_value})'. Expected format: (min,max)"
                                        )
                                    parsed_parent_values.append(tuple(range_parts))
                                except ValueError as e:
                                    if "could not convert string to float" in str(e):
                                        raise ValueError(
                                            f"Error in CPT for node '{node_id}' on line {line_num}:\n"
                                            f"Invalid number in range specification '({current_value})'. All range values must be numbers."
                                        )
                                    raise
                            current_value = ''
                        elif char == ',' and not in_range:
                            if current_value.strip():
                                parsed_parent_values.append(current_value.strip())
                            current_value = ''
                        else:
                            current_value += char
                    
                    # Handle any remaining value
                    if current_value.strip():
                        parsed_parent_values.append(current_value.strip())

                    parent_values = tuple(parsed_parent_values)
                    probabilities = [float(p.strip()) for p in parts[1].split(',') if p.strip()]
                    
                    self.logger.debug(f"Parent values: {parent_values}")
                    self.logger.debug(f"Probabilities: {probabilities}")
                    
                    # Validate number of parent values considering ranges as single values
                    if len(parsed_parent_values) != len(parents):
                        formatted_values = []
                        for val in parsed_parent_values:
                            if isinstance(val, tuple) and len(val) == 2:
                                formatted_values.append(f"({val[0]},{val[1]})")
                            else:
                                formatted_values.append(str(val))
                        
                        raise ValueError(
                            f"Error in CPT for node '{node_id}' on line {line_num}:\n"
                            f"Expected {len(parents)} values (one for each of the following parents: {', '.join(parents)}) "
                            f"but got the following {len(parsed_parent_values)} values: {', '.join(formatted_values)}.\n\n"
                            f"These numbers should match (either {len(parents)} and {len(parents)} "
                            f"or {len(parsed_parent_values)} and {len(parsed_parent_values)}, but not {len(parents)} and {len(parsed_parent_values)}), "
                            f"so, either there is a mistake in the parents listed or there is a mistake in the values listed in the table."
                        )

                    if len(probabilities) != len(states):
                        raise ValueError(
                            f"Error in CPT for node '{node_id}' on line {line_num}:\n"
                            f"Expected {len(states)} probabilities for states {states},\n"
                            f"but got {len(probabilities)} values: {probabilities}"
                        )

                    # Validate probability sum
                    if abs(sum(probabilities) - 1.0) > 1e-10:
                        raise ValueError(
                            f"Error in CPT for node '{node_id}' on line {line_num}:\n"
                            f"Probabilities do not sum to 1.0 (sum = {sum(probabilities)})"
                        )

                    table[parent_values] = probabilities
                    valid_entries += 1
                    self.logger.debug(f"Added entry to table with parent_values={parent_values}, probabilities={probabilities}")
                    self.logger.debug(f"Current valid_entries count: {valid_entries}")
                else:
                    raise ValueError(
                        f"Error in CPT for node '{node_id}' on line {line_num}:\n"
                        f"Invalid format. Expected either:\n"
                        f"1. A comma-separated list of probabilities (for nodes with no parents), or\n"
                        f"2. parent_values | probability_values"
                    )

            except ValueError as e:
                if "could not convert string to float" in str(e):
                    self.logger.error(f"Error parsing probabilities on line {line_num}: {str(e)}")
                    raise ValueError(
                        f"Error in CPT for node '{node_id}' on line {line_num}:\n"
                        f"Invalid probability value found. All probabilities must be numbers."
                    )
                else:
                    self.logger.error(f"Error parsing line {line_num}: {str(e)}")
                    raise

        self.logger.debug(f"Finished processing all lines")
        self.logger.debug(f"Final table contents: {table}")
        self.logger.debug(f"Final valid_entries count: {valid_entries}")
        
        if valid_entries != expected_entries:
            noParents = "parent" if len(parents) == 1 else "parents"
            error_message = (
                f"Error in CPT for node '{node_id}':\n"
                f"This node has {len(parents)} {noParents} {parents}, requiring {expected_entries} combination entries.\n"
                f"Found {valid_entries} entries in the file.\n"
                f"Please provide exactly {expected_entries} combination entries."
            )
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        return table
        
    def _calculate_expected_entries(self, parents: List[str], node_id: str) -> int:
        """Calculate expected number of CPT entries based on parent types."""
        expected_entries = 1
        for parent in parents:
            parent_node = next((node for node in self.model.nodes.values() if node.id == parent), None)
            if parent_node is None:
                self.logger.error(f"Parent node {parent} not found for node {node_id}")
                raise ValueError(f"Parent node {parent} not found for node {node_id}")
                
            if parent_node.variable_type == 'continuous':
                # For continuous parents, we expect 3 ranges by default
                # (low, medium, high ranges as specified in format)
                expected_entries *= 3
            else:
                # For discrete parents, use the number of states
                if parent not in self.node_states:
                    self.logger.error(f"States not found for discrete parent node {parent}")
                    raise ValueError(f"States not found for discrete parent node {parent}")
                expected_entries *= len(self.node_states[parent])
                
        return expected_entries

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

