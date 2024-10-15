# src/input_parsing/cpt_parser.py

import re
import logging
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
    def __init__(self):
        self.cpts = {}
        self.metadata = {}
        self.logger = logging.getLogger(__name__)

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except IOError as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

        try:
            self._parse_metadata(content)
            self._parse_cpts(content)
        except ValueError as e:
            self.logger.error(f"Error parsing CPT file: {str(e)}")
            raise

        return self.cpts

    def _parse_metadata(self, content: str) -> None:
        metadata_pattern = r'META\s+(\w+)\s+(.+)'
        for match in re.finditer(metadata_pattern, content):
            key, value = match.groups()
            self.metadata[key] = value
        self.logger.info(f"Parsed metadata: {self.metadata}")

    def _parse_cpts(self, content: str) -> None:
        cpt_pattern = r'CPT\s+(\w+)\s+(\w+)(.*?)END_CPT'
        for match in re.finditer(cpt_pattern, content, re.DOTALL):
            node_id, node_name, cpt_content = match.groups()
            try:
                self.cpts[node_id] = self._parse_cpt_content(node_id, node_name, cpt_content)
            except ValueError as e:
                self.logger.error(f"Error parsing CPT for node {node_id}: {str(e)}")
                raise
        self.logger.info(f"Parsed CPTs for {len(self.cpts)} nodes")

    def _parse_cpt_content(self, node_id: str, node_name: str, content: str) -> Dict[str, Any]:
        lines = content.strip().split('\n')
        cpt_type = None
        parents = []
        distribution_params = {}

        for line in lines:
            if line.startswith('TYPE'):
                cpt_type = line.split()[1]
            elif line.startswith('PARENTS'):
                parents = line.split()[1:]
            elif line.startswith('STATES'):
                distribution_params['states'] = line.split()[1:]
            elif line.startswith('TABLE'):
                distribution_params['table'] = self._parse_table(lines[lines.index(line)+1:])
            elif line.startswith('DISTRIBUTION'):
                distribution_params['distribution'] = self._parse_distribution(lines[lines.index(line)+1:])

        return DistributionFactory.create_distribution(node_name, cpt_type, parents, distribution_params)

    def _parse_table(self, table_lines: List[str]) -> Dict[tuple, List[float]]:
        table = {}
        for line in table_lines:
            if line.strip() == '':
                continue
            try:
                parts = line.split('|')
                parent_values = tuple(parts[0].strip().split(','))
                probabilities = [float(p) for p in parts[1].strip().split(',')]
                table[parent_values] = probabilities
            except (ValueError, IndexError) as e:
                self.logger.error(f"Error parsing table line '{line}': {str(e)}")
                raise ValueError(f"Invalid table format in line: {line}")
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