# src/probability_distribution/discrete_distribution.py

from typing import List, Dict, Any
import random
import numpy as np
import logging
from .distribution import Distribution

class DiscreteDistribution(Distribution):
    def __init__(self, variable: str, parents: List[str], states: List[str], probabilities: Dict[tuple, List[float]], node_states: Dict[str, List[str]]):
        super().__init__(variable, parents)
        self.logger = logging.getLogger(__name__)
        self.states = states
        self.probabilities = probabilities
        self.node_states = node_states
        self.logger.debug(f"Initializing DiscreteDistribution for {variable}")
        self.logger.debug(f"Parents: {parents}")
        self.logger.debug(f"States: {states}")
        self.logger.debug(f"Probabilities keys: {list(probabilities.keys())}")
        self._validate_probabilities()

    def _validate_probabilities(self):
        expected_combinations = 1
        for parent in self.parents:
            parent_states = self.node_states.get(parent)
            if parent_states is None:
                raise ValueError(f"Missing states for parent '{parent}' in variable '{self.variable}'")
            expected_combinations *= len(parent_states)
        
        self.logger.debug(f"Variable: {self.variable}, Parents: {self.parents}")
        self.logger.debug(f"Expected combinations: {expected_combinations}")
        self.logger.debug(f"Actual combinations: {len(self.probabilities)}")

        if len(self.probabilities) != expected_combinations:
            raise ValueError(f"Incomplete CPT for variable '{self.variable}'. Expected {expected_combinations} probability entries, but got {len(self.probabilities)}.")

        for prob_list in self.probabilities.values():
            if len(prob_list) != len(self.states):
                raise ValueError(f"Invalid probability list for variable '{self.variable}'. Expected {len(self.states)} probabilities, but got {len(prob_list)}.")
            if abs(sum(prob_list) - 1.0) > 1e-10:
                raise ValueError(f"Probabilities for variable '{self.variable}' do not sum to 1.")
                                                        
    def get_probability(self, value: Any, parent_values: Dict[str, Any]) -> float:
        if value not in self.values:
            raise ValueError(f"Value '{value}' is not in the list of possible values for variable '{self.variable}'.")
        
        parent_combination = tuple(parent_values[parent] for parent in self.parents)
        if parent_combination not in self.probabilities:
            raise ValueError(f"No probability defined for parent combination {parent_combination} for variable '{self.variable}'. Please ensure the CPT is complete.")
        
        index = self.values.index(value)
        return self.probabilities[parent_combination][index]

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "parents": self.parents,
            "values": self.values,
            "table": np.array(list(self.probabilities.values())).T
        }

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.variable = parameters["variable"]
        self.parents = parameters["parents"]
        self.values = parameters["values"]
        self.probabilities = parameters["probabilities"]
        self._validate_probabilities()

    def sample(self, parent_values: Dict[str, Any]) -> Any:
        parent_combination = tuple(parent_values[parent] for parent in self.parents)
        if parent_combination not in self.probabilities:
            raise ValueError(f"No probability defined for parent combination {parent_combination} for variable '{self.variable}'. Please ensure the CPT is complete.")
        
        probs = self.probabilities[parent_combination]
        return np.random.choice(self.values, p=probs)