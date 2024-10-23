# src/probability_distribution/discrete_distribution.py

from typing import List, Dict, Any, Tuple
import random
import numpy as np
import logging
from .distribution import Distribution

class DiscreteDistribution(Distribution):
    """
    A class representing discrete probability distributions for educational variables.
    
    Attributes:
        variable (str): The name of the variable this distribution represents
        parents (List[str]): List of parent variable names
        states (List[str]): Possible states/values for this variable (e.g., "poor", "average", "good")
        probabilities (Dict[tuple, List[float]]): Conditional probability tables
        node_states (Dict[str, List[str]]): States for all nodes in the network
        logger (logging.Logger): Logger instance for this class
    """

    def __init__(self, variable: str, parents: List[str], states: List[str], 
                 probabilities: Dict[tuple, List[float]], node_states: Dict[str, List[str]]):
        """
        Initialize a discrete distribution.

        Args:
            variable: Name of the variable
            parents: List of parent variable names
            states: Possible states for this variable
            probabilities: Conditional probability tables
            node_states: Dictionary mapping node names to their possible states

        Raises:
            ValueError: If probabilities are invalid or states are empty
            KeyError: If parent nodes are missing from node_states
        """
        super().__init__(variable, parents)
        self.logger = logging.getLogger(__name__)
        
        if not states:
            raise ValueError(f"No states provided for variable '{variable}'")
        
        self.states = states
        self.probabilities = probabilities
        self.node_states = node_states
        
        self.logger.debug(f"Initializing DiscreteDistribution for {variable}")
        self.logger.debug(f"Parents: {parents}")
        self.logger.debug(f"States: {states}")
        self.logger.debug(f"Probabilities keys: {list(probabilities.keys())}")
        
        try:
            self._validate_probabilities()
        except (ValueError, KeyError) as e:
            self.logger.error(f"Validation failed for {variable}: {str(e)}")
            raise

    def _validate_probabilities(self) -> None:
        """
        Validate the probability distributions.
        
        Checks:
        1. All required parent combinations exist
        2. Probabilities sum to 1 (within floating point tolerance)
        3. Correct number of probability values for each combination
        4. All probabilities are between 0 and 1
        
        Raises:
            ValueError: If probabilities are invalid
            KeyError: If required parent states are missing
        """
        expected_combinations = 1
        for parent in self.parents:
            if parent not in self.node_states:
                raise KeyError(f"Missing states for parent '{parent}' in variable '{self.variable}'")
            expected_combinations *= len(self.node_states[parent])
        
        self.logger.debug(f"Variable: {self.variable}, Parents: {self.parents}")
        self.logger.debug(f"Expected combinations: {expected_combinations}")
        self.logger.debug(f"Actual combinations: {len(self.probabilities)}")

        if len(self.probabilities) != expected_combinations:
            raise ValueError(
                f"Incomplete CPT for variable '{self.variable}'. "
                f"Expected {expected_combinations} probability entries, "
                f"but got {len(self.probabilities)}."
            )

        for parent_values, prob_list in self.probabilities.items():
            # Check probability list length
            if len(prob_list) != len(self.states):
                raise ValueError(
                    f"Invalid probability list for variable '{self.variable}' "
                    f"with parent values {parent_values}. Expected {len(self.states)} "
                    f"probabilities, but got {len(prob_list)}."
                )
            
            # Check each probability is between 0 and 1
            if not all(0 <= p <= 1 for p in prob_list):
                raise ValueError(
                    f"Invalid probabilities for variable '{self.variable}' "
                    f"with parent values {parent_values}. All probabilities "
                    f"must be between 0 and 1."
                )
            
            # Check sum is 1 (within floating point tolerance)
            if abs(sum(prob_list) - 1.0) > 1e-10:
                raise ValueError(
                    f"Probabilities for variable '{self.variable}' with parent "
                    f"values {parent_values} do not sum to 1 "
                    f"(sum = {sum(prob_list)})."
                )

    def get_probability(self, value: Any, parent_values: Dict[str, Any]) -> float:
        """
        Get the probability of a specific value given parent values.

        Args:
            value: The value to get the probability for
            parent_values: Dictionary mapping parent names to their values

        Returns:
            float: The probability value

        Raises:
            ValueError: If value is invalid or parent values are missing
            KeyError: If parent combination doesn't exist in probability table
        """
        if value not in self.states:
            raise ValueError(
                f"Value '{value}' is not in the list of possible values for "
                f"variable '{self.variable}': {self.states}"
            )
        
        try:
            parent_combination = tuple(parent_values[parent] for parent in self.parents)
        except KeyError as e:
            raise ValueError(
                f"Missing parent value for {e} in variable '{self.variable}'. "
                f"Required parent values: {self.parents}"
            )

        if parent_combination not in self.probabilities:
            raise KeyError(
                f"No probability defined for parent combination {parent_combination} "
                f"for variable '{self.variable}'. Please ensure the CPT is complete."
            )
        
        index = self.states.index(value)
        return self.probabilities[parent_combination][index]

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the distribution parameters.

        Returns:
            Dict containing variable information, parents, states, and probability table
        """
        return {
            "variable": self.variable,
            "parents": self.parents,
            "values": self.states,
            "table": np.array(list(self.probabilities.values())).T
        }

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the distribution parameters.

        Args:
            parameters: Dictionary containing variable info, parents, states, and probabilities

        Raises:
            ValueError: If parameters are invalid
        """
        self.variable = parameters["variable"]
        self.parents = parameters["parents"]
        self.states = parameters["values"]
        self.probabilities = parameters["probabilities"]
        self._validate_probabilities()

    def sample(self, parent_values: Dict[str, Any]) -> Any:
        """
        Generate a random sample from the distribution given parent values.

        Args:
            parent_values: Dictionary mapping parent names to their values

        Returns:
            A randomly sampled state based on the conditional probabilities

        Raises:
            ValueError: If parent values are invalid
            KeyError: If parent combination doesn't exist
        """
        try:
            parent_combination = tuple(parent_values[parent] for parent in self.parents)
        except KeyError as e:
            raise ValueError(
                f"Missing parent value for {e} in variable '{self.variable}'. "
                f"Required parent values: {self.parents}"
            )

        if parent_combination not in self.probabilities:
            raise KeyError(
                f"No probability defined for parent combination {parent_combination} "
                f"for variable '{self.variable}'. Please ensure the CPT is complete."
            )
        
        probs = self.probabilities[parent_combination]
        return np.random.choice(self.states, p=probs)

    def __str__(self) -> str:
        """String representation of the distribution."""
        return (f"DiscreteDistribution(variable={self.variable}, "
                f"parents={self.parents}, states={self.states})")

    def __repr__(self) -> str:
        """Detailed string representation of the distribution."""
        return self.__str__()