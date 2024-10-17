# src/network_structure/node.py

import logging
from typing import List, Dict, Any, Set
from src.probability_distribution import Distribution, DiscreteDistribution, ContinuousDistribution

class Node:
    VALID_TYPES = {"discrete", "continuous"}

    def __init__(self, id: str, name: str, variable_type: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Creating node with id='{id}', name='{name}', variable_type='{variable_type}'")
        self.id = id.strip()
        self.name = name.strip()
        self.set_variable_type(variable_type)
        self.parents = []
        self.children = []
        self.states = []

    def set_variable_type(self, variable_type: str):
        original_type = variable_type
        variable_type = variable_type.lower().strip()
        self.logger.info(f"Setting variable type for node {self.id}: original='{original_type}', processed='{variable_type}'")
        if variable_type not in self.VALID_TYPES:
            self.logger.error(f"Invalid variable type for node {self.id}: '{variable_type}'. Valid types are {self.VALID_TYPES}")
            raise ValueError(f"Invalid variable type '{variable_type}' for node {self.id}. Must be one of {self.VALID_TYPES}")
        self.variable_type = variable_type
        self.logger.info(f"Variable type for node {self.id} set to '{self.variable_type}'")

    def add_states(self, states: List[str]):  # Add this method
        self.states = states
        self.logger.info(f"Added states for node {self.id}: {self.states}")
    
    def add_parent(self, parent: 'Node') -> None:
        if parent not in self.parents:
            self.parents.append(parent)
            parent.children.append(self)

    def remove_parent(self, parent: 'Node') -> None:
        if parent in self.parents:
            self.parents.remove(parent)
            parent.children.remove(self)

    def set_distribution(self, distribution: Distribution) -> None:
        if (isinstance(distribution, DiscreteDistribution) and self.variable_type != "discrete") or \
           (isinstance(distribution, ContinuousDistribution) and self.variable_type != "continuous"):
            raise ValueError("Distribution type does not match node variable type")
        self.distribution = distribution

    def get_probability(self, value: Any, parent_values: Dict[str, Any]) -> float:
        if self.distribution is None:
            raise ValueError("Distribution has not been set for this node")
        return self.distribution.get_probability(value, parent_values)

    def sample(self, parent_values: Dict[str, Any]) -> Any:
        if self.distribution is None:
            raise ValueError("Distribution has not been set for this node")
        return self.distribution.sample(parent_values)

    def get_markov_blanket(self) -> Set['Node']:
        markov_blanket = set(self.parents + self.children)
        for child in self.children:
            markov_blanket.update(child.parents)
        markov_blanket.discard(self)
        return markov_blanket

    def __str__(self) -> str:
        return f"Node(id={self.id}, name={self.name}, type={self.variable_type})"

    def __repr__(self) -> str:
        return self.__str__()