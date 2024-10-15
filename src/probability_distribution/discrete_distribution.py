# src/probability_distribution/discrete_distribution.py

from typing import List, Dict, Any
import random
from .distribution import Distribution

class DiscreteDistribution(Distribution):
    def __init__(self, variable: str, parents: List[str], values: List[Any], probabilities: Dict[tuple, List[float]]):
        super().__init__(variable, parents)
        self.values = values
        self.probabilities = probabilities
        self._validate_probabilities()

    def _validate_probabilities(self):
        for prob_list in self.probabilities.values():
            if abs(sum(prob_list) - 1.0) > 1e-10:
                raise ValueError("Probabilities must sum to 1 for each parent combination")

    def get_probability(self, value: Any, parent_values: Dict[str, Any]) -> float:
        if value not in self.values:
            raise ValueError(f"Value {value} is not in the list of possible values")
        
        parent_combination = tuple(parent_values[parent] for parent in self.parents)
        if parent_combination not in self.probabilities:
            raise ValueError(f"No probability defined for parent combination {parent_combination}")
        
        index = self.values.index(value)
        return self.probabilities[parent_combination][index]

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "parents": self.parents,
            "values": self.values,
            "probabilities": self.probabilities
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
            raise ValueError(f"No probability defined for parent combination {parent_combination}")
        
        probs = self.probabilities[parent_combination]
        return random.choices(self.values, weights=probs, k=1)[0]