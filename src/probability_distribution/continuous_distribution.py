# src/probability_distribution/continuous_distribution.py

from typing import List, Dict, Any
import math
import random
from .distribution import Distribution

class ContinuousDistribution(Distribution):
    def __init__(self, variable: str, parents: List[str], distribution_type: str = "gaussian"):
        super().__init__(variable, parents)
        self.distribution_type = distribution_type
        if distribution_type != "gaussian":
            raise ValueError("Only Gaussian distribution is currently supported")
        self.parameters: Dict[tuple, Dict[str, float]] = {}

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters = parameters["parameters"]
        for param_dict in self.parameters.values():
            if "mean" not in param_dict or "variance" not in param_dict:
                raise ValueError("Gaussian distribution requires 'mean' and 'variance' parameters")

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "parents": self.parents,
            "distribution_type": self.distribution_type,
            "parameters": self.parameters
        }

    def get_probability(self, value: float, parent_values: Dict[str, Any]) -> float:
        parent_combination = tuple(parent_values[parent] for parent in self.parents)
        if parent_combination not in self.parameters:
            raise ValueError(f"No parameters defined for parent combination {parent_combination}")
        
        mean = self.parameters[parent_combination]["mean"]
        variance = self.parameters[parent_combination]["variance"]
        
        return (1 / (math.sqrt(2 * math.pi * variance))) * math.exp(-((value - mean) ** 2) / (2 * variance))

    def sample(self, parent_values: Dict[str, Any]) -> float:
        parent_combination = tuple(parent_values[parent] for parent in self.parents)
        if parent_combination not in self.parameters:
            raise ValueError(f"No parameters defined for parent combination {parent_combination}")
        
        mean = self.parameters[parent_combination]["mean"]
        std_dev = math.sqrt(self.parameters[parent_combination]["variance"])
        
        return random.gauss(mean, std_dev)