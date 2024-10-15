# src/probability_distribution/clg_distribution.py

from typing import List, Dict, Any
import math
import random
from .distribution import Distribution

class CLGDistribution(Distribution):
    def __init__(self, variable: str, continuous_parents: List[str], discrete_parents: List[str]):
        super().__init__(variable, continuous_parents + discrete_parents)
        self.continuous_parents = continuous_parents
        self.discrete_parents = discrete_parents
        self.parameters: Dict[tuple, Dict[str, Any]] = {}

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters = parameters["parameters"]
        for param_dict in self.parameters.values():
            if "mean_base" not in param_dict or "variance" not in param_dict or "coefficients" not in param_dict:
                raise ValueError("CLG distribution requires 'mean_base', 'variance', and 'coefficients' parameters")

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "continuous_parents": self.continuous_parents,
            "discrete_parents": self.discrete_parents,
            "parameters": self.parameters
        }

    def get_probability(self, value: float, parent_values: Dict[str, Any]) -> float:
        discrete_parent_combination = tuple(parent_values[parent] for parent in self.discrete_parents)
        if discrete_parent_combination not in self.parameters:
            raise ValueError(f"No parameters defined for discrete parent combination {discrete_parent_combination}")
        
        params = self.parameters[discrete_parent_combination]
        mean = params["mean_base"]
        for i, parent in enumerate(self.continuous_parents):
            mean += params["coefficients"][i] * parent_values[parent]
        
        variance = params["variance"]
        
        return (1 / (math.sqrt(2 * math.pi * variance))) * math.exp(-((value - mean) ** 2) / (2 * variance))

    def sample(self, parent_values: Dict[str, Any]) -> float:
        discrete_parent_combination = tuple(parent_values[parent] for parent in self.discrete_parents)
        if discrete_parent_combination not in self.parameters:
            raise ValueError(f"No parameters defined for discrete parent combination {discrete_parent_combination}")
        
        params = self.parameters[discrete_parent_combination]
        mean = params["mean_base"]
        for i, parent in enumerate(self.continuous_parents):
            mean += params["coefficients"][i] * parent_values[parent]
        
        std_dev = math.sqrt(params["variance"])
        
        return random.gauss(mean, std_dev)