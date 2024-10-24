# src/probability_distribution/continuous_distribution.py

from typing import List, Dict, Any
import math
import random
from scipy.stats import truncnorm
from .distribution import Distribution

class ContinuousDistribution(Distribution):
    def __init__(self, variable: str, parents: List[str], distribution_type: str = "gaussian"):
        super().__init__(variable, parents)
        self.distribution_type = distribution_type
        if distribution_type not in ["gaussian", "truncated_gaussian"]:
            raise ValueError("Only Gaussian and truncated Gaussian distributions are currently supported")
        self.parameters: Dict[tuple, Dict[str, float]] = {}

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters = parameters["parameters"]
        if self.distribution_type == "gaussian":
            for param_dict in self.parameters.values():
                if "mean" not in param_dict or "variance" not in param_dict:
                    raise ValueError("Gaussian distribution requires 'mean' and 'variance' parameters")
        elif self.distribution_type == "truncated_gaussian":
            for param_dict in self.parameters.values():
                if not all(key in param_dict for key in ["mean", "variance", "lower", "upper"]):
                    raise ValueError("Truncated Gaussian distribution requires 'mean', 'variance', 'lower', and 'upper' parameters")

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
        
        params = self.parameters[parent_combination]
        
        if self.distribution_type == "gaussian":
            mean = params["mean"]
            variance = params["variance"]
            return (1 / (math.sqrt(2 * math.pi * variance))) * math.exp(-((value - mean) ** 2) / (2 * variance))
        else:  # truncated_gaussian
            mean = params["mean"]
            std_dev = math.sqrt(params["variance"])
            lower = params["lower"]
            upper = params["upper"]
            
            # Calculate truncated normal parameters
            a = (lower - mean) / std_dev
            b = (upper - mean) / std_dev
            
            # Use scipy's truncnorm for proper handling of truncated normal distribution
            return truncnorm.pdf(value, a, b, loc=mean, scale=std_dev)

    def sample(self, parent_values: Dict[str, Any]) -> float:
        parent_combination = tuple(parent_values[parent] for parent in self.parents)
        if parent_combination not in self.parameters:
            raise ValueError(f"No parameters defined for parent combination {parent_combination}")
        
        params = self.parameters[parent_combination]
        
        if self.distribution_type == "gaussian":
            mean = params["mean"]
            std_dev = math.sqrt(params["variance"])
            return random.gauss(mean, std_dev)
        else:  # truncated_gaussian
            mean = params["mean"]
            std_dev = math.sqrt(params["variance"])
            lower = params["lower"]
            upper = params["upper"]
            
            # Calculate truncated normal parameters
            a = (lower - mean) / std_dev
            b = (upper - mean) / std_dev
            
            # Use scipy's truncnorm for sampling
            return truncnorm.rvs(a, b, loc=mean, scale=std_dev)