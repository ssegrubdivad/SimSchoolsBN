# src/probability_distribution/continuous_distribution.py

from typing import List, Dict, Any
import math
import random
import logging
from scipy.stats import truncnorm
from .distribution import Distribution

class ContinuousDistribution(Distribution):
    def __init__(self, variable: str, parents: List[str], distribution_type: str = "gaussian"):
        super().__init__(variable, parents)
        self.logger = logging.getLogger(__name__)
        self.distribution_type = distribution_type
        if distribution_type not in ["gaussian", "truncated_gaussian"]:
            raise ValueError("Only Gaussian and truncated Gaussian distributions are currently supported")
        self.parameters: Dict[tuple, Dict[str, float]] = {}

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the parameters for the distribution.
        
        Args:
            parameters: Dictionary containing:
                parameters: Dict containing distribution parameters:
                    For Gaussian: 'mean', 'variance'
                    For Truncated Gaussian: 'mean', 'variance', 'lower', 'upper'
        """
        self.logger.debug(f"Setting parameters for {self.variable}: {parameters}")
        
        if "parameters" not in parameters:
            raise ValueError("Parameters dictionary must contain 'parameters' key")
                
        params = parameters["parameters"]
        self.logger.debug(f"Extracted params dictionary: {params}")
        
        # For nodes with no parents, we use an empty tuple as the key
        # and store all parameters under that key
        if not self.parents:
            param_dict = params[()]  # Get the inner parameters dictionary
            
            # First get the distribution type from parameters if available
            if 'type' in param_dict:
                self.distribution_type = param_dict['type']
            self.logger.debug(f"Distribution type set to: {self.distribution_type}")
            
            # Convert and validate parameters based on distribution type
            if self.distribution_type == "gaussian":
                if "mean" not in param_dict or "variance" not in param_dict:
                    raise ValueError("Gaussian distribution requires 'mean' and 'variance' parameters")
                param_dict["mean"] = float(param_dict["mean"])
                param_dict["variance"] = float(param_dict["variance"])
                if param_dict["variance"] <= 0:
                    raise ValueError("Variance must be positive")
                    
            elif self.distribution_type == "truncated_gaussian":
                required_params = ["mean", "variance", "lower", "upper"]
                missing_params = [param for param in required_params if param not in param_dict]
                if missing_params:
                    raise ValueError(
                        f"Truncated Gaussian distribution requires parameters: {', '.join(required_params)}\n"
                        f"Missing: {', '.join(missing_params)}"
                    )
                
                for param in required_params:
                    param_dict[param] = float(param_dict[param])
                    
                if param_dict["variance"] <= 0:
                    raise ValueError("Variance must be positive")
                if param_dict["lower"] >= param_dict["upper"]:
                    raise ValueError("Lower bound must be less than upper bound")
            
            self.logger.debug(f"Final param_dict: {param_dict}")
            # Store parameters under empty tuple key
            self.parameters = {(): param_dict}
            self.logger.debug(f"Stored parameters: {self.parameters}")
            
        else:
            # Handle parameters for nodes with parents
            # This would be extended for conditional distributions
            raise NotImplementedError("Parameter setting for continuous nodes with parents not yet implemented")

    def get_parameters(self) -> Dict[str, Any]:
        self.logger.debug(f"Getting parameters for {self.variable}")
        self.logger.debug(f"Current parameters: {self.parameters}")
        self.logger.debug(f"Current distribution type: {self.distribution_type}")
        
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