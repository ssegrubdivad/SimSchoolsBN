# src/probability_distribution/clg_distribution.py

from typing import List, Dict, Any
import math
import random
import logging
from .distribution import Distribution

class CLGDistribution(Distribution):
    def __init__(self, variable: str, continuous_parents: List[str], discrete_parents: List[str]):
        """Initialize a Conditional Linear Gaussian distribution."""
        super().__init__(variable, continuous_parents + discrete_parents)
        self.continuous_parents = continuous_parents
        self.discrete_parents = discrete_parents
        self.parameters = None
        self.logger = logging.getLogger(__name__)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set the parameters for the CLG distribution.
        
        Args:
            params: Dictionary containing:
                - parameters: Dict containing:
                    - mean_base: float
                    - coefficients: List[float]
                    - variance: float
                    - continuous_parents: List[str]
        """
        if "parameters" not in params:
            raise ValueError("Parameters dictionary must contain 'parameters' key")
            
        parameters = params["parameters"]
        required_params = {
            'mean_base': float,
            'coefficients': list,
            'variance': float
        }
        
        # Validate all required parameters are present and of correct type
        for param, param_type in required_params.items():
            if param not in parameters:
                raise ValueError(f"Missing required parameter '{param}'")
            if not isinstance(parameters[param], param_type):
                raise ValueError(f"Parameter '{param}' must be of type {param_type.__name__}")
                
        # Validate coefficients match continuous parents
        if len(parameters['coefficients']) != len(self.continuous_parents):
            raise ValueError(
                f"Number of coefficients ({len(parameters['coefficients'])}) must match "
                f"number of continuous parents ({len(self.continuous_parents)})"
            )
            
        # Validate variance is positive
        if parameters['variance'] <= 0:
            raise ValueError("Variance must be positive")
            
        self.parameters = parameters
        self.logger.debug(f"Set CLG parameters for {self.variable}: {parameters}")

    def get_probability(self, value: float, parent_values: Dict[str, Any]) -> float:
        """Get the probability density for a specific value given parent values."""
        if self.parameters is None:
            raise ValueError("Distribution parameters not set")
            
        # Calculate mean based on continuous parents
        mean = self.parameters['mean_base']
        for i, parent in enumerate(self.continuous_parents):
            if parent not in parent_values:
                raise ValueError(f"Missing value for continuous parent {parent}")
            mean += self.parameters['coefficients'][i] * parent_values[parent]
            
        variance = self.parameters['variance']
        
        # Calculate Gaussian probability density
        return (1 / (math.sqrt(2 * math.pi * variance))) * math.exp(-((value - mean) ** 2) / (2 * variance))

    def sample(self, parent_values: Dict[str, Any]) -> float:
        """Generate a random sample from the distribution given parent values."""
        if self.parameters is None:
            raise ValueError("Distribution parameters not set")
            
        # Calculate mean based on continuous parents
        mean = self.parameters['mean_base']
        for i, parent in enumerate(self.continuous_parents):
            if parent not in parent_values:
                raise ValueError(f"Missing value for continuous parent {parent}")
            mean += self.parameters['coefficients'][i] * parent_values[parent]
            
        std_dev = math.sqrt(self.parameters['variance'])
        
        return random.gauss(mean, std_dev)

    def get_parameters(self) -> Dict[str, Any]:
        """Get the distribution parameters."""
        if self.parameters is None:
            raise ValueError("Distribution parameters not set")
        return {
            'variable': self.variable,
            'continuous_parents': self.continuous_parents,
            'discrete_parents': self.discrete_parents,
            'parameters': self.parameters
        }