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
        self.distribution_type = 'clg'

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.logger.debug(f"Setting CLG parameters for {self.variable}: {parameters}")

        """
        Set the parameters for the CLG distribution.
        
        Args:
            parameters: Dictionary containing:
                parameters: Dict containing:
                    - mean_base: float
                    - coefficients: List[float]
                    - variance: float
                    - continuous_parents: List[str]
        """
        if "parameters" not in parameters:
            raise ValueError("Parameters dictionary must contain 'parameters' key")
                
        params = parameters["parameters"]
            
        # Validate required parameters
        required_params = {
            'mean_base': float,
            'coefficients': list,
            'variance': float
        }
        
        # Validate all required parameters are present and of correct type
        for param, param_type in required_params.items():
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}'")
            if not isinstance(params[param], param_type):
                raise ValueError(f"Parameter '{param}' must be of type {param_type.__name__}")
                    
        # Validate coefficients match continuous parents
        if len(params['coefficients']) != len(self.continuous_parents):
            raise ValueError(
                f"Number of coefficients ({len(params['coefficients'])}) must match "
                f"number of continuous parents ({len(self.continuous_parents)})"
            )
                
        # Validate variance is positive
        if params['variance'] <= 0:
            raise ValueError("Variance must be positive")
                
        self.parameters = params
        self.logger.debug(f"Set CLG parameters for {self.variable}: {params}")

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
            
        self.logger.debug(f"Getting CLG parameters for {self.variable}")
        self.logger.debug(f"Raw parameters: {self.parameters}")
        
        # Format parameters for validator expectations
        formatted_params = {
            'variable': self.variable,
            'continuous_parents': self.continuous_parents,
            'discrete_parents': self.discrete_parents,
            'parameters': {
                (): self.parameters  # Store parameters under empty tuple key
            }
        }
        
        self.logger.debug(f"Formatted parameters: {formatted_params}")
        return formatted_params