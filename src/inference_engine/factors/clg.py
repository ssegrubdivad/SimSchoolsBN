# src/inference_engine/factors/clg.py

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from copy import deepcopy
import logging

from .base import Factor, FactorValidationResult
from .gaussian import GaussianFactor
from .discrete import DiscreteFactor

class CLGFactor(Factor):
    """
    Implementation of Conditional Linear Gaussian factor.
    Maintains exact parameter specifications and relationships.
    
    Mathematical Properties:
    - f(x|y,z) = N(α(z) + β(z)ᵀy, σ²(z))
    - x: continuous variable
    - y: continuous parents
    - z: discrete parents
    - Exact parameter specifications required
    - Linear relationship with continuous parents
    """
    
    def __init__(self, 
                 variables: List[str],
                 continuous_var: str,
                 continuous_parents: List[str],
                 discrete_parents: List[str],
                 discrete_states: Dict[str, List[str]],
                 parameters: Dict[Tuple[str, ...], Dict[str, Any]],
                 name: Optional[str] = None):
        """
        Initialize CLG factor.
        
        Args:
            variables: All variables in factor's scope
            continuous_var: The continuous variable
            continuous_parents: List of continuous parent variables
            discrete_parents: List of discrete parent variables
            discrete_states: Dictionary mapping discrete variables to their states
            parameters: Dictionary mapping discrete configurations to CLG parameters:
                {(discrete_config): {
                    'mean_base': float,
                    'coefficients': List[float],
                    'variance': float
                }}
            name: Optional name for factor
            
        Raises:
            ValueError: If parameters are invalid or incomplete
        """
        super().__init__(variables, name)
        
        if continuous_var not in variables:
            raise ValueError(f"Continuous variable {continuous_var} not in variables")
            
        self.continuous_var = continuous_var
        self.continuous_parents = continuous_parents
        self.discrete_parents = discrete_parents
        self.discrete_states = discrete_states
        self.parameters = parameters
        
        self.logger.debug(
            f"Created CLG factor {self.name} with {len(continuous_parents)} "
            f"continuous parents and {len(discrete_parents)} discrete parents"
        )

    def validate(self) -> FactorValidationResult:
        """
        Validate CLG factor properties.
        
        Returns:
            FactorValidationResult with validation details
            
        Mathematical Requirements:
        - Complete parameter sets for all discrete configurations
        - Valid coefficients for continuous parents
        - Positive variance for all configurations
        - Linear relationship preservation
        """
        # First perform base validation
        base_result = super().validate()
        if not base_result.is_valid:
            return base_result
            
        try:
            # Verify discrete states exist
            for parent in self.discrete_parents:
                if parent not in self.discrete_states:
                    return FactorValidationResult(
                        is_valid=False,
                        message=f"Missing states for discrete parent: {parent}",
                        error_bound=float('inf')
                    )
                    
            # Get all possible discrete configurations
            configs = self._get_all_discrete_configs()
            
            # Verify parameters exist for all configurations
            for config in configs:
                if config not in self.parameters:
                    return FactorValidationResult(
                        is_valid=False,
                        message=f"Missing parameters for configuration: {config}",
                        error_bound=float('inf')
                    )
                    
                params = self.parameters[config]
                
                # Verify required parameters exist
                required_params = {'mean_base', 'coefficients', 'variance'}
                if not all(param in params for param in required_params):
                    missing = required_params - set(params.keys())
                    return FactorValidationResult(
                        is_valid=False,
                        message=f"Missing parameters {missing} for config {config}",
                        error_bound=float('inf')
                    )
                    
                # Verify coefficient vector length
                if len(params['coefficients']) != len(self.continuous_parents):
                    return FactorValidationResult(
                        is_valid=False,
                        message=f"Number of coefficients ({len(params['coefficients'])}) "
                                f"does not match number of continuous parents "
                                f"({len(self.continuous_parents)}) for config {config}",
                        error_bound=float('inf')
                    )
                    
                # Verify positive variance
                if params['variance'] <= 0:
                    return FactorValidationResult(
                        is_valid=False,
                        message=f"Variance must be positive for config {config}",
                        error_bound=float('inf')
                    )
                    
                # Check for numerical issues
                if params['variance'] < 1e-13:
                    self._numerical_issues.append(
                        f"Very small variance for config {config}: {params['variance']}"
                    )
                    
                if abs(params['mean_base']) > 1e8:
                    self._numerical_issues.append(
                        f"Large mean base for config {config}: {params['mean_base']}"
                    )
                    
                if any(abs(c) > 1e8 for c in params['coefficients']):
                    self._numerical_issues.append(
                        f"Large coefficients for config {config}"
                    )
                    
            self._validated = True
            return FactorValidationResult(
                is_valid=True,
                message="Validation successful",
                error_bound=np.finfo(float).eps,
                numerical_issues=self._numerical_issues
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return FactorValidationResult(
                is_valid=False,
                message=f"Validation failed: {str(e)}",
                error_bound=float('inf')
            )

    def multiply(self, other: 'Factor') -> 'Factor':
        """
        Multiply with another CLG factor.
        
        Args:
            other: Factor to multiply with
            
        Returns:
            New factor representing the product
            
        Raises:
            ValueError: If factors are incompatible
            
        Mathematical Properties:
        - Exact parameter computation
        - Proper relationship preservation
        - Error tracking
        """
        # First perform basic multiplication validation
        super().multiply(other)
        
        if not isinstance(other, CLGFactor):
            raise ValueError("Can only multiply with another CLGFactor")
            
        # Combine variables and parents
        new_continuous_var = self.continuous_var
        new_continuous_parents = list(set(self.continuous_parents + 
                                        other.continuous_parents))
        new_discrete_parents = list(set(self.discrete_parents + 
                                      other.discrete_parents))
        
        # Combine discrete states
        new_discrete_states = {**self.discrete_states, **other.discrete_states}
        
        # Get all discrete configurations
        configs = self._get_all_discrete_configs(new_discrete_parents, new_discrete_states)
        
        # Initialize new parameters
        new_parameters = {}
        
        # Process each configuration
        for config in configs:
            # Get relevant parameters from each factor
            self_params = self._get_params_for_config(config)
            other_params = other._get_params_for_config(config)
            
            if self_params is None or other_params is None:
                continue
                
            # Combine parameters in precision form
            precision1 = 1.0 / self_params['variance']
            precision2 = 1.0 / other_params['variance']
            new_precision = precision1 + precision2
            
            # Check condition number
            condition_number = max(precision1, precision2) / min(precision1, precision2)
            if condition_number > 1e13:
                self._numerical_issues.append(
                    f"Poor conditioning for config {config}: {condition_number}"
                )
            
            # Combine base means
            new_mean_base = (
                precision1 * self_params['mean_base'] + 
                precision2 * other_params['mean_base']
            ) / new_precision
            
            # Combine coefficients
            new_coefficients = []
            for parent in new_continuous_parents:
                coef1 = self._get_coefficient(parent, self_params)
                coef2 = other._get_coefficient(parent, other_params)
                new_coef = (
                    precision1 * coef1 + precision2 * coef2
                ) / new_precision
                new_coefficients.append(new_coef)
                
            new_parameters[config] = {
                'mean_base': new_mean_base,
                'coefficients': new_coefficients,
                'variance': 1.0 / new_precision
            }
            
        result = CLGFactor(
            list(set(self.variables + other.variables)),
            new_continuous_var,
            new_continuous_parents,
            new_discrete_parents,
            new_discrete_states,
            new_parameters,
            f"{self.name}_{other.name}"
        )
        
        # Track error propagation
        result._error_bound = (
            self._error_bound + 
            other._error_bound + 
            len(configs) * np.finfo(float).eps
        )
        
        # Validate result
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid multiplication result: {validation.message}")
            
        return result

    def marginalize(self, variables: List[str]) -> 'Factor':
        """
        Marginalize out specified variables.
        
        Args:
            variables: Variables to marginalize out
            
        Returns:
            New marginalized factor
            
        Raises:
            ValueError: If marginalization is invalid
            
        Mathematical Properties:
        - Exact marginalization
        - Relationship preservation
        - Error tracking
        """
        # First perform basic marginalization validation
        super().marginalize(variables)
        
        # Identify variables to keep
        keep_cont_parents = [v for v in self.continuous_parents 
                           if v not in variables]
        keep_disc_parents = [v for v in self.discrete_parents 
                           if v not in variables]
        
        # If marginalizing continuous variable, result is discrete
        if self.continuous_var in variables:
            # Create discrete factor over remaining discrete parents
            values = self._marginalize_continuous()
            return DiscreteFactor(
                keep_disc_parents,
                {p: self.discrete_states[p] for p in keep_disc_parents},
                values,
                f"{self.name}_marginalized"
            )
            
        # Initialize new parameters with adjusted dimensionality
        new_parameters = {}
        
        # Process each configuration
        for config, params in self.parameters.items():
            # Project configuration onto kept variables
            new_config = self._project_config(config, keep_disc_parents)
            
            if new_config not in new_parameters:
                new_parameters[new_config] = {
                    'mean_base': params['mean_base'],
                    'coefficients': [
                        c for i, c in enumerate(params['coefficients'])
                        if self.continuous_parents[i] in keep_cont_parents
                    ],
                    'variance': params['variance']
                }
            else:
                # Combine parameters for same configuration
                old_params = new_parameters[new_config]
                precision1 = 1.0 / old_params['variance']
                precision2 = 1.0 / params['variance']
                new_precision = precision1 + precision2
                
                new_parameters[new_config] = {
                    'mean_base': (
                        precision1 * old_params['mean_base'] + 
                        precision2 * params['mean_base']
                    ) / new_precision,
                    'coefficients': [
                        (precision1 * c1 + precision2 * c2) / new_precision
                        for c1, c2 in zip(
                            old_params['coefficients'],
                            [c for i, c in enumerate(params['coefficients'])
                             if self.continuous_parents[i] in keep_cont_parents]
                        )
                    ],
                    'variance': 1.0 / new_precision
                }
                
        result = CLGFactor(
            [self.continuous_var] + keep_cont_parents + keep_disc_parents,
            self.continuous_var,
            keep_cont_parents,
            keep_disc_parents,
            {p: self.discrete_states[p] for p in keep_disc_parents},
            new_parameters,
            f"{self.name}_marginalized"
        )
        
        # Track marginalization error
        result._error_bound = (
            self._error_bound + 
            len(self.parameters) * np.finfo(float).eps
        )
        
        # Validate result
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid marginalization result: {validation.message}")
            
        return result

    def _get_all_discrete_configs(self, 
                                parents: Optional[List[str]] = None,
                                states: Optional[Dict[str, List[str]]] = None) -> List[Tuple[str, ...]]:
        """Get all possible discrete parent configurations."""
        if parents is None:
            parents = self.discrete_parents
        if states is None:
            states = self.discrete_states
            
        def _recursive_configs(remaining: List[str]) -> List[Tuple[str, ...]]:
            if not remaining:
                return [()]
                
            parent = remaining[0]
            sub_configs = _recursive_configs(remaining[1:])
            
            configs = []
            for state in states[parent]:
                for sub in sub_configs:
                    configs.append((state,) + sub)
                    
            return configs
            
        return _recursive_configs(parents)

    def _get_params_for_config(self, 
                             config: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """Get parameters for a discrete configuration."""
        # Project configuration onto our discrete parents
        projected = self._project_config(
            config,
            self.discrete_parents
        )
        return self.parameters.get(projected)

    def _get_coefficient(self,
                        parent: str,
                        params: Dict[str, Any]) -> float:
        """Get coefficient for a continuous parent."""
        try:
            idx = self.continuous_parents.index(parent)
            return params['coefficients'][idx]
        except ValueError:
            return 0.0

    def _project_config(self,
                       config: Tuple[str, ...],
                       parents: List[str]) -> Tuple[str, ...]:
        """Project configuration onto subset of parents."""
        # Create mapping from all parents to their values
        full_config = dict(zip(self.discrete_parents, config))
        # Extract values for desired parents
        return tuple(full_config[p] for p in parents)