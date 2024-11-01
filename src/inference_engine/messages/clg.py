# src/inference_engine/messages/clg.py

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
import logging
from scipy.stats import truncnorm
from enum import Enum
import numpy as np

from .base import Message, MessageType, ValidationResult, ConstantMessage

class CLGMessage(Message):
    """
    Message containing Conditional Linear Gaussian information.
    Must explicitly specify all parameters for each discrete parent configuration.
    """
    def __init__(self, 
                 source_id: str, 
                 target_id: str,
                 continuous_var: str,
                 continuous_parents: List[str],
                 discrete_parents: List[str],
                 direction: str,
                 discrete_states: Dict[str, List[str]],
                 parameters: Dict[tuple, Dict[str, Union[float, List[float]]]]):
        """
        Initialize a CLG message.
        
        Args:
            source_id: Identifier of the sending node
            target_id: Identifier of the receiving node
            continuous_var: The continuous variable
            continuous_parents: List of continuous parent variables
            discrete_parents: List of discrete parent variables
            direction: Message direction
            discrete_states: Dictionary mapping discrete parents to their states
            parameters: Dictionary mapping discrete state combinations to CLG parameters:
                       {
                           (discrete_state_combo): {
                               'mean_base': float,
                               'coefficients': List[float],
                               'variance': float
                           }
                       }
        """
        variables = [continuous_var] + continuous_parents + discrete_parents
        super().__init__(source_id, target_id, variables, direction)
        self.continuous_var = continuous_var
        self.continuous_parents = continuous_parents
        self.discrete_parents = discrete_parents
        self.discrete_states = discrete_states
        self.parameters = parameters
        self.logger.debug(f"Created CLGMessage from {source_id} to {target_id}")

    def validate(self) -> ValidationResult:
        """Validate CLG message content."""
        basic_result = super().validate()
        if not basic_result.is_valid:
            return basic_result

        # Check discrete states completeness
        if not all(p in self.discrete_states for p in self.discrete_parents):
            missing = set(self.discrete_parents) - set(self.discrete_states.keys())
            return ValidationResult(
                False,
                f"Missing states for discrete parents: {missing}"
            )

        # Get all discrete parent configurations
        configs = self._get_discrete_configurations()

        # Check parameters for each configuration
        for config in configs:
            config_key = tuple(config.values())
            
            # Check if configuration exists in parameters
            if config_key not in self.parameters:
                return ValidationResult(
                    False,
                    f"Missing parameters for discrete configuration: {config}"
                )

            params = self.parameters[config_key]
            required_params = {'mean_base', 'coefficients', 'variance'}

            # Check required parameters exist
            if not all(p in params for p in required_params):
                missing = required_params - set(params.keys())
                return ValidationResult(
                    False,
                    f"Missing parameters {missing} for configuration {config}"
                )

            # Check coefficient vector length
            if len(params['coefficients']) != len(self.continuous_parents):
                return ValidationResult(
                    False,
                    f"Number of coefficients ({len(params['coefficients'])}) does not match "
                    f"number of continuous parents ({len(self.continuous_parents)}) "
                    f"for configuration {config}"
                )

            # Check variance
            if params['variance'] <= 0:
                return ValidationResult(
                    False,
                    f"Variance must be positive, got {params['variance']} "
                    f"for configuration {config}"
                )

        self._validated = True
        return ValidationResult(True, "Validation successful")

    def combine(self, other: 'Message') -> 'Message':
        """Combine with another CLG message."""
        super().combine(other)
        
        if not isinstance(other, CLGMessage):
            raise ValueError("Can only combine CLG messages with other CLG messages")

        # Verify compatible structure
        if (self.continuous_var != other.continuous_var or
            set(self.continuous_parents) != set(other.continuous_parents) or
            set(self.discrete_parents) != set(other.discrete_parents)):
            raise ValueError("Messages must have compatible structure for combination")

        # Combine parameters for each discrete configuration
        new_params = {}
        configs = self._get_discrete_configurations()

        for config in configs:
            config_key = tuple(config.values())
            params1 = self.parameters[config_key]
            params2 = other.parameters[config_key]

            # Precision-weighted combination
            precision1 = 1.0 / params1['variance']
            precision2 = 1.0 / params2['variance']
            new_precision = precision1 + precision2
            new_variance = 1.0 / new_precision

            # Combine mean bases and coefficients
            new_mean_base = (precision1 * params1['mean_base'] + 
                           precision2 * params2['mean_base']) / new_precision
            
            new_coefficients = [
                (precision1 * c1 + precision2 * c2) / new_precision
                for c1, c2 in zip(params1['coefficients'], params2['coefficients'])
            ]

            new_params[config_key] = {
                'mean_base': new_mean_base,
                'coefficients': new_coefficients,
                'variance': new_variance
            }

        result = CLGMessage(
            f"{self.source_id}_{other.source_id}",
            self.target_id,
            self.continuous_var,
            self.continuous_parents,
            self.discrete_parents,
            self.direction,
            self.discrete_states,
            new_params
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Message combination resulted in invalid message: {validation.message}")

        return result

    def marginalize(self, variables: List[str]) -> 'Message':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # Separate variables by type
        cont_vars = [v for v in variables if v in [self.continuous_var] + self.continuous_parents]
        disc_vars = [v for v in variables if v in self.discrete_parents]

        if not cont_vars and not disc_vars:
            return self

        if self.continuous_var in variables:
            # Marginalizing out the main continuous variable results in a constant message
            return ConstantMessage(
                self.source_id,
                self.target_id,
                [v for v in self.variables if v not in variables],
                self.direction,
                1.0
            )

        # Create new message with updated structure
        new_cont_parents = [v for v in self.continuous_parents if v not in variables]
        new_disc_parents = [v for v in self.discrete_parents if v not in variables]
        new_disc_states = {p: self.discrete_states[p] for p in new_disc_parents}

        # For each remaining discrete configuration, marginalize continuous parents
        new_params = {}
        remaining_configs = self._get_discrete_configurations(new_disc_parents)

        for new_config in remaining_configs:
            new_config_key = tuple(new_config.values())
            
            # Combine parameters from all compatible original configurations
            orig_configs = self._get_compatible_configs(new_config)
            
            for orig_config in orig_configs:
                orig_params = self.parameters[tuple(orig_config.values())]
                
                # Update parameters, removing marginalized continuous parents
                marginalized_indices = [i for i, p in enumerate(self.continuous_parents) 
                                     if p in variables]
                new_coefficients = [c for i, c in enumerate(orig_params['coefficients'])
                                  if i not in marginalized_indices]
                
                if new_config_key in new_params:
                    # Combine with existing parameters
                    existing = new_params[new_config_key]
                    precision1 = 1.0 / existing['variance']
                    precision2 = 1.0 / orig_params['variance']
                    new_precision = precision1 + precision2
                    
                    new_params[new_config_key] = {
                        'mean_base': (precision1 * existing['mean_base'] + 
                                    precision2 * orig_params['mean_base']) / new_precision,
                        'coefficients': [
                            (precision1 * c1 + precision2 * c2) / new_precision
                            for c1, c2 in zip(existing['coefficients'], new_coefficients)
                        ],
                        'variance': 1.0 / new_precision
                    }
                else:
                    new_params[new_config_key] = {
                        'mean_base': orig_params['mean_base'],
                        'coefficients': new_coefficients,
                        'variance': orig_params['variance']
                    }

        result = CLGMessage(
            self.source_id,
            self.target_id,
            self.continuous_var,
            new_cont_parents,
            new_disc_parents,
            self.direction,
            new_disc_states,
            new_params
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Marginalization resulted in invalid message: {validation.message}")

        return result

    def _get_discrete_configurations(self, parents: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Get all possible discrete parent configurations."""
        if parents is None:
            parents = self.discrete_parents

        def _recursive_configs(remaining_parents: List[str]) -> List[Dict[str, str]]:
            if not remaining_parents:
                return [{}]
            
            parent = remaining_parents[0]
            sub_configs = _recursive_configs(remaining_parents[1:])
            configs = []
            
            for state in self.discrete_states[parent]:
                for sub in sub_configs:
                    configs.append({parent: state, **sub})
                    
            return configs

        return _recursive_configs(parents)

    def _get_compatible_configs(self, partial_config: Dict[str, str]) -> List[Dict[str, str]]:
        """Get all original configurations compatible with a partial configuration."""
        return [
            config for config in self._get_discrete_configurations()
            if all(config[k] == v for k, v in partial_config.items())
        ]

    def get_type(self) -> MessageType:
        """Get message type."""
        return MessageType.CLG