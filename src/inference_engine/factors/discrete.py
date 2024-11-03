# src/inference_engine/factors/discrete.py

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from copy import deepcopy
import logging

from .base import Factor, FactorValidationResult

class DiscreteFactor(Factor):
    """
    Implementation of discrete probability factor.
    Maintains exact probability values without approximation.
    
    Mathematical Properties:
    - Finite state space: X ∈ {x₁, x₂, ..., xₙ}
    - Complete probability specification required
    - ∑P(X = x) = 1 (within 1e-10)
    - P(X = x) ≥ 0 for all x
    """
    
    def __init__(self, 
                 variables: List[str],
                 states: Dict[str, List[str]],
                 values: Dict[Tuple[str, ...], float],
                 name: Optional[str] = None):
        """
        Initialize discrete factor.
        
        Args:
            variables: List of variables in factor's scope
            states: Dictionary mapping variables to their possible states
            values: Dictionary mapping state combinations to probabilities
            name: Optional name for factor
            
        Raises:
            ValueError: If parameters are invalid or incomplete
        """
        super().__init__(variables, name)
        self.states = states
        self._values = values
        self.logger.debug(
            f"Created discrete factor {self.name} with {len(variables)} variables"
        )

    def validate(self) -> FactorValidationResult:
        """
        Validate factor properties and parameters.
        
        Returns:
            FactorValidationResult with validation details
            
        Mathematical Requirements:
        - All states must be explicitly specified
        - All probabilities must be specified
        - Probabilities must sum to 1 (within 1e-10)
        - All probabilities must be non-negative
        """
        # First perform base validation
        base_result = super().validate()
        if not base_result.is_valid:
            return base_result
            
        try:
            # Verify states exist for all variables
            missing_states = set(self.variables) - set(self.states.keys())
            if missing_states:
                return FactorValidationResult(
                    is_valid=False,
                    message=f"Missing states for variables: {missing_states}",
                    error_bound=float('inf')
                )
                
            # Verify complete probability specification
            expected_combinations = 1
            for var in self.variables:
                expected_combinations *= len(self.states[var])
                
            if len(self._values) != expected_combinations:
                return FactorValidationResult(
                    is_valid=False,
                    message=f"Incomplete probability specification. Expected {expected_combinations} "
                           f"combinations, got {len(self._values)}",
                    error_bound=float('inf')
                )
                
            # Verify probability properties
            prob_sum = sum(self._values.values())
            if abs(prob_sum - 1.0) > 1e-10:
                return FactorValidationResult(
                    is_valid=False,
                    message=f"Probabilities sum to {prob_sum}, not 1.0",
                    error_bound=abs(prob_sum - 1.0)
                )
                
            if any(p < 0 or p > 1 for p in self._values.values()):
                return FactorValidationResult(
                    is_valid=False,
                    message="All probabilities must be between 0 and 1",
                    error_bound=float('inf')
                )
                
            # Check for numerical issues
            small_probs = [p for p in self._values.values() if 0 < p < 1e-14]
            if small_probs:
                self._numerical_issues.append(
                    f"Very small probabilities detected: {len(small_probs)} values < 1e-14"
                )
                
            self._validated = True
            return FactorValidationResult(
                is_valid=True,
                message="Validation successful",
                error_bound=1e-10,
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
        Multiply with another factor.
        
        Args:
            other: Factor to multiply with
            
        Returns:
            New factor representing the product
            
        Raises:
            ValueError: If factors are incompatible or invalid
            
        Mathematical Properties:
        - Exact probability multiplication
        - Proper error bound tracking
        - Automatic renormalization if needed
        """
        # First perform basic multiplication validation
        super().multiply(other)
        
        if not isinstance(other, DiscreteFactor):
            raise ValueError("Can only multiply with another DiscreteFactor")
            
        # Create combined variable list and states
        new_variables = list(set(self.variables + other.variables))
        new_states = {**self.states, **other.states}
        
        # Initialize result values
        new_values = {}
        
        # Compute probabilities in log space for numerical stability
        max_log_prob = float('-inf')
        log_probs = {}
        
        for assignment in self._get_all_assignments(new_variables, new_states):
            # Get relevant assignments for each factor
            self_assignment = self._project_assignment(assignment, self.variables)
            other_assignment = self._project_assignment(assignment, other.variables)
            
            # Multiply probabilities in log space
            if self_assignment in self._values and other_assignment in other._values:
                prob1 = self._values[self_assignment]
                prob2 = other._values[other_assignment]
                
                if prob1 > 0 and prob2 > 0:
                    log_prob = np.log(prob1) + np.log(prob2)
                    log_probs[assignment] = log_prob
                    max_log_prob = max(max_log_prob, log_prob)
        
        # Convert back from log space and normalize
        total_prob = 0.0
        for assignment, log_prob in log_probs.items():
            # Subtract max_log_prob for numerical stability
            new_values[assignment] = np.exp(log_prob - max_log_prob)
            total_prob += new_values[assignment]
            
        # Normalize
        for assignment in new_values:
            new_values[assignment] /= total_prob
            
        result = DiscreteFactor(
            new_variables,
            new_states,
            new_values,
            f"{self.name}_{other.name}"
        )
        
        # Track error propagation
        result._error_bound = (
            self._error_bound + 
            other._error_bound + 
            abs(1.0 - total_prob) + 
            np.finfo(float).eps
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
            New factor with specified variables marginalized out
            
        Raises:
            ValueError: If marginalization is invalid
            
        Mathematical Properties:
        - Exact probability summation
        - Error bound tracking
        - Proper normalization
        """
        # First perform basic marginalization validation
        super().marginalize(variables)
        
        # Identify variables to keep
        keep_vars = [v for v in self.variables if v not in variables]
        if not keep_vars:
            raise ValueError("Cannot marginalize all variables")
            
        # Create new states dictionary
        new_states = {v: self.states[v] for v in keep_vars}
        
        # Initialize result values
        new_values = {}
        
        # Group and sum probabilities
        for assignment, prob in self._values.items():
            keep_assignment = self._project_assignment(
                dict(zip(self.variables, assignment)),
                keep_vars
            )
            
            if keep_assignment not in new_values:
                new_values[keep_assignment] = 0.0
            new_values[keep_assignment] += prob
            
        result = DiscreteFactor(
            keep_vars,
            new_states,
            new_values,
            f"{self.name}_marginalized"
        )
        
        # Track error from summation
        result._error_bound = (
            self._error_bound + 
            len(self._values) * np.finfo(float).eps
        )
        
        # Validate result
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid marginalization result: {validation.message}")
            
        return result

    def reduce(self, evidence: Dict[str, Any]) -> 'Factor':
        """
        Reduce factor using evidence.
        
        Args:
            evidence: Dictionary mapping variables to their observed values
            
        Returns:
            New factor reduced by evidence
            
        Raises:
            ValueError: If evidence is invalid
            
        Mathematical Properties:
        - Exact reduction
        - Proper renormalization
        - Error tracking
        """
        # First perform basic reduction validation
        super().reduce(evidence)
        
        # Create new variable list excluding evidence variables
        new_variables = [v for v in self.variables if v not in evidence]
        
        # Create new states dictionary
        new_states = {v: self.states[v] for v in new_variables}
        
        # Initialize result values
        new_values = {}
        
        # Process each assignment
        for assignment, prob in self._values.items():
            # Check if assignment is consistent with evidence
            assignment_dict = dict(zip(self.variables, assignment))
            if all(assignment_dict[var] == val for var, val in evidence.items()):
                # Keep only non-evidence variables
                new_assignment = self._project_assignment(
                    assignment_dict,
                    new_variables
                )
                new_values[new_assignment] = prob
                
        # Normalize the result
        total_prob = sum(new_values.values())
        if total_prob > 0:
            new_values = {k: v/total_prob for k, v in new_values.items()}
            
        result = DiscreteFactor(
            new_variables,
            new_states,
            new_values,
            f"{self.name}_reduced"
        )
        
        # Track error from reduction and normalization
        result._error_bound = (
            self._error_bound + 
            abs(1.0 - total_prob) + 
            np.finfo(float).eps
        )
        
        # Validate result
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid reduction result: {validation.message}")
            
        return result

    def normalize(self) -> 'Factor':
        """
        Normalize factor probabilities.
        
        Returns:
            New normalized factor
            
        Mathematical Properties:
        - Exact normalization
        - Error bound tracking
        """
        total = sum(self._values.values())
        if total <= 0:
            raise ValueError("Cannot normalize factor with zero probabilities")
            
        new_values = {k: v/total for k, v in self._values.items()}
        
        result = DiscreteFactor(
            self.variables,
            self.states,
            new_values,
            f"{self.name}_normalized"
        )
        
        # Track normalization error
        result._error_bound = self._error_bound + abs(1.0 - total)
        
        # Validate result
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid normalization result: {validation.message}")
            
        return result

    def _matching_variable_type(self, variable: str, other: 'Factor') -> bool:
        """Check if variable has matching states in both factors."""
        if not isinstance(other, DiscreteFactor):
            return False
        if variable not in self.states or variable not in other.states:
            return False
        return set(self.states[variable]) == set(other.states[variable])

    def _get_all_assignments(self, 
                           variables: List[str],
                           states: Dict[str, List[str]]) -> List[Tuple[str, ...]]:
        """Get all possible state assignments for variables."""
        if not variables:
            return [()]
            
        var = variables[0]
        sub_assignments = self._get_all_assignments(variables[1:], states)
        
        assignments = []
        for state in states[var]:
            for sub in sub_assignments:
                assignments.append((state,) + sub)
                
        return assignments

    def _project_assignment(self,
                          assignment: Dict[str, str],
                          variables: List[str]) -> Tuple[str, ...]:
        """Project assignment onto subset of variables."""
        return tuple(assignment[var] for var in variables)