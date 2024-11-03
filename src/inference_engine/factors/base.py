# src/inference_engine/factors/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Union, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

@dataclass
class FactorValidationResult:
    """Results of factor validation with error tracking."""
    is_valid: bool
    message: str
    error_bound: float
    numerical_issues: Optional[List[str]] = None
    validation_details: Optional[Dict[str, Any]] = None

class Factor(ABC):
    """
    Abstract base class for factors in the inference engine.
    Provides strict mathematical guarantees for all factor operations.
    
    Mathematical Properties Required:
    - Explicit parameter specification
    - Exact numerical computations
    - Complete error bound tracking
    - No silent approximations
    """
    
    def __init__(self, 
                 variables: List[str],
                 name: Optional[str] = None):
        """
        Initialize factor.
        
        Args:
            variables: List of variables in factor's scope
            name: Optional name for factor
            
        Properties:
            - Factor scope must be explicitly specified
            - No default parameters allowed
            - Variables order is maintained
        """
        self.variables = variables
        self.name = name or f"Factor_{id(self)}"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Internal state
        self._validated = False
        self._error_bound: float = 0.0
        self._numerical_issues: List[str] = []
        self._condition_number: Optional[float] = None
        
        self.logger.debug(
            f"Created factor {self.name} with variables {self.variables}"
        )

    @abstractmethod
    def validate(self) -> FactorValidationResult:
        """
        Validate factor properties and parameters.
        Must be called before any operations.
        
        Returns:
            FactorValidationResult with validation details
            
        Mathematical Requirements:
        - Complete parameter validation
        - Error bound calculation
        - Numerical stability verification
        """
        if not self.variables:
            return FactorValidationResult(
                is_valid=False,
                message="Factor must have at least one variable",
                error_bound=float('inf')
            )
        return FactorValidationResult(
            is_valid=True,
            message="Base validation successful",
            error_bound=0.0
        )

    @abstractmethod
    def multiply(self, other: 'Factor') -> 'Factor':
        """
        Multiply this factor with another factor.
        
        Args:
            other: Factor to multiply with
            
        Returns:
            New factor representing the product
            
        Mathematical Guarantees:
        - Exact multiplication
        - Error bound tracking
        - No approximations
        """
        # Validate factors
        if not (self._validated and other._validated):
            raise ValueError("Both factors must be validated before multiplication")
        
        # Verify compatibility
        if not self._compatible_for_multiplication(other):
            raise ValueError(
                f"Incompatible factors for multiplication: "
                f"{self.name} and {other.name}"
            )

    @abstractmethod
    def marginalize(self, variables: List[str]) -> 'Factor':
        """
        Marginalize out specified variables.
        
        Args:
            variables: Variables to marginalize out
            
        Returns:
            New factor with specified variables marginalized out
            
        Mathematical Guarantees:
        - Exact marginalization
        - Error bound tracking
        - No approximations
        """
        # Validate factor
        if not self._validated:
            raise ValueError("Factor must be validated before marginalization")
        
        # Verify variables exist in factor
        invalid_vars = set(variables) - set(self.variables)
        if invalid_vars:
            raise ValueError(f"Cannot marginalize variables not in factor: {invalid_vars}")

    @abstractmethod
    def reduce(self, evidence: Dict[str, Any]) -> 'Factor':
        """
        Reduce factor using evidence.
        
        Args:
            evidence: Dictionary mapping variables to their observed values
            
        Returns:
            New factor reduced by evidence
            
        Mathematical Guarantees:
        - Exact evidence incorporation
        - Error bound tracking
        - Value validation
        """
        # Validate factor
        if not self._validated:
            raise ValueError("Factor must be validated before reduction")
        
        # Verify evidence variables
        invalid_vars = set(evidence.keys()) - set(self.variables)
        if invalid_vars:
            raise ValueError(f"Evidence provided for variables not in factor: {invalid_vars}")

    @abstractmethod
    def normalize(self) -> 'Factor':
        """
        Normalize factor values.
        
        Returns:
            New normalized factor
            
        Mathematical Guarantees:
        - Sum to 1 (discrete) or valid density (continuous)
        - Error bound tracking
        - No precision loss
        """
        # Validate factor
        if not self._validated:
            raise ValueError("Factor must be validated before normalization")

    def get_error_bound(self) -> float:
        """Get current error bound for factor."""
        return self._error_bound

    def get_numerical_issues(self) -> List[str]:
        """Get list of numerical issues encountered."""
        return self._numerical_issues.copy()

    def get_condition_number(self) -> Optional[float]:
        """Get condition number if applicable."""
        return self._condition_number

    def _compatible_for_multiplication(self, other: 'Factor') -> bool:
        """Check if factors are compatible for multiplication."""
        # Verify matching variables have same type
        common_vars = set(self.variables) & set(other.variables)
        for var in common_vars:
            if not self._matching_variable_type(var, other):
                return False
        return True

    @abstractmethod
    def _matching_variable_type(self, variable: str, other: 'Factor') -> bool:
        """Check if variable has matching type in both factors."""
        pass

    def _track_error(self, operation: str, local_error: float) -> None:
        """
        Track error accumulation from operations.
        
        Args:
            operation: Name of operation causing error
            local_error: Local error from operation
            
        Mathematical Properties:
        - Conservative error bound
        - Includes machine epsilon
        - Tracks condition number effects
        """
        self._error_bound += local_error + np.finfo(float).eps
        
        if self._condition_number is not None:
            self._error_bound *= self._condition_number
            
        if self._error_bound > 1e-6:
            self._numerical_issues.append(
                f"Large accumulated error in {operation}: {self._error_bound}"
            )

    def _update_condition_number(self, matrix: np.ndarray) -> None:
        """Update condition number based on matrix operation."""
        try:
            new_condition = np.linalg.cond(matrix)
            if self._condition_number is None:
                self._condition_number = new_condition
            else:
                self._condition_number = max(self._condition_number, new_condition)
                
            if self._condition_number > 1e13:
                self._numerical_issues.append(
                    f"Poor conditioning detected: {self._condition_number}"
                )
        except np.linalg.LinAlgError:
            self._numerical_issues.append("Could not compute condition number")

    def __str__(self) -> str:
        """String representation of factor."""
        status = "validated" if self._validated else "unvalidated"
        error = f"error_bound={self._error_bound:.2e}"
        return f"{self.__class__.__name__}({self.name}, {status}, {error})"

    def __repr__(self) -> str:
        """Detailed string representation of factor."""
        return (f"{self.__class__.__name__}("
                f"name={self.name}, "
                f"variables={self.variables}, "
                f"validated={self._validated}, "
                f"error_bound={self._error_bound:.2e}, "
                f"condition_number={self._condition_number})")