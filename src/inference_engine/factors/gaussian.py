# src/inference_engine/factors/gaussian.py

from typing import Dict, List, Any, Optional, Union
import numpy as np
from scipy.stats import norm
import logging

from .base import Factor, FactorValidationResult

class GaussianFactor(Factor):
    """
    Implementation of univariate Gaussian factor.
    Maintains exact parameter specifications and error tracking.
    
    Mathematical Properties:
    - Domain: x ∈ ℝ
    - PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    - Parameters: mean (μ), variance (σ²)
    - Error tracking in precision form
    """
    
    def __init__(self, 
                 variables: List[str],
                 mean: float,
                 variance: float,
                 name: Optional[str] = None):
        """
        Initialize Gaussian factor.
        
        Args:
            variables: List of variables (must be single variable)
            mean: Mean (μ) of distribution
            variance: Variance (σ²) of distribution
            name: Optional name for factor
            
        Raises:
            ValueError: If multiple variables or invalid parameters
        """
        if len(variables) != 1:
            raise ValueError("GaussianFactor must have exactly one variable")
            
        super().__init__(variables, name)
        self.mean = mean
        self.variance = variance
        self.logger.debug(
            f"Created Gaussian factor {self.name} with μ={mean}, σ²={variance}"
        )

    def validate(self) -> FactorValidationResult:
        """
        Validate Gaussian factor properties.
        
        Returns:
            FactorValidationResult with validation details
            
        Mathematical Requirements:
        - Variance must be strictly positive
        - Parameters must be finite
        - No default values allowed
        """
        # First perform base validation
        base_result = super().validate()
        if not base_result.is_valid:
            return base_result
            
        try:
            # Verify parameters are finite
            if not (np.isfinite(self.mean) and np.isfinite(self.variance)):
                return FactorValidationResult(
                    is_valid=False,
                    message="Parameters must be finite",
                    error_bound=float('inf')
                )
                
            # Verify positive variance
            if self.variance <= 0:
                return FactorValidationResult(
                    is_valid=False,
                    message=f"Variance must be positive, got {self.variance}",
                    error_bound=float('inf')
                )
                
            # Check for numerical stability issues
            if self.variance < 1e-13:
                self._numerical_issues.append(
                    f"Very small variance: {self.variance}"
                )
                
            if abs(self.mean) > 1e8:
                self._numerical_issues.append(
                    f"Large mean value: {self.mean}"
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
        Multiply with another Gaussian factor.
        
        Args:
            other: Factor to multiply with
            
        Returns:
            New factor representing the product
            
        Raises:
            ValueError: If factors are incompatible
            
        Mathematical Properties:
        - Exact parameter computation in precision form
        - Proper error bound tracking
        - Numerical stability monitoring
        """
        # First perform basic multiplication validation
        super().multiply(other)
        
        if not isinstance(other, GaussianFactor):
            raise ValueError("Can only multiply with another GaussianFactor")
            
        # Convert to precision form for numerical stability
        precision1 = 1.0 / self.variance
        precision2 = 1.0 / other.variance
        
        # Check condition number
        condition_number = max(precision1, precision2) / min(precision1, precision2)
        if condition_number > 1e13:
            self._numerical_issues.append(
                f"Poor conditioning in precision combination: {condition_number}"
            )
            
        # Combine parameters in precision form
        new_precision = precision1 + precision2
        new_variance = 1.0 / new_precision
        new_mean = (precision1 * self.mean + precision2 * other.mean) / new_precision
        
        result = GaussianFactor(
            self.variables,
            new_mean,
            new_variance,
            f"{self.name}_{other.name}"
        )
        
        # Track error propagation
        result._error_bound = (
            self._error_bound + 
            other._error_bound + 
            condition_number * np.finfo(float).eps
        )
        
        # Update condition number
        result._condition_number = condition_number
        
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
            ValueError: If attempting to marginalize the only variable
            
        Mathematical Properties:
        - Integration over variables
        - Error bound preservation
        """
        # First perform basic marginalization validation
        super().marginalize(variables)
        
        # For univariate Gaussian, marginalization over the variable
        # results in a constant factor with value 1
        if self.variables[0] in variables:
            from .base import ConstantFactor
            return ConstantFactor(
                [],
                1.0,
                f"{self.name}_marginalized"
            )
            
        # If not marginalizing the variable, return unchanged
        return self

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
        - Exact parameter computation
        - Proper error tracking
        """
        # First perform basic reduction validation
        super().reduce(evidence)
        
        # For Gaussian, reduction with evidence creates a constant factor
        if self.variables[0] in evidence:
            value = float(evidence[self.variables[0]])
            prob = self._compute_probability(value)
            
            from .base import ConstantFactor
            return ConstantFactor(
                [],
                prob,
                f"{self.name}_reduced"
            )
            
        # If no relevant evidence, return unchanged
        return self

    def normalize(self) -> 'Factor':
        """
        Normalize factor (Gaussian is already normalized).
        
        Returns:
            Self (Gaussian factor is inherently normalized)
            
        Mathematical Properties:
        - Maintains normalization by design
        """
        return self

    def _compute_probability(self, value: float) -> float:
        """
        Compute probability density at a value.
        
        Mathematical Properties:
        - Exact density computation
        - Numerical stability checks
        """
        # Use scipy.stats.norm for numerical stability
        return norm.pdf(value, loc=self.mean, scale=np.sqrt(self.variance))

    def _matching_variable_type(self, variable: str, other: 'Factor') -> bool:
        """Check if variable types are compatible."""
        return isinstance(other, GaussianFactor)

    def get_parameters(self) -> Dict[str, float]:
        """Get distribution parameters."""
        return {
            'mean': self.mean,
            'variance': self.variance,
            'precision': 1.0 / self.variance
        }

    def to_precision_form(self) -> Dict[str, float]:
        """
        Convert to precision form for numerical stability.
        
        Returns:
            Dictionary with precision parameters
            
        Mathematical Properties:
        - Exact conversion
        - Numerical stability monitoring
        """
        precision = 1.0 / self.variance
        precision_adjusted_mean = precision * self.mean
        
        if precision < 1e-13:
            self._numerical_issues.append(
                f"Very small precision: {precision}"
            )
            
        return {
            'precision': precision,
            'precision_adjusted_mean': precision_adjusted_mean,
            'log_normalization': -0.5 * (np.log(2 * np.pi) - np.log(precision))
        }