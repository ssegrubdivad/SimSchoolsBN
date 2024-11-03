# src/inference_engine/factors/truncated_gaussian.py

from typing import Dict, List, Any, Optional, Union
import numpy as np
from scipy.stats import truncnorm
import logging

from .gaussian import GaussianFactor
from .base import Factor, FactorValidationResult

class TruncatedGaussianFactor(GaussianFactor):
    """
    Implementation of truncated Gaussian factor.
    Maintains exact parameter specifications with bounds.
    
    Mathematical Properties:
    - Domain: x ∈ [a,b]
    - PDF: f(x) = φ((x-μ)/σ) / (σ(Φ((b-μ)/σ) - Φ((a-μ)/σ)))
    - Parameters: mean (μ), variance (σ²), bounds (a,b)
    - Proper bound handling
    """
    
    def __init__(self, 
                 variables: List[str],
                 mean: float,
                 variance: float,
                 lower_bound: float,
                 upper_bound: float,
                 name: Optional[str] = None):
        """
        Initialize truncated Gaussian factor.
        
        Args:
            variables: List of variables (must be single variable)
            mean: Mean (μ) of underlying Gaussian
            variance: Variance (σ²) of underlying Gaussian
            lower_bound: Lower bound (a) of domain
            upper_bound: Upper bound (b) of domain
            name: Optional name for factor
            
        Raises:
            ValueError: If bounds are invalid
        """
        super().__init__(variables, mean, variance, name)
        if lower_bound >= upper_bound:
            raise ValueError(
                f"Lower bound ({lower_bound}) must be less than "
                f"upper bound ({upper_bound})"
            )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.logger.debug(
            f"Created truncated Gaussian factor {self.name} "
            f"with bounds [{lower_bound}, {upper_bound}]"
        )

    def validate(self) -> FactorValidationResult:
        """
        Validate truncated Gaussian factor properties.
        
        Returns:
            FactorValidationResult with validation details
            
        Mathematical Requirements:
        - All base Gaussian requirements
        - Valid bounds (a < b)
        - Finite bounds
        """
        # First perform base Gaussian validation
        base_result = super().validate()
        if not base_result.is_valid:
            return base_result
            
        try:
            # Verify finite bounds
            if not (np.isfinite(self.lower_bound) and np.isfinite(self.upper_bound)):
                return FactorValidationResult(
                    is_valid=False,
                    message="Bounds must be finite",
                    error_bound=float('inf')
                )
                
            # Verify bound ordering (already checked in init, but verify again)
            if self.lower_bound >= self.upper_bound:
                return FactorValidationResult(
                    is_valid=False,
                    message=f"Invalid bounds: [{self.lower_bound}, {self.upper_bound}]",
                    error_bound=float('inf')
                )
                
            # Check numerical stability of bounds
            std_dev = np.sqrt(self.variance)
            alpha = (self.lower_bound - self.mean) / std_dev
            beta = (self.upper_bound - self.mean) / std_dev
            
            if abs(alpha) > 1e8 or abs(beta) > 1e8:
                self._numerical_issues.append(
                    "Standardized bounds are very large"
                )
                
            if beta - alpha < 1e-10:
                self._numerical_issues.append(
                    "Bounds are very close relative to standard deviation"
                )
                
            return FactorValidationResult(
                is_valid=True,
                message="Validation successful",
                error_bound=base_result.error_bound,
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
        Multiply with another truncated Gaussian factor.
        
        Args:
            other: Factor to multiply with
            
        Returns:
            New factor representing the product
            
        Raises:
            ValueError: If factors are incompatible
            
        Mathematical Properties:
        - Exact parameter computation
        - Proper bound intersection
        - Error tracking
        """
        if not isinstance(other, TruncatedGaussianFactor):
            raise ValueError("Can only multiply with another TruncatedGaussianFactor")
            
        # Get combined parameters through base multiplication
        base_result = super().multiply(other)
        
        # Compute intersection of bounds
        new_lower = max(self.lower_bound, other.lower_bound)
        new_upper = min(self.upper_bound, other.upper_bound)
        
        if new_lower >= new_upper:
            raise ValueError("Multiplication results in empty domain")
            
        result = TruncatedGaussianFactor(
            base_result.variables,
            base_result.mean,
            base_result.variance,
            new_lower,
            new_upper,
            f"{self.name}_{other.name}"
        )
        
        # Track errors from base multiplication and bound intersection
        result._error_bound = (
            base_result._error_bound +
            abs(new_upper - new_lower) * np.finfo(float).eps
        )
        
        # Validate result
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid multiplication result: {validation.message}")
            
        return result

    def _compute_probability(self, value: float) -> float:
        """
        Compute truncated Gaussian probability density.
        
        Mathematical Properties:
        - Exact density computation
        - Proper bound handling
        """
        if value < self.lower_bound or value > self.upper_bound:
            return 0.0
            
        # Use scipy.stats.truncnorm for numerical stability
        a = (self.lower_bound - self.mean) / np.sqrt(self.variance)
        b = (self.upper_bound - self.mean) / np.sqrt(self.variance)
        
        return truncnorm.pdf(
            value,
            a,
            b,
            loc=self.mean,
            scale=np.sqrt(self.variance)
        )

    def get_parameters(self) -> Dict[str, float]:
        """Get distribution parameters including bounds."""
        params = super().get_parameters()
        params.update({
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound
        })
        return params

    def _matching_variable_type(self, variable: str, other: 'Factor') -> bool:
        """Check if variable types are compatible."""
        return isinstance(other, (GaussianFactor, TruncatedGaussianFactor))