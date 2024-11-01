# src/inference_engine/messages/gaussian.py

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
import logging
from scipy.stats import truncnorm
from enum import Enum
import numpy as np

from .base import Message, MessageType, ValidationResult, ConstantMessage

class GaussianMessage(Message):
    """
    Message containing Gaussian distribution information.
    All parameters must be explicitly specified.
    """
    def __init__(self, 
                 source_id: str, 
                 target_id: str,
                 variables: List[str],
                 direction: str,
                 mean: float,
                 variance: float):
        """
        Initialize a Gaussian message.
        
        Args:
            source_id: Identifier of the sending node
            target_id: Identifier of the receiving node
            variables: List of variables (must be single variable)
            direction: Message direction
            mean: Mean of the Gaussian distribution
            variance: Variance of the Gaussian distribution
        """
        super().__init__(source_id, target_id, variables, direction)
        if len(variables) != 1:
            raise ValueError("Gaussian message must involve exactly one variable")
        self.mean = mean
        self.variance = variance
        self.logger.debug(f"Created GaussianMessage from {source_id} to {target_id}")

    def validate(self) -> ValidationResult:
        """Validate Gaussian message content."""
        basic_result = super().validate()
        if not basic_result.is_valid:
            return basic_result

        # Validate variance
        if self.variance <= 0:
            return ValidationResult(
                False,
                f"Variance must be positive, got {self.variance}"
            )

        self._validated = True
        return ValidationResult(True, "Validation successful")

    def combine(self, other: 'Message') -> 'Message':
        """Combine with another Gaussian message."""
        super().combine(other)
        
        if not isinstance(other, GaussianMessage):
            raise ValueError("Can only combine Gaussian messages with other Gaussian messages")

        # Precision-weighted combination for numerical stability
        precision1 = 1.0 / self.variance
        precision2 = 1.0 / other.variance
        new_precision = precision1 + precision2
        new_variance = 1.0 / new_precision
        new_mean = (precision1 * self.mean + precision2 * other.mean) / new_precision

        result = GaussianMessage(
            f"{self.source_id}_{other.source_id}",
            self.target_id,
            self.variables,
            self.direction,
            new_mean,
            new_variance
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Message combination resulted in invalid message: {validation.message}")

        return result

    def marginalize(self, variables: List[str]) -> 'Message':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # For univariate Gaussian, marginalization over the variable
        # results in a constant message with value 1
        if self.variables[0] in variables:
            return ConstantMessage(
                self.source_id,
                self.target_id,
                [],
                self.direction,
                1.0
            )
        return self

    def get_type(self) -> MessageType:
        """Get message type."""
        return MessageType.GAUSSIAN

class TruncatedGaussianMessage(Message):
    """
    Message containing truncated Gaussian distribution information.
    All parameters including bounds must be explicitly specified.
    """
    def __init__(self, 
                 source_id: str, 
                 target_id: str,
                 variables: List[str],
                 direction: str,
                 mean: float,
                 variance: float,
                 lower_bound: float,
                 upper_bound: float):
        """
        Initialize a truncated Gaussian message.
        
        Args:
            source_id: Identifier of the sending node
            target_id: Identifier of the receiving node
            variables: List of variables (must be single variable)
            direction: Message direction
            mean: Mean of the underlying Gaussian
            variance: Variance of the underlying Gaussian
            lower_bound: Lower truncation bound
            upper_bound: Upper truncation bound
        """
        super().__init__(source_id, target_id, variables, direction)
        if len(variables) != 1:
            raise ValueError("Truncated Gaussian message must involve exactly one variable")
        self.mean = mean
        self.variance = variance
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.logger.debug(f"Created TruncatedGaussianMessage from {source_id} to {target_id}")

    def validate(self) -> ValidationResult:
        """Validate truncated Gaussian message content."""
        basic_result = super().validate()
        if not basic_result.is_valid:
            return basic_result

        # Validate variance
        if self.variance <= 0:
            return ValidationResult(
                False,
                f"Variance must be positive, got {self.variance}"
            )

        # Validate bounds
        if self.lower_bound >= self.upper_bound:
            return ValidationResult(
                False,
                f"Lower bound ({self.lower_bound}) must be less than "
                f"upper bound ({self.upper_bound})"
            )

        self._validated = True
        return ValidationResult(True, "Validation successful")

    def combine(self, other: 'Message') -> 'Message':
        """Combine with another truncated Gaussian message."""
        super().combine(other)
        
        if not isinstance(other, TruncatedGaussianMessage):
            raise ValueError("Can only combine truncated Gaussian messages with other truncated Gaussian messages")

        # New bounds are intersection of intervals
        new_lower = max(self.lower_bound, other.lower_bound)
        new_upper = min(self.upper_bound, other.upper_bound)

        if new_lower >= new_upper:
            raise ValueError("Message combination results in empty interval")

        # Precision-weighted combination for numerical stability
        precision1 = 1.0 / self.variance
        precision2 = 1.0 / other.variance
        new_precision = precision1 + precision2
        new_variance = 1.0 / new_precision
        new_mean = (precision1 * self.mean + precision2 * other.mean) / new_precision

        result = TruncatedGaussianMessage(
            f"{self.source_id}_{other.source_id}",
            self.target_id,
            self.variables,
            self.direction,
            new_mean,
            new_variance,
            new_lower,
            new_upper
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Message combination resulted in invalid message: {validation.message}")

        return result

    def marginalize(self, variables: List[str]) -> 'Message':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # For univariate truncated Gaussian, marginalization over the variable
        # results in a constant message with value 1
        if self.variables[0] in variables:
            return ConstantMessage(
                self.source_id,
                self.target_id,
                [],
                self.direction,
                1.0
            )
        return self

    def get_type(self) -> MessageType:
        """Get message type."""
        return MessageType.TRUNCATED_GAUSSIAN

class MultivariateGaussianMessage(Message):
    """
    Message containing multivariate Gaussian distribution information.
    All parameters must be explicitly specified.
    """
    def __init__(self, 
                 source_id: str, 
                 target_id: str,
                 variables: List[str],
                 direction: str,
                 mean: np.ndarray,
                 covariance: np.ndarray):
        """
        Initialize a multivariate Gaussian message.
        
        Args:
            source_id: Identifier of the sending node
            target_id: Identifier of the receiving node
            variables: List of variables
            direction: Message direction
            mean: Mean vector
            covariance: Covariance matrix
        """
        super().__init__(source_id, target_id, variables, direction)
        self.mean = np.array(mean)
        self.covariance = np.array(covariance)
        if self.mean.shape != (len(variables),):
            raise ValueError(f"Mean vector dimension {self.mean.shape} does not match "
                           f"number of variables {len(variables)}")
        if self.covariance.shape != (len(variables), len(variables)):
            raise ValueError(f"Covariance matrix shape {self.covariance.shape} does not match "
                           f"number of variables {len(variables)}")
        self.logger.debug(f"Created MultivariateGaussianMessage from {source_id} to {target_id}")

    def validate(self) -> ValidationResult:
        """Validate multivariate Gaussian message content."""
        basic_result = super().validate()
        if not basic_result.is_valid:
            return basic_result

        # Check symmetry of covariance matrix
        if not np.allclose(self.covariance, self.covariance.T):
            return ValidationResult(
                False,
                "Covariance matrix must be symmetric"
            )

        # Check positive definiteness using Cholesky decomposition
        try:
            np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError:
            return ValidationResult(
                False,
                "Covariance matrix must be positive definite"
            )

        self._validated = True
        return ValidationResult(True, "Validation successful")

    def combine(self, other: 'Message') -> 'Message':
        """Combine with another multivariate Gaussian message."""
        super().combine(other)
        
        if not isinstance(other, MultivariateGaussianMessage):
            raise ValueError("Can only combine multivariate Gaussian messages with other multivariate Gaussian messages")

        # Ensure same variable ordering
        if self.variables != other.variables:
            other = other._reorder_variables(self.variables)

        try:
            # Convert to precision form for stable multiplication
            precision1 = np.linalg.inv(self.covariance)
            precision2 = np.linalg.inv(other.covariance)
            
            # New precision and covariance
            new_precision = precision1 + precision2
            new_covariance = np.linalg.inv(new_precision)
            
            # New mean using precision-weighted combination
            new_mean = new_covariance @ (precision1 @ self.mean + precision2 @ other.mean)
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error in matrix operations: {str(e)}")

        result = MultivariateGaussianMessage(
            f"{self.source_id}_{other.source_id}",
            self.target_id,
            self.variables,
            self.direction,
            new_mean,
            new_covariance
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Message combination resulted in invalid message: {validation.message}")

        return result

    def marginalize(self, variables: List[str]) -> 'Message':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # Get indices of variables to keep and marginalize
        keep_idx = [i for i, var in enumerate(self.variables) if var not in variables]
        
        if not keep_idx:  # Marginalizing all variables
            return ConstantMessage(
                self.source_id,
                self.target_id,
                [],
                self.direction,
                1.0
            )

        # Extract relevant parts of mean and covariance
        new_mean = self.mean[keep_idx]
        new_cov = self.covariance[np.ix_(keep_idx, keep_idx)]
        remaining_vars = [var for var in self.variables if var not in variables]

        result = MultivariateGaussianMessage(
            self.source_id,
            self.target_id,
            remaining_vars,
            self.direction,
            new_mean,
            new_cov
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Marginalization resulted in invalid message: {validation.message}")

        return result

    def _reorder_variables(self, new_order: List[str]) -> 'MultivariateGaussianMessage':
        """Create a new message with variables in the specified order."""
        # Get permutation indices
        perm = [self.variables.index(var) for var in new_order]
        
        # Reorder mean and covariance
        new_mean = self.mean[perm]
        new_cov = self.covariance[np.ix_(perm, perm)]
        
        return MultivariateGaussianMessage(
            self.source_id,
            self.target_id,
            new_order,
            self.direction,
            new_mean,
            new_cov
        )

    def get_type(self) -> MessageType:
        """Get message type."""
        return MessageType.MULTIVARIATE_GAUSSIAN
