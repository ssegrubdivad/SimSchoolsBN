# src/probability_distribution/factors.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from scipy.stats import truncnorm
import logging
from dataclasses import dataclass
from numpy.linalg import LinAlgError

from .distribution import Distribution
from ..inference_engine.messages.base import ValidationResult


@dataclass
class ComputationResult:
    """Results of a factor computation with error tracking."""
    value: Union[Dict[Tuple, float], np.ndarray]
    error_bound: float
    numerical_issues: Optional[List[str]] = None

class Factor(ABC):
    """
    Abstract base class for all factor types in the Bayesian Network.
    Enforces strict validation and explicit parameter specification.
    """
    def __init__(self, name: str, variables: List[str]):
        """
        Initialize a factor.
        
        Args:
            name: Unique identifier for this factor
            variables: List of variables involved in this factor
        """
        self.name = name
        self.variables = variables
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._parameters: Dict = {}
        self._validated = False
        
        # Set up logging with high precision for numerical values
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        Validate the factor's parameters and structure.
        Must be called before any operations can be performed.
        
        Returns:
            ValidationResult containing validation status and details
        """
        pass

    @abstractmethod
    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        """
        Get the factor's value for a specific variable assignment.
        
        Args:
            assignment: Dictionary mapping variables to their values
            
        Returns:
            float: The factor value (probability or density)
            
        Raises:
            ValueError: If the factor hasn't been validated or assignment is invalid
        """
        if not self._validated:
            raise ValueError(f"Factor {self.name} must be validated before use")
        
        # Check if all required variables are present
        missing_vars = set(self.variables) - set(assignment.keys())
        if missing_vars:
            raise ValueError(f"Missing assignments for variables: {missing_vars}")

    @abstractmethod
    def multiply(self, other: 'Factor') -> 'Factor':
        """
        Multiply this factor with another factor.
        
        Args:
            other: Another factor to multiply with
            
        Returns:
            Factor: A new factor representing the product
            
        Raises:
            ValueError: If factors are incompatible or unvalidated
        """
        if not (self._validated and other._validated):
            raise ValueError("Both factors must be validated before multiplication")

    @abstractmethod
    def marginalize(self, variables: List[str]) -> 'Factor':
        """
        Marginalize out the specified variables from this factor.
        
        Args:
            variables: List of variables to marginalize out
            
        Returns:
            Factor: A new factor with specified variables marginalized out
            
        Raises:
            ValueError: If marginalization is invalid or factor unvalidated
        """
        if not self._validated:
            raise ValueError(f"Factor {self.name} must be validated before marginalization")
        
        if not set(variables).issubset(set(self.variables)):
            raise ValueError(f"Cannot marginalize variables that are not in the factor: "
                           f"{set(variables) - set(self.variables)}")

    def get_log_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        """
        Get the natural logarithm of the factor value.
        Implemented for numerical stability in products.
        
        Args:
            assignment: Dictionary mapping variables to their values
            
        Returns:
            float: The log of the factor value
            
        Raises:
            ValueError: If factor value is 0 or negative
        """
        value = self.get_value(assignment)
        if value <= 0:
            raise ValueError(f"Cannot compute log of non-positive factor value: {value}")
        return np.log(value)

    @property
    def parameters(self) -> Dict:
        """Get the factor's parameters."""
        return self._parameters.copy()  # Return a copy to prevent modification

    @parameters.setter
    def parameters(self, params: Dict):
        """
        Set the factor's parameters.
        Invalidates the factor until validate() is called again.
        
        Args:
            params: Dictionary of parameters specific to the factor type
        """
        self._parameters = params.copy()  # Store a copy to prevent external modification
        self._validated = False
        self.logger.info(f"Parameters set for factor {self.name}. Validation required.")

class DiscreteFactor(Factor):
    """
    Represents a discrete probability factor.
    All probabilities must be explicitly specified.
    """
    def __init__(self, name: str, variables: List[str], states: Dict[str, List[str]]):
        """
        Initialize a discrete factor.
        
        Args:
            name: Factor identifier
            variables: List of variables in this factor
            states: Dictionary mapping each variable to its possible states
        """
        super().__init__(name, variables)
        self.states = states
        self._probability_table: Optional[Dict[Tuple, float]] = None

    def validate(self) -> ValidationResult:
        """
        Validate the discrete factor's probability table.
        
        Returns:
            ValidationResult indicating validation status
        """
        if not self._probability_table:
            return ValidationResult(False, "Probability table not set")

        try:
            # Check completeness
            required_states = 1
            for var in self.variables:
                required_states *= len(self.states[var])
            
            if len(self._probability_table) != required_states:
                return ValidationResult(
                    False,
                    f"Incomplete probability table. Expected {required_states} entries, "
                    f"got {len(self._probability_table)}"
                )

            # Check probability sum for each conditional context
            for context in self._get_conditional_contexts():
                prob_sum = sum(self._get_probabilities_for_context(context))
                if abs(prob_sum - 1.0) > 1e-10:  # Use precise tolerance
                    return ValidationResult(
                        False,
                        f"Probabilities for context {context} sum to {prob_sum}, not 1.0"
                    )

            self._validated = True
            return ValidationResult(True, "Validation successful")

        except Exception as e:
            return ValidationResult(False, f"Validation failed: {str(e)}")

    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        """Get probability value for given assignment."""
        super().get_value(assignment)  # Call parent for basic validation
        
        # Convert assignment to tuple for table lookup
        key = tuple(assignment[var] for var in self.variables)
        if key not in self._probability_table:
            raise ValueError(f"No probability defined for assignment: {assignment}")
            
        return self._probability_table[key]

    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply with another factor."""
        super().multiply(other)  # Call parent for validation
        
        if isinstance(other, DiscreteFactor):
            return self._multiply_discrete(other)
        elif isinstance(other, (GaussianFactor, TruncatedGaussianFactor)):
            return self._multiply_continuous(other)
        else:
            raise ValueError(f"Multiplication not supported between {type(self)} and {type(other)}")

    def marginalize(self, variables: List[str]) -> 'Factor':
        """Marginalize out specified variables."""
        super().marginalize(variables)  # Call parent for validation
        
        # Create new factor with remaining variables
        remaining_vars = [v for v in self.variables if v not in variables]
        remaining_states = {v: self.states[v] for v in remaining_vars}
        
        result = DiscreteFactor(f"{self.name}_marginalized", remaining_vars, remaining_states)
        new_table = {}
        
        # For each assignment to remaining variables, sum over marginalized variables
        for assignment in self._get_all_assignments(remaining_vars):
            prob_sum = 0.0
            for marg_assignment in self._get_all_assignments(variables):
                full_assignment = {**assignment, **marg_assignment}
                prob_sum += self.get_value(full_assignment)
            new_table[tuple(assignment.values())] = prob_sum
            
        result._probability_table = new_table
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Marginalization resulted in invalid factor: {validation.message}")
            
        return result

    def _get_conditional_contexts(self) -> List[Tuple]:
        """Get all possible conditioning contexts."""
        if len(self.variables) <= 1:
            return [()]
        
        parent_vars = self.variables[1:]
        return [tuple(assignment.values()) 
                for assignment in self._get_all_assignments(parent_vars)]

    def _get_probabilities_for_context(self, context: Tuple) -> List[float]:
        """Get all probabilities for a given conditioning context."""
        child_var = self.variables[0]
        return [self._probability_table[(*state, *context)]
                for state in [(s,) for s in self.states[child_var]]]

    def _get_all_assignments(self, variables: List[str]) -> List[Dict[str, str]]:
        """Get all possible assignments for given variables."""
        if not variables:
            return [{}]
        
        var = variables[0]
        sub_assignments = self._get_all_assignments(variables[1:])
        assignments = []
        
        for state in self.states[var]:
            for sub in sub_assignments:
                assignments.append({var: state, **sub})
                
        return assignments

    def _multiply_discrete(self, other: 'DiscreteFactor') -> 'DiscreteFactor':
        """Multiply with another discrete factor."""
        # Combine variables and states
        new_vars = list(set(self.variables + other.variables))
        new_states = {**self.states, **other.states}
        
        result = DiscreteFactor(f"{self.name}_{other.name}", new_vars, new_states)
        new_table = {}
        
        # For each possible assignment to all variables
        for assignment in result._get_all_assignments(new_vars):
            # Get relevant assignments for each factor
            self_assignment = {var: assignment[var] for var in self.variables}
            other_assignment = {var: assignment[var] for var in other.variables}
            
            # Multiply probabilities
            new_table[tuple(assignment.values())] = (
                self.get_value(self_assignment) * other.get_value(other_assignment)
            )
            
        result._probability_table = new_table
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Multiplication resulted in invalid factor: {validation.message}")
            
        return result

    def _multiply_continuous(self, other: Union['GaussianFactor', 'TruncatedGaussianFactor']) -> 'MixedFactor':
        """Multiply with a continuous factor."""
        raise NotImplementedError("Mixed factor multiplication to be implemented")


class GaussianFactor(Factor):
    """
    Represents a Gaussian (Normal) distribution factor.
    All parameters must be explicitly specified.
    """
    def __init__(self, name: str, variables: List[str]):
        """
        Initialize a Gaussian factor.
        
        Args:
            name: Factor identifier
            variables: List of variables (should be single variable for univariate)
        """
        super().__init__(name, variables)
        if len(variables) != 1:
            raise ValueError("Univariate Gaussian factor must have exactly one variable")

    def validate(self) -> ValidationResult:
        """
        Validate the Gaussian factor's parameters.
        
        Returns:
            ValidationResult indicating validation status
        """
        required_params = {'mean', 'variance'}
        
        # Check for required parameters
        if not all(param in self._parameters for param in required_params):
            missing = required_params - set(self._parameters.keys())
            return ValidationResult(
                False,
                f"Missing required parameters: {missing}"
            )
            
        # Validate variance
        if self._parameters['variance'] <= 0:
            return ValidationResult(
                False,
                f"Variance must be positive, got {self._parameters['variance']}"
            )
            
        self._validated = True
        return ValidationResult(True, "Validation successful")

    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        """Get probability density for given assignment."""
        super().get_value(assignment)
        
        x = float(assignment[self.variables[0]])
        μ = self._parameters['mean']
        σ² = self._parameters['variance']
        
        # Standard Gaussian PDF formula
        return (1 / np.sqrt(2 * np.pi * σ²)) * np.exp(-(x - μ)**2 / (2 * σ²))

    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply with another factor."""
        super().multiply(other)
        
        if isinstance(other, GaussianFactor):
            return self._multiply_gaussian(other)
        elif isinstance(other, DiscreteFactor):
            return other._multiply_continuous(self)
        else:
            raise ValueError(f"Multiplication not supported between {type(self)} and {type(other)}")

    def marginalize(self, variables: List[str]) -> 'Factor':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # For a univariate Gaussian, marginalization over its variable
        # results in a constant factor of 1
        if self.variables[0] in variables:
            return ConstantFactor(f"{self.name}_marginalized", [], 1.0)
            
        return self

    def _multiply_gaussian(self, other: 'GaussianFactor') -> 'GaussianFactor':
        """Multiply with another Gaussian factor."""
        if self.variables != other.variables:
            raise ValueError("Can only multiply Gaussian factors over same variable")
            
        # Precision-weighted parameters for numerical stability
        σ₁² = self._parameters['variance']
        σ₂² = other._parameters['variance']
        μ₁ = self._parameters['mean']
        μ₂ = other._parameters['mean']
        
        # Combined precision
        τ₁ = 1 / σ₁²
        τ₂ = 1 / σ₂²
        τ = τ₁ + τ₂
        
        # New parameters
        new_variance = 1 / τ
        new_mean = (τ₁ * μ₁ + τ₂ * μ₂) / τ
        
        result = GaussianFactor(f"{self.name}_{other.name}", self.variables)
        result.parameters = {
            'mean': new_mean,
            'variance': new_variance
        }
        
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Multiplication resulted in invalid factor: {validation.message}")
            
        return result


class TruncatedGaussianFactor(Factor):
    """
    Represents a Truncated Gaussian distribution factor.
    All parameters including bounds must be explicitly specified.
    """
    def __init__(self, name: str, variables: List[str]):
        """
        Initialize a Truncated Gaussian factor.
        
        Args:
            name: Factor identifier
            variables: List of variables (should be single variable for univariate)
        """
        super().__init__(name, variables)
        if len(variables) != 1:
            raise ValueError("Truncated Gaussian factor must have exactly one variable")

    def validate(self) -> ValidationResult:
        """
        Validate the Truncated Gaussian factor's parameters.
        
        Returns:
            ValidationResult indicating validation status
        """
        required_params = {'mean', 'variance', 'lower', 'upper'}
        
        # Check for required parameters
        if not all(param in self._parameters for param in required_params):
            missing = required_params - set(self._parameters.keys())
            return ValidationResult(
                False,
                f"Missing required parameters: {missing}"
            )
            
        # Validate variance
        if self._parameters['variance'] <= 0:
            return ValidationResult(
                False,
                f"Variance must be positive, got {self._parameters['variance']}"
            )
            
        # Validate bounds
        if self._parameters['lower'] >= self._parameters['upper']:
            return ValidationResult(
                False,
                f"Lower bound must be less than upper bound, got "
                f"[{self._parameters['lower']}, {self._parameters['upper']}]"
            )
            
        self._validated = True
        return ValidationResult(True, "Validation successful")

    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        """Get probability density for given assignment."""
        super().get_value(assignment)
        
        x = float(assignment[self.variables[0]])
        μ = self._parameters['mean']
        σ = np.sqrt(self._parameters['variance'])
        a = self._parameters['lower']
        b = self._parameters['upper']
        
        # Check bounds
        if x < a or x > b:
            return 0.0
            
        # Standardized bounds
        alpha = (a - μ) / σ
        beta = (b - μ) / σ
        z = (x - μ) / σ
        
        # Truncated normal PDF using scipy
        return truncnorm.pdf(x, alpha, beta, loc=μ, scale=σ)

    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply with another factor."""
        super().multiply(other)
        
        if isinstance(other, TruncatedGaussianFactor):
            return self._multiply_truncated_gaussian(other)
        elif isinstance(other, DiscreteFactor):
            return other._multiply_continuous(self)
        else:
            raise ValueError(f"Multiplication not supported between {type(self)} and {type(other)}")

    def marginalize(self, variables: List[str]) -> 'Factor':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # For a univariate Truncated Gaussian, marginalization over its variable
        # results in a constant factor of 1
        if self.variables[0] in variables:
            return ConstantFactor(f"{self.name}_marginalized", [], 1.0)
            
        return self

    def _multiply_truncated_gaussian(self, other: 'TruncatedGaussianFactor') -> 'TruncatedGaussianFactor':
        """Multiply with another Truncated Gaussian factor."""
        if self.variables != other.variables:
            raise ValueError("Can only multiply Truncated Gaussian factors over same variable")
            
        # New bounds are intersection of intervals
        new_lower = max(self._parameters['lower'], other._parameters['lower'])
        new_upper = min(self._parameters['upper'], other._parameters['upper'])
        
        if new_lower >= new_upper:
            raise ValueError("Multiplication results in empty interval")
            
        # Precision-weighted parameters
        σ₁² = self._parameters['variance']
        σ₂² = other._parameters['variance']
        μ₁ = self._parameters['mean']
        μ₂ = other._parameters['mean']
        
        τ₁ = 1 / σ₁²
        τ₂ = 1 / σ₂²
        τ = τ₁ + τ₂
        
        # New parameters
        new_variance = 1 / τ
        new_mean = (τ₁ * μ₁ + τ₂ * μ₂) / τ
        
        result = TruncatedGaussianFactor(f"{self.name}_{other.name}", self.variables)
        result.parameters = {
            'mean': new_mean,
            'variance': new_variance,
            'lower': new_lower,
            'upper': new_upper
        }
        
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Multiplication resulted in invalid factor: {validation.message}")
            
        return result


class CLGFactor(Factor):
    """
    Represents a Conditional Linear Gaussian distribution factor.
    Must explicitly specify all parameters for each discrete parent configuration.
    """
    def __init__(self, name: str, continuous_var: str, continuous_parents: List[str], 
                 discrete_parents: List[str], discrete_states: Dict[str, List[str]]):
        """
        Initialize a CLG factor.
        
        Args:
            name: Factor identifier
            continuous_var: The continuous child variable
            continuous_parents: List of continuous parent variables
            discrete_parents: List of discrete parent variables
            discrete_states: Dictionary mapping discrete parents to their states
        """
        variables = [continuous_var] + continuous_parents + discrete_parents
        super().__init__(name, variables)
        self.continuous_var = continuous_var
        self.continuous_parents = continuous_parents
        self.discrete_parents = discrete_parents
        self.discrete_states = discrete_states

    def validate(self) -> ValidationResult:
        """
        Validate the CLG factor's parameters.
        
        Returns:
            ValidationResult indicating validation status
        """
        # Get all discrete parent configurations
        discrete_configs = self._get_discrete_configurations()
        
        for config in discrete_configs:
            config_key = tuple(config.values())
            
            if config_key not in self._parameters:
                return ValidationResult(
                    False,
                    f"Missing parameters for discrete configuration: {config}"
                )
                
            params = self._parameters[config_key]
            required_params = {'mean_base', 'coefficients', 'variance'}
            
            # Check required parameters
            if not all(param in params for param in required_params):
                missing = required_params - set(params.keys())
                return ValidationResult(
                    False,
                    f"Missing parameters {missing} for configuration {config}"
                )
                
            # Validate coefficient vector length
            if len(params['coefficients']) != len(self.continuous_parents):
                return ValidationResult(
                    False,
                    f"Number of coefficients ({len(params['coefficients'])}) does not match "
                    f"number of continuous parents ({len(self.continuous_parents)}) "
                    f"for configuration {config}"
                )
                
            # Validate variance
            if params['variance'] <= 0:
                return ValidationResult(
                    False,
                    f"Variance must be positive, got {params['variance']} "
                    f"for configuration {config}"
                )
        
        self._validated = True
        return ValidationResult(True, "Validation successful")

    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        """Get probability density for given assignment."""
        super().get_value(assignment)
        
        # Extract assignments
        x = float(assignment[self.continuous_var])
        cont_parent_vals = [float(assignment[p]) for p in self.continuous_parents]
        disc_config = {p: assignment[p] for p in self.discrete_parents}
        
        # Get parameters for this configuration
        config_key = tuple(disc_config.values())
        if config_key not in self._parameters:
            raise ValueError(f"No parameters for discrete configuration: {disc_config}")
            
        params = self._parameters[config_key]
        
        # Calculate mean
        mean = params['mean_base']
        for coef, val in zip(params['coefficients'], cont_parent_vals):
            mean += coef * val
            
        # Calculate density using Gaussian PDF
        variance = params['variance']
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean)**2 / (2 * variance))

    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply with another factor."""
        super().multiply(other)
        raise NotImplementedError("CLG factor multiplication to be implemented")

    def marginalize(self, variables: List[str]) -> 'Factor':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        raise NotImplementedError("CLG factor marginalization to be implemented")

    def _get_discrete_configurations(self) -> List[Dict[str, str]]:
        """Get all possible discrete parent configurations."""
        def _recursive_configs(parents: List[str]) -> List[Dict[str, str]]:
            if not parents:
                return [{}]
            
            parent = parents[0]
            sub_configs = _recursive_configs(parents[1:])
            configs = []
            
            for state in self.discrete_states[parent]:
                for sub in sub_configs:
                    configs.append({parent: state, **sub})
                    
            return configs
            
        return _recursive_configs(self.discrete_parents)


class MultivariateGaussianFactor(Factor):
    """
    Represents a Multivariate Gaussian distribution factor.
    All parameters must be explicitly specified.
    """
    def __init__(self, name: str, variables: List[str]):
        """
        Initialize a Multivariate Gaussian factor.
        
        Args:
            name: Factor identifier
            variables: List of continuous variables
        """
        super().__init__(name, variables)
        self.dimension = len(variables)

    def validate(self) -> ValidationResult:
        """
        [Previous validation checks remain the same...]
        """
        # Check positive definiteness
        try:
            # Try Cholesky decomposition - only works for positive definite matrices
            np.linalg.cholesky(self._parameters['covariance'])
        except LinAlgError:
            return ValidationResult(
                False,
                "Covariance matrix must be positive definite"
            )
            
        self._validated = True
        return ValidationResult(True, "Validation successful")

    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        """Get probability density for given assignment."""
        super().get_value(assignment)
        
        # Convert assignment to vector
        x = np.array([float(assignment[var]) for var in self.variables])
        μ = np.array(self._parameters['mean'])
        Σ = np.array(self._parameters['covariance'])
        
        # Calculate multivariate normal density
        k = self.dimension
        diff = x - μ
        
        try:
            # Use Cholesky decomposition for numerical stability
            L = np.linalg.cholesky(Σ)
            solved = np.linalg.solve(L, diff)
            quadratic = np.sum(solved**2)
            
            log_det = 2 * np.sum(np.log(np.diag(L)))
            log_density = -0.5 * (k * np.log(2 * np.pi) + log_det + quadratic)
            
            return np.exp(log_density)
            
        except LinAlgError as e:
            raise ValueError(f"Error computing density: {str(e)}")

    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply with another factor."""
        super().multiply(other)
        
        if isinstance(other, MultivariateGaussianFactor):
            return self._multiply_multivariate_gaussian(other)
        elif isinstance(other, DiscreteFactor):
            return other._multiply_continuous(self)
        else:
            raise ValueError(f"Multiplication not supported between {type(self)} and {type(other)}")

    def marginalize(self, variables: List[str]) -> 'Factor':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # Get indices of variables to keep and marginalize
        keep_idx = [i for i, var in enumerate(self.variables) if var not in variables]
        
        if not keep_idx:  # Marginalizing all variables
            return ConstantFactor(f"{self.name}_marginalized", [], 1.0)
            
        # Extract relevant parts of mean and covariance
        new_mean = self._parameters['mean'][keep_idx]
        new_cov = self._parameters['covariance'][np.ix_(keep_idx, keep_idx)]
        
        # Create new factor
        result = MultivariateGaussianFactor(
            f"{self.name}_marginalized",
            [var for var in self.variables if var not in variables]
        )
        result.parameters = {
            'mean': new_mean,
            'covariance': new_cov
        }
        
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Marginalization resulted in invalid factor: {validation.message}")
            
        return result

    def _multiply_multivariate_gaussian(self, other: 'MultivariateGaussianFactor') -> 'MultivariateGaussianFactor':
        """Multiply with another multivariate Gaussian factor."""
        if set(self.variables) != set(other.variables):
            raise ValueError("Can only multiply Multivariate Gaussian factors over same variables")
            
        # Ensure same variable ordering
        if self.variables != other.variables:
            other = other._reorder_variables(self.variables)
            
        # Convert to precision form for stable multiplication
        Σ₁ = np.array(self._parameters['covariance'])
        Σ₂ = np.array(other._parameters['covariance'])
        μ₁ = np.array(self._parameters['mean'])
        μ₂ = np.array(other._parameters['mean'])
        
        # Compute precisions
        try:
            Λ₁ = np.linalg.inv(Σ₁)
            Λ₂ = np.linalg.inv(Σ₂)
            
            # New precision and covariance
            Λ = Λ₁ + Λ₂
            new_covariance = np.linalg.inv(Λ)
            
            # New mean using precision-weighted combination
            new_mean = new_covariance @ (Λ₁ @ μ₁ + Λ₂ @ μ₂)
            
        except LinAlgError as e:
            raise ValueError(f"Error in matrix operations: {str(e)}")
            
        result = MultivariateGaussianFactor(f"{self.name}_{other.name}", self.variables)
        result.parameters = {
            'mean': new_mean,
            'covariance': new_covariance
        }
        
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Multiplication resulted in invalid factor: {validation.message}")
            
        return result

    def _reorder_variables(self, new_order: List[str]) -> 'MultivariateGaussianFactor':
        """Create a new factor with variables in the specified order."""
        # Get permutation indices
        perm = [self.variables.index(var) for var in new_order]
        
        # Reorder mean and covariance
        new_mean = self._parameters['mean'][perm]
        new_cov = self._parameters['covariance'][np.ix_(perm, perm)]
        
        result = MultivariateGaussianFactor(f"{self.name}_reordered", new_order)
        result.parameters = {
            'mean': new_mean,
            'covariance': new_cov
        }
        
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Variable reordering resulted in invalid factor: {validation.message}")
            
        return result


class ConstantFactor(Factor):
    """Represents a constant factor, useful for marginalization results."""
    def __init__(self, name: str, variables: List[str], value: float = 1.0):
        super().__init__(name, variables)
        self._value = value
        self._validated = True

    def validate(self) -> ValidationResult:
        return ValidationResult(True, "Constant factor is always valid")

    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float:
        return self._value

    def multiply(self, other: 'Factor') -> 'Factor':
        super().multiply(other)
        if isinstance(other, ConstantFactor):
            return ConstantFactor(f"{self.name}_{other.name}", [], self._value * other._value)
        return other  # Constant factor acts as multiplicative identity

    def marginalize(self, variables: List[str]) -> 'Factor':
        return self  # Marginalizing has no effect on constant factor