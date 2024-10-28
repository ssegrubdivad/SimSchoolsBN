# src/inference_engine/message_computation.py

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from src.network_structure.bayesian_network import BayesianNetwork
from src.probability_distribution import (
    Distribution, DiscreteDistribution, GaussianDistribution,
    CLGDistribution, MultivariateGaussianDistribution
)
from src.validation import ValidationResult

@dataclass
class ComputationResult:
    """Results of a message computation with error tracking."""
    value: Union[Dict[Tuple, float], np.ndarray]
    error_bound: float
    numerical_issues: Optional[List[str]] = None

class MessageOperator(ABC):
    """
    Abstract base class for message operations.
    Ensures mathematical rigor in all computations.
    """
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.error_threshold = 1e-10

    @abstractmethod
    def combine(self, message1: Distribution, message2: Distribution) -> ComputationResult:
        """
        Combine two messages with strict error tracking.
        
        Args:
            message1: First message distribution
            message2: Second message distribution
            
        Returns:
            ComputationResult containing combined message and error bounds
            
        Raises:
            ValueError: If messages are incompatible or computation fails
        """
        pass

    @abstractmethod
    def marginalize(self, message: Distribution, 
                   variables: List[str]) -> ComputationResult:
        """
        Marginalize variables from message with error tracking.
        
        Args:
            message: Distribution to marginalize
            variables: Variables to marginalize out
            
        Returns:
            ComputationResult containing marginalized message and error bounds
            
        Raises:
            ValueError: If marginalization is invalid or computation fails
        """
        pass

    def validate_computation(self, result: ComputationResult) -> ValidationResult:
        """
        Validate computation results against error bounds.
        
        Args:
            result: Computation result to validate
            
        Returns:
            ValidationResult indicating whether results meet precision requirements
        """
        if result.error_bound > self.error_threshold:
            return ValidationResult(
                False,
                f"Computation error {result.error_bound} exceeds threshold {self.error_threshold}",
                {"numerical_issues": result.numerical_issues}
            )
        return ValidationResult(True, "Computation meets precision requirements")

class DiscreteMessageOperator(MessageOperator):
    """
    Handles operations on discrete messages with exact computation.
    """
    def combine(self, message1: DiscreteDistribution, 
                message2: DiscreteDistribution) -> ComputationResult:
        """Combine discrete messages with exact probability computation."""
        if not isinstance(message1, DiscreteDistribution) or \
           not isinstance(message2, DiscreteDistribution):
            raise ValueError("Both messages must be discrete distributions")

        # Track numerical issues
        numerical_issues = []
        
        # Combine probabilities in log space for numerical stability
        result_probs = {}
        max_log_prob = float('-inf')
        
        for states1, prob1 in message1.probabilities.items():
            for states2, prob2 in message2.probabilities.items():
                if prob1 <= 0 or prob2 <= 0:
                    continue
                    
                # Compute in log space
                log_prob = np.log(prob1) + np.log(prob2)
                max_log_prob = max(max_log_prob, log_prob)
                result_probs[states1 + states2] = log_prob

        # Convert back from log space with normalization
        total_prob = 0.0
        for states in result_probs:
            # Subtract max_log_prob for numerical stability
            result_probs[states] = np.exp(result_probs[states] - max_log_prob)
            total_prob += result_probs[states]

        # Normalize and compute error bound
        error_bound = 0.0
        for states in result_probs:
            normalized_prob = result_probs[states] / total_prob
            old_prob = result_probs[states]
            result_probs[states] = normalized_prob
            
            # Track normalization error
            error_bound = max(error_bound, abs(normalized_prob - old_prob))
            
            if normalized_prob < 1e-15:
                numerical_issues.append(f"Very small probability for states {states}")

        return ComputationResult(result_probs, error_bound, numerical_issues)

    def marginalize(self, message: DiscreteDistribution, 
                   variables: List[str]) -> ComputationResult:
        """Marginalize discrete distribution with exact summation."""
        if not isinstance(message, DiscreteDistribution):
            raise ValueError("Message must be a discrete distribution")

        # Track numerical issues
        numerical_issues = []
        
        # Determine variables to keep
        keep_vars = [v for v in message.variables if v not in variables]
        if not keep_vars:
            raise ValueError("Cannot marginalize all variables")

        # Create new probability table
        result_probs = {}
        max_prob = 0.0
        
        # Group probabilities by kept variables and sum
        for states, prob in message.probabilities.items():
            keep_states = tuple(states[message.variables.index(v)] for v in keep_vars)
            result_probs[keep_states] = result_probs.get(keep_states, 0.0) + prob
            max_prob = max(max_prob, result_probs[keep_states])

        # Track error from summation
        error_bound = 0.0
        for states, prob in result_probs.items():
            # Estimate error from floating point summation
            error_bound = max(error_bound, abs(prob * np.finfo(float).eps))
            
            if prob < 1e-15:
                numerical_issues.append(f"Very small probability for states {states}")

        return ComputationResult(result_probs, error_bound, numerical_issues)

class GaussianMessageOperator(MessageOperator):
    """
    Handles operations on Gaussian messages with precise computation.
    """
    def combine(self, message1: GaussianDistribution, 
                message2: GaussianDistribution) -> ComputationResult:
        """Combine Gaussian messages using precision-weighted parameters."""
        if not isinstance(message1, GaussianDistribution) or \
           not isinstance(message2, GaussianDistribution):
            raise ValueError("Both messages must be Gaussian distributions")

        numerical_issues = []
        
        # Convert to precision form for numerical stability
        try:
            precision1 = 1.0 / message1.variance
            precision2 = 1.0 / message2.variance
            
            # Check precision matrix conditioning
            if max(precision1, precision2) / min(precision1, precision2) > 1e13:
                numerical_issues.append("Poor conditioning in precision combination")
            
            # Combine precisions
            new_precision = precision1 + precision2
            new_variance = 1.0 / new_precision
            
            # Combine means
            new_mean = (precision1 * message1.mean + precision2 * message2.mean) / new_precision
            
            # Estimate error bound based on condition number
            error_bound = np.finfo(float).eps * (
                abs(new_mean) + abs(new_variance)
            ) * max(precision1, precision2) / min(precision1, precision2)
            
        except ZeroDivisionError:
            raise ValueError("Zero or infinite variance encountered")

        return ComputationResult(
            {"mean": new_mean, "variance": new_variance},
            error_bound,
            numerical_issues
        )

    def marginalize(self, message: GaussianDistribution, 
                   variables: List[str]) -> ComputationResult:
        """
        Marginalize Gaussian distribution.
        For univariate Gaussian, marginalization is trivial if not marginalizing
        the only variable.
        """
        if not isinstance(message, GaussianDistribution):
            raise ValueError("Message must be a Gaussian distribution")

        if message.variables[0] in variables:
            # Marginalizing out the variable results in a constant
            return ComputationResult(
                {"mean": 0.0, "variance": float('inf')},
                0.0,
                ["Marginalized all variables"]
            )

        # If not marginalizing the variable, return unchanged
        return ComputationResult(
            {"mean": message.mean, "variance": message.variance},
            0.0,
            None
        )

class MessageComputationEngine:
    """
    Main engine for message computations.
    Ensures exact computations with error tracking.
    """
    def __init__(self, model: BayesianNetwork):
        """
        Initialize computation engine.
        
        Args:
            model: The Bayesian network model
        """
        self.model = model
        self.operators = {
            DiscreteDistribution: DiscreteMessageOperator(),
            GaussianDistribution: GaussianMessageOperator(),
            # Additional operators to be implemented
        }
        self.logger = logging.getLogger(__name__)

    def compute_message(self, 
                       source_id: str, 
                       target_id: str, 
                       incoming_messages: List[Distribution]) -> Distribution:
        """
        Compute message to be sent from source to target.
        
        Args:
            source_id: ID of source node
            target_id: ID of target node
            incoming_messages: List of incoming messages to combine
            
        Returns:
            Computed message distribution
            
        Raises:
            ValueError: If computation fails or precision requirements not met
        """
        if not incoming_messages:
            return self._create_uniform_message(source_id, target_id)

        # Combine incoming messages
        result = self._combine_messages(incoming_messages)
        
        # Marginalize appropriate variables
        vars_to_remove = [
            var for var in result.variables 
            if var != self.model.nodes[target_id].id
        ]
        if vars_to_remove:
            result = self._marginalize_message(result, vars_to_remove)

        # Validate final result
        operator = self._get_operator(type(result))
        validation = operator.validate_computation(result)
        if not validation.is_valid:
            raise ValueError(f"Message computation failed validation: {validation.message}")

        return result

    def _combine_messages(self, 
                         messages: List[Distribution]) -> Distribution:
        """
        Combine multiple messages with error tracking.
        
        Args:
            messages: List of messages to combine
            
        Returns:
            Combined message distribution
            
        Raises:
            ValueError: If combination fails or precision requirements not met
        """
        if not messages:
            raise ValueError("No messages to combine")

        result = messages[0]
        for message in messages[1:]:
            operator = self._get_operator(type(result))
            computation = operator.combine(result, message)
            
            # Validate combination
            validation = operator.validate_computation(computation)
            if not validation.is_valid:
                raise ValueError(f"Message combination failed: {validation.message}")
                
            result = self._create_distribution(type(result), computation)

        return result

    def _marginalize_message(self, 
                           message: Distribution,
                           variables: List[str]) -> Distribution:
        """
        Marginalize variables from message with error tracking.
        
        Args:
            message: Distribution to marginalize
            variables: Variables to marginalize out
            
        Returns:
            Marginalized distribution
            
        Raises:
            ValueError: If marginalization fails or precision requirements not met
        """
        operator = self._get_operator(type(message))
        computation = operator.marginalize(message, variables)
        
        # Validate marginalization
        validation = operator.validate_computation(computation)
        if not validation.is_valid:
            raise ValueError(f"Message marginalization failed: {validation.message}")

        return self._create_distribution(type(message), computation)

    def _get_operator(self, message_type: type) -> MessageOperator:
        """Get appropriate operator for message type."""
        if message_type not in self.operators:
            raise ValueError(f"No operator available for message type: {message_type}")
        return self.operators[message_type]

    def _create_distribution(self, 
                           dist_type: type,
                           computation: ComputationResult) -> Distribution:
        """Create distribution from computation result."""
        if dist_type == DiscreteDistribution:
            return DiscreteDistribution(
                variables=[],  # To be filled based on computation
                probabilities=computation.value
            )
        elif dist_type == GaussianDistribution:
            return GaussianDistribution(
                variables=[],  # To be filled based on computation
                mean=computation.value["mean"],
                variance=computation.value["variance"]
            )
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def _create_uniform_message(self, 
                              source_id: str,
                              target_id: str) -> Distribution:
        """Create appropriate uniform message based on variable type."""
        target_node = self.model.nodes[target_id]
        if isinstance(target_node.distribution, DiscreteDistribution):
            states = target_node.distribution.states
            prob = 1.0 / len(states)
            return DiscreteDistribution(
                variables=[target_id],
                probabilities={state: prob for state in states}
            )
        elif isinstance(target_node.distribution, GaussianDistribution):
            return GaussianDistribution(
                variables=[target_id],
                mean=0.0,
                variance=1e6  # Large variance for high uncertainty
            )
        else:
            raise ValueError(f"Unsupported distribution type for node {target_id}")

class CLGMessageOperator(MessageOperator):
    """
    Handles operations on Conditional Linear Gaussian messages.
    Maintains exact relationships between discrete and continuous components.
    """
    def combine(self, message1: CLGDistribution, 
                message2: CLGDistribution) -> ComputationResult:
        """
        Combine CLG messages while maintaining exact relationships.
        
        Args:
            message1: First CLG message
            message2: Second CLG message
            
        Returns:
            ComputationResult with combined discrete and continuous components
            
        Raises:
            ValueError: If messages are incompatible or computation fails
        """
        if not isinstance(message1, CLGDistribution) or \
           not isinstance(message2, CLGDistribution):
            raise ValueError("Both messages must be CLG distributions")

        numerical_issues = []
        
        # Combine discrete components first
        discrete_result = self._combine_discrete_components(
            message1.discrete_probabilities,
            message2.discrete_probabilities
        )
        
        # For each discrete configuration, combine continuous components
        continuous_params = {}
        max_error = 0.0
        
        for config in discrete_result.value.keys():
            try:
                # Get parameters for this configuration
                params1 = message1.get_conditional_parameters(config)
                params2 = message2.get_conditional_parameters(config)
                
                # Combine means and coefficients using precision weighting
                precision1 = 1.0 / params1['variance']
                precision2 = 1.0 / params2['variance']
                new_precision = precision1 + precision2
                
                if max(precision1, precision2) / min(precision1, precision2) > 1e13:
                    numerical_issues.append(
                        f"Poor conditioning in precision combination for config {config}"
                    )
                
                # Combine base means
                new_mean_base = (
                    precision1 * params1['mean_base'] + 
                    precision2 * params2['mean_base']
                ) / new_precision
                
                # Combine coefficients
                new_coefficients = []
                for c1, c2 in zip(params1['coefficients'], params2['coefficients']):
                    coef = (precision1 * c1 + precision2 * c2) / new_precision
                    new_coefficients.append(coef)
                
                # New variance
                new_variance = 1.0 / new_precision
                
                continuous_params[config] = {
                    'mean_base': new_mean_base,
                    'coefficients': new_coefficients,
                    'variance': new_variance
                }
                
                # Track maximum error
                error = np.finfo(float).eps * (
                    abs(new_mean_base) + 
                    sum(abs(c) for c in new_coefficients) + 
                    abs(new_variance)
                ) * max(precision1, precision2) / min(precision1, precision2)
                
                max_error = max(max_error, error)
                
            except ZeroDivisionError:
                raise ValueError(f"Zero or infinite variance encountered for config {config}")
        
        return ComputationResult(
            {
                'discrete': discrete_result.value,
                'continuous': continuous_params
            },
            max(discrete_result.error_bound, max_error),
            numerical_issues + (discrete_result.numerical_issues or [])
        )

    def marginalize(self, message: CLGDistribution, 
                   variables: List[str]) -> ComputationResult:
        """
        Marginalize variables from CLG message.
        Handles discrete and continuous variables separately while maintaining relationships.
        """
        if not isinstance(message, CLGDistribution):
            raise ValueError("Message must be a CLG distribution")

        numerical_issues = []
        
        # Separate variables by type
        discrete_vars = [v for v in variables if v in message.discrete_variables]
        continuous_vars = [v for v in variables if v in message.continuous_variables]
        
        # Marginalize discrete part first
        if discrete_vars:
            discrete_result = self._marginalize_discrete_components(
                message.discrete_probabilities,
                discrete_vars
            )
        else:
            discrete_result = ComputationResult(
                message.discrete_probabilities,
                0.0,
                None
            )
        
        # For each remaining discrete configuration, marginalize continuous part
        continuous_params = {}
        max_error = 0.0
        
        for config in discrete_result.value.keys():
            try:
                params = message.get_conditional_parameters(config)
                
                if continuous_vars:
                    # Remove coefficients for marginalized continuous variables
                    remaining_indices = [
                        i for i, var in enumerate(message.continuous_variables)
                        if var not in continuous_vars
                    ]
                    
                    new_coefficients = [params['coefficients'][i] for i in remaining_indices]
                    
                    # Adjust variance for marginalized variables
                    marginalized_variance = sum(
                        params['coefficients'][i]**2
                        for i in range(len(params['coefficients']))
                        if i not in remaining_indices
                    ) * params['variance']
                    
                    new_variance = params['variance'] + marginalized_variance
                    
                    continuous_params[config] = {
                        'mean_base': params['mean_base'],
                        'coefficients': new_coefficients,
                        'variance': new_variance
                    }
                    
                    # Track error from variance adjustment
                    error = np.finfo(float).eps * (
                        abs(params['mean_base']) +
                        sum(abs(c) for c in new_coefficients) +
                        abs(new_variance)
                    )
                else:
                    continuous_params[config] = params
                    error = 0.0
                    
                max_error = max(max_error, error)
                
            except Exception as e:
                raise ValueError(f"Error marginalizing continuous components for config {config}: {str(e)}")
        
        return ComputationResult(
            {
                'discrete': discrete_result.value,
                'continuous': continuous_params
            },
            max(discrete_result.error_bound, max_error),
            numerical_issues + (discrete_result.numerical_issues or [])
        )

    def _combine_discrete_components(self, probs1: Dict, probs2: Dict) -> ComputationResult:
        """Helper method to combine discrete components of CLG messages."""
        # Use log space for numerical stability
        result_probs = {}
        max_log_prob = float('-inf')
        numerical_issues = []
        
        for states1, prob1 in probs1.items():
            for states2, prob2 in probs2.items():
                if prob1 <= 0 or prob2 <= 0:
                    continue
                    
                log_prob = np.log(prob1) + np.log(prob2)
                max_log_prob = max(max_log_prob, log_prob)
                result_probs[self._combine_states(states1, states2)] = log_prob

        # Convert back from log space with normalization
        total_prob = 0.0
        for states in result_probs:
            result_probs[states] = np.exp(result_probs[states] - max_log_prob)
            total_prob += result_probs[states]

        error_bound = 0.0
        for states in result_probs:
            normalized_prob = result_probs[states] / total_prob
            error_bound = max(error_bound, abs(normalized_prob - result_probs[states]))
            result_probs[states] = normalized_prob
            
            if normalized_prob < 1e-15:
                numerical_issues.append(f"Very small probability for states {states}")

        return ComputationResult(result_probs, error_bound, numerical_issues)

    def _marginalize_discrete_components(self, probs: Dict, 
                                      variables: List[str]) -> ComputationResult:
        """Helper method to marginalize discrete components of CLG messages."""
        result_probs = {}
        numerical_issues = []
        max_prob = 0.0
        
        # Group and sum probabilities
        for states, prob in probs.items():
            new_states = self._remove_variables_from_states(states, variables)
            result_probs[new_states] = result_probs.get(new_states, 0.0) + prob
            max_prob = max(max_prob, result_probs[new_states])

        # Track error from summation
        error_bound = 0.0
        for states, prob in result_probs.items():
            error_bound = max(error_bound, abs(prob * np.finfo(float).eps))
            
            if prob < 1e-15:
                numerical_issues.append(f"Very small probability for states {states}")

        return ComputationResult(result_probs, error_bound, numerical_issues)

    @staticmethod
    def _combine_states(states1: tuple, states2: tuple) -> tuple:
        """Helper method to combine state tuples."""
        return states1 + states2

    @staticmethod
    def _remove_variables_from_states(states: tuple, variables: List[str]) -> tuple:
        """Helper method to remove variables from state tuple."""
        return tuple(s for s, v in zip(states, variables) if v not in variables)


class MultivariateGaussianMessageOperator(MessageOperator):
    """
    Handles operations on multivariate Gaussian messages.
    Maintains numerical stability in matrix operations.
    """
    def combine(self, message1: MultivariateGaussianDistribution, 
                message2: MultivariateGaussianDistribution) -> ComputationResult:
        """
        Combine multivariate Gaussian messages using precision form.
        
        Args:
            message1: First multivariate Gaussian message
            message2: Second multivariate Gaussian message
            
        Returns:
            ComputationResult with combined parameters
            
        Raises:
            ValueError: If messages are incompatible or computation fails
        """
        if not isinstance(message1, MultivariateGaussianDistribution) or \
           not isinstance(message2, MultivariateGaussianDistribution):
            raise ValueError("Both messages must be multivariate Gaussian distributions")

        numerical_issues = []
        
        try:
            # Convert to precision form
            precision1 = np.linalg.inv(message1.covariance)
            precision2 = np.linalg.inv(message2.covariance)
            
            # Check conditioning
            cond1 = np.linalg.cond(precision1)
            cond2 = np.linalg.cond(precision2)
            
            if max(cond1, cond2) > 1e13:
                numerical_issues.append("Poor conditioning in precision matrices")
            
            # Combine precisions
            new_precision = precision1 + precision2
            
            # Use Cholesky decomposition for stable inverse
            try:
                L = np.linalg.cholesky(new_precision)
                new_covariance = self._solve_triangular_system(L)
            except np.linalg.LinAlgError:
                numerical_issues.append("Failed Cholesky decomposition, using eigendecomposition")
                new_covariance = self._stable_inverse(new_precision)
            
            # Combine means
            new_mean = new_covariance @ (
                precision1 @ message1.mean + precision2 @ message2.mean
            )
            
            # Estimate error bound
            error_bound = np.finfo(float).eps * (
                np.linalg.norm(new_mean) +
                np.linalg.norm(new_covariance, 'fro')
            ) * max(cond1, cond2)
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Matrix operation failed: {str(e)}")

        return ComputationResult(
            {
                'mean': new_mean,
                'covariance': new_covariance
            },
            error_bound,
            numerical_issues
        )

    def marginalize(self, message: MultivariateGaussianDistribution, 
                   variables: List[str]) -> ComputationResult:
        """
        Marginalize variables from multivariate Gaussian message.
        
        Args:
            message: Multivariate Gaussian message to marginalize
            variables: Variables to marginalize out
            
        Returns:
            ComputationResult with marginalized parameters
            
        Raises:
            ValueError: If marginalization fails
        """
        if not isinstance(message, MultivariateGaussianDistribution):
            raise ValueError("Message must be a multivariate Gaussian distribution")

        numerical_issues = []
        
        # Get indices of variables to keep and marginalize
        keep_idx = [
            i for i, var in enumerate(message.variables)
            if var not in variables
        ]
        
        if not keep_idx:
            raise ValueError("Cannot marginalize all variables")

        try:
            # Extract relevant parts of mean and covariance
            new_mean = message.mean[keep_idx]
            new_covariance = message.covariance[np.ix_(keep_idx, keep_idx)]
            
            # Check conditioning of marginalized covariance
            condition_number = np.linalg.cond(new_covariance)
            if condition_number > 1e13:
                numerical_issues.append("Poor conditioning in marginalized covariance")
            
            # Estimate error bound
            error_bound = np.finfo(float).eps * (
                np.linalg.norm(new_mean) +
                np.linalg.norm(new_covariance, 'fro')
            ) * condition_number
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Matrix operation failed: {str(e)}")

        return ComputationResult(
            {
                'mean': new_mean,
                'covariance': new_covariance
            },
            error_bound,
            numerical_issues
        )

    def _solve_triangular_system(self, L: np.ndarray) -> np.ndarray:
        """Solve system using Cholesky decomposition."""
        n = L.shape[0]
        return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))

    def _stable_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Compute stable inverse using eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        # Zero out small eigenvalues
        eigvals[eigvals < 1e-13] = 0
        # Compute inverse using eigendecomposition
        return eigvecs @ np.diag(1.0 / np.where(eigvals > 0, eigvals, 1)) @ eigvecs.T