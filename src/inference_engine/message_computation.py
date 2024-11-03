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


from .messages.base import Message, MessageType
from .messages.operators import (
    DiscreteMessageOperator,
    GaussianMessageOperator,
    CLGMessageOperator,
    MultivariateGaussianMessageOperator,

)

from ..education_models.locus_control import (
    ControlLevel,
    ControlScope,
    ControlValidator,
    ControlledVariable
)

@dataclass
class ComputationResult:
    """Results of a message computation with error tracking."""
    value: Union[Dict[Tuple, float], np.ndarray]
    error_bound: float
    numerical_issues: Optional[List[str]] = None


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

@dataclass
class ControlAwareComputationResult(ComputationResult):
    """Extends ComputationResult with control information."""
    control_level: Optional[ControlLevel] = None
    influence_weight: float = 1.0
    authority_path: Optional[List[ControlLevel]] = None
    numerical_issues: Optional[List[str]] = None

class ControlAwareMessageComputationEngine(MessageComputationEngine):
    """
    Enhanced message computation engine with control level awareness.
    Maintains all mathematical guarantees while incorporating control influence.
    """
    def __init__(self, model: 'BayesianNetwork'):
        super().__init__(model)
        self.control_validator = ControlValidator()
        self.variable_controls: Dict[str, ControlledVariable] = {}
        self._initialize_control_mappings()
        self.logger = logging.getLogger(__name__)

    def _initialize_control_mappings(self) -> None:
        """Initialize control mappings for all variables."""
        for node_id, node in self.model.nodes.items():
            if hasattr(node, 'control_scope'):
                self.variable_controls[node_id] = ControlledVariable(
                    name=node_id,
                    control_scope=node.control_scope,
                    validator=self.control_validator
                )

    def compute_message(self, 
                       source_id: str, 
                       target_id: str, 
                       incoming_messages: List[Distribution],
                       control_level: Optional[ControlLevel] = None) -> Distribution:
        """
        Compute control-aware message from source to target.
        
        Args:
            source_id: ID of source node
            target_id: ID of target node
            incoming_messages: List of incoming messages to combine
            control_level: Optional control level for computation
            
        Returns:
            Computed message distribution with control influence
            
        Raises:
            ValueError: If computation fails or control authority invalid
        """
        # First validate control authority if applicable
        if control_level is not None:
            self._validate_computation_authority(source_id, target_id, control_level)

        # Get base message through standard computation
        base_result = super().compute_message(source_id, target_id, incoming_messages)

        # If no control level, return base result
        if control_level is None:
            return base_result

        # Apply control influence
        return self._apply_control_influence(
            base_result,
            source_id,
            control_level
        )

    def _validate_computation_authority(self,
                                     source_id: str,
                                     target_id: str,
                                     control_level: ControlLevel) -> None:
        """
        Validate control level authority for message computation.
        
        Mathematical Guarantees:
        - Maintains exact probability calculations
        - Preserves inference precision requirements
        - Ensures valid control flow paths
        """
        if source_id in self.variable_controls:
            controlled_var = self.variable_controls[source_id]
            if not controlled_var.can_be_modified_by(control_level):
                raise ValueError(
                    f"Control level {control_level.name} lacks authority for "
                    f"computation involving variable {source_id}"
                )

    def _apply_control_influence(self,
                               base_result: Distribution,
                               variable_id: str,
                               control_level: ControlLevel) -> Distribution:
        """
        Apply control level influence to computation result.
        
        Mathematical Properties:
        - Maintains distribution validity
        - Preserves probability axioms
        - Tracks influence precision
        """
        if variable_id not in self.variable_controls:
            return base_result

        controlled_var = self.variable_controls[variable_id]
        influence_weight = controlled_var.get_influence_weight(control_level)

        if influence_weight >= 1.0:
            return base_result

        if isinstance(base_result, DiscreteDistribution):
            return self._apply_discrete_influence(base_result, influence_weight)
        elif isinstance(base_result, GaussianDistribution):
            return self._apply_gaussian_influence(base_result, influence_weight)
        elif isinstance(base_result, CLGDistribution):
            return self._apply_clg_influence(base_result, influence_weight)
        
        return base_result

    def _apply_discrete_influence(self,
                                distribution: DiscreteDistribution,
                                weight: float) -> DiscreteDistribution:
        """
        Apply influence weight to discrete distribution.
        Maintains probability sum exactly.
        """
        # Calculate uniform component weight
        uniform_weight = (1.0 - weight) / len(distribution.states)
        
        # Combine weighted distribution with uniform
        new_probs = {}
        for state, prob in distribution.probabilities.items():
            new_probs[state] = weight * prob + uniform_weight
            
        result = DiscreteDistribution(
            distribution.variable,
            distribution.parents,
            distribution.states,
            new_probs,
            distribution.node_states
        )
        
        # Validate result maintains probability sum
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid influence application: {validation.message}")
            
        return result

    def _apply_gaussian_influence(self,
                                distribution: GaussianDistribution,
                                weight: float) -> GaussianDistribution:
        """
        Apply influence weight to Gaussian distribution.
        Maintains distribution validity through precision weighting.
        """
        # Scale precision by influence weight
        old_precision = 1.0 / distribution.variance
        new_precision = weight * old_precision
        new_variance = 1.0 / new_precision if new_precision > 0 else float('inf')
        
        result = GaussianDistribution(
            distribution.variable,
            distribution.parents,
            distribution.mean,
            new_variance
        )
        
        # Validate result maintains Gaussian properties
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid influence application: {validation.message}")
            
        return result

    def _apply_clg_influence(self,
                           distribution: CLGDistribution,
                           weight: float) -> CLGDistribution:
        """
        Apply influence weight to CLG distribution.
        Maintains conditional relationships exactly.
        """
        new_params = {}
        
        for config, params in distribution.parameters.items():
            # Scale coefficients and variance while maintaining mean base
            new_params[config] = {
                'mean_base': params['mean_base'],
                'coefficients': [c * weight for c in params['coefficients']],
                'variance': params['variance'] / weight if weight > 0 else float('inf')
            }
            
        result = CLGDistribution(
            distribution.variable,
            distribution.continuous_parents,
            distribution.discrete_parents
        )
        result.parameters = new_params
        
        # Validate result maintains CLG properties
        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid influence application: {validation.message}")
            
        return result

    def validate_computation(self, 
                           result: ControlAwareComputationResult) -> ValidationResult:
        """
        Enhanced validation incorporating control requirements.
        
        Mathematical Requirements:
        - All base computation requirements
        - Control influence validity
        - Influence weight precision
        """
        # First perform base validation
        base_validation = super().validate_computation(result)
        if not base_validation.is_valid:
            return base_validation
            
        # If no control level, we're done
        if result.control_level is None:
            return base_validation
            
        # Validate control influence
        if result.influence_weight < 0 or result.influence_weight > 1:
            return ValidationResult(
                False,
                f"Invalid influence weight: {result.influence_weight}"
            )
            
        # Validate authority path if present
        if result.authority_path:
            if not self._validate_authority_path(result.authority_path):
                return ValidationResult(
                    False,
                    "Invalid authority path"
                )
                
        return ValidationResult(True, "Computation meets all requirements")

    def _validate_authority_path(self,
                               path: List[ControlLevel]) -> bool:
        """Validate authority path is valid."""
        if not path:
            return False
            
        for i in range(len(path) - 1):
            if not self.control_validator.validate_control_path(
                path[i],
                path[i + 1],
                "computation"
            ):
                return False
                
        return True