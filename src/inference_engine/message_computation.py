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
