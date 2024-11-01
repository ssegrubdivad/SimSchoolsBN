# src/inference_engine/messages/discrete.py

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
import logging
from scipy.stats import truncnorm
from enum import Enum
import numpy as np

from .base import Message, MessageType, ValidationResult, ConstantMessage

class DiscreteMessage(Message):
    """
    Message containing discrete probability values.
    All probabilities must be explicitly specified.
    """
    def __init__(self, 
                 source_id: str, 
                 target_id: str, 
                 variables: List[str],
                 direction: str,
                 states: Dict[str, List[str]],
                 probabilities: Dict[tuple, float]):
        """
        Initialize a discrete message.
        
        Args:
            source_id: Identifier of the sending node
            target_id: Identifier of the receiving node
            variables: List of variables in the message
            direction: Message direction
            states: Dictionary mapping variables to their possible states
            probabilities: Dictionary mapping state combinations to probabilities
        """
        super().__init__(source_id, target_id, variables, direction)
        self.states = states
        self.probabilities = probabilities
        self.logger.debug(f"Created DiscreteMessage from {source_id} to {target_id}")

    def validate(self) -> ValidationResult:
        """Validate discrete message content."""
        # First perform basic validation
        basic_result = super().validate()
        if not basic_result.is_valid:
            return basic_result

        # Validate states
        if not all(var in self.states for var in self.variables):
            return ValidationResult(
                False,
                f"Missing states for variables: "
                f"{set(self.variables) - set(self.states.keys())}"
            )

        # Validate probability table completeness
        required_entries = 1
        for var in self.variables:
            required_entries *= len(self.states[var])

        if len(self.probabilities) != required_entries:
            return ValidationResult(
                False,
                f"Incomplete probability table. Expected {required_entries} entries, "
                f"got {len(self.probabilities)}"
            )

        # Validate probability sum and values
        prob_sum = sum(self.probabilities.values())
        if abs(prob_sum - 1.0) > 1e-10:
            return ValidationResult(
                False,
                f"Probabilities sum to {prob_sum}, not 1.0"
            )

        if not all(0 <= p <= 1 for p in self.probabilities.values()):
            return ValidationResult(
                False,
                "All probabilities must be between 0 and 1"
            )

        self._validated = True
        return ValidationResult(True, "Validation successful")

    def combine(self, other: 'Message') -> 'Message':
        """Combine with another discrete message."""
        super().combine(other)
        
        if not isinstance(other, DiscreteMessage):
            raise ValueError("Can only combine discrete messages with other discrete messages")

        # Create new probability table
        new_probs = {}
        for states, prob in self.probabilities.items():
            if states in other.probabilities:
                new_probs[states] = prob * other.probabilities[states]

        # Normalize probabilities
        total = sum(new_probs.values())
        if total <= 0:
            raise ValueError("Combined probabilities sum to zero")
        new_probs = {k: v/total for k, v in new_probs.items()}

        result = DiscreteMessage(
            f"{self.source_id}_{other.source_id}",
            self.target_id,
            self.variables,
            self.direction,
            self.states,
            new_probs
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Message combination resulted in invalid message: {validation.message}")

        return result

    def marginalize(self, variables: List[str]) -> 'Message':
        """Marginalize out specified variables."""
        super().marginalize(variables)
        
        # Identify variables to keep
        keep_vars = [v for v in self.variables if v not in variables]
        
        if not keep_vars:
            raise ValueError("Cannot marginalize all variables")

        # Create new probability table
        new_probs = {}
        new_states = {v: self.states[v] for v in keep_vars}

        # Group probabilities by kept variables and sum
        for states, prob in self.probabilities.items():
            keep_states = tuple(states[self.variables.index(v)] for v in keep_vars)
            new_probs[keep_states] = new_probs.get(keep_states, 0.0) + prob

        result = DiscreteMessage(
            self.source_id,
            self.target_id,
            keep_vars,
            self.direction,
            new_states,
            new_probs
        )

        validation = result.validate()
        if not validation.is_valid:
            raise ValueError(f"Marginalization resulted in invalid message: {validation.message}")

        return result

    def get_type(self) -> MessageType:
        """Get message type."""
        return MessageType.DISCRETE