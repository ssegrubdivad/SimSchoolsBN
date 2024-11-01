# src/inference_engine/messages/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
import logging
from scipy.stats import truncnorm
from enum import Enum

class MessageType(Enum):
    """Enumeration of possible message types."""
    DISCRETE = "discrete"
    GAUSSIAN = "gaussian"
    TRUNCATED_GAUSSIAN = "truncated_gaussian"
    CLG = "clg"
    MULTIVARIATE_GAUSSIAN = "multivariate_gaussian"

@dataclass
class ValidationResult:
    """Results of message validation."""
    is_valid: bool
    message: str
    details: Optional[Dict] = None

class Message(ABC):
    """
    Abstract base class for factor graph messages.
    All messages must maintain exact representations without approximation.
    """
    def __init__(self, 
                 source_id: str, 
                 target_id: str, 
                 variables: List[str],
                 direction: str):
        """
        Initialize a message.
        
        Args:
            source_id: Identifier of the sending node
            target_id: Identifier of the receiving node
            variables: List of variables involved in the message
            direction: Either 'factor_to_variable' or 'variable_to_factor'
        """
        self.source_id = source_id
        self.target_id = target_id
        self.variables = variables
        self.direction = direction
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._validated = False

    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        Validate the message content.
        Must be called before any operations can be performed.
        """
        if not self.variables:
            return ValidationResult(False, "Message must involve at least one variable")
        
        if self.direction not in {'factor_to_variable', 'variable_to_factor'}:
            return ValidationResult(False, f"Invalid message direction: {self.direction}")
        
        return ValidationResult(True, "Basic validation passed")

    @abstractmethod
    def combine(self, other: 'Message') -> 'Message':
        """
        Combine this message with another message.
        Must maintain exact representations.
        
        Args:
            other: Another message to combine with
            
        Returns:
            A new message representing the combination
            
        Raises:
            ValueError: If messages are incompatible or unvalidated
        """
        if not (self._validated and other._validated):
            raise ValueError("Both messages must be validated before combination")
        
        if set(self.variables) != set(other.variables):
            raise ValueError("Messages must involve the same variables for combination")

    @abstractmethod
    def marginalize(self, variables: List[str]) -> 'Message':
        """
        Marginalize out specified variables from the message.
        Must maintain exact representations.
        
        Args:
            variables: List of variables to marginalize out
            
        Returns:
            A new message with specified variables marginalized out
            
        Raises:
            ValueError: If marginalization is invalid or message unvalidated
        """
        if not self._validated:
            raise ValueError("Message must be validated before marginalization")
        
        if not set(variables).issubset(set(self.variables)):
            raise ValueError(f"Cannot marginalize variables that are not in the message: "
                           f"{set(variables) - set(self.variables)}")

    @abstractmethod
    def get_type(self) -> MessageType:
        """Get the type of this message."""
        pass

class ConstantMessage(Message):
    """Message representing a constant value, useful for marginalization results."""
    def __init__(self, 
                 source_id: str, 
                 target_id: str,
                 variables: List[str],
                 direction: str,
                 value: float = 1.0):
        super().__init__(source_id, target_id, variables, direction)
        self.value = value
        self._validated = True

    def validate(self) -> ValidationResult:
        """Constant messages are always valid."""
        return ValidationResult(True, "Constant message is always valid")

    def combine(self, other: 'Message') -> 'Message':
        """Combine with another message."""
        super().combine(other)
        
        if isinstance(other, ConstantMessage):
            return ConstantMessage(
                f"{self.source_id}_{other.source_id}",
                self.target_id,
                [],
                self.direction,
                self.value * other.value
            )
        return other  # Constant message acts as multiplicative identity

    def marginalize(self, variables: List[str]) -> 'Message':
        """Marginalize out specified variables."""
        return self  # Marginalization has no effect on constant message

    def get_type(self) -> MessageType:
        """Get message type."""
        return None  # Constant messages don't have a specific type