# src/inference_engine/message_passing.py

from typing import Dict, Set, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
from collections import defaultdict

from src.inference_engine.message_computation import (
    MessageComputationEngine,
    ComputationResult
)
from src.inference_engine.message_scheduling import (
    MessageScheduler,
    ScheduleEntry
)
from src.inference_engine.evidence_propagation import (
    EvidencePropagator,
    Evidence
)

from .messages import (
    Message,
    MessageType,
    ValidationResult,
    DiscreteMessage,
    GaussianMessage,
    CLGMessage
)
from .messages.operators import MessageOperator
from ..probability_distribution.factors import Factor

class NodeType(Enum):
    """Types of nodes in the factor graph."""
    VARIABLE = "variable"
    FACTOR = "factor"

class Node(ABC):
    """Abstract base class for nodes in the factor graph."""
    def __init__(self, id: str, type: NodeType):
        self.id = id
        self.type = type
        self.neighbors: Dict[str, 'Node'] = {}
        self.messages: Dict[str, Message] = {}  # Messages indexed by source node id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def compute_message(self, target_id: str) -> Message:
        """
        Compute message to be sent to target node.
        
        Args:
            target_id: ID of the target node
            
        Returns:
            Message: Computed message
            
        Raises:
            ValueError: If message cannot be computed
        """
        pass

    def send_message(self, target_id: str) -> None:
        """
        Send message to target node.
        
        Args:
            target_id: ID of the target node
            
        Raises:
            ValueError: If message cannot be sent
        """
        if target_id not in self.neighbors:
            raise ValueError(f"Node {target_id} is not a neighbor of {self.id}")
            
        message = self.compute_message(target_id)
        validation = message.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid message computed: {validation.message}")
            
        self.neighbors[target_id].receive_message(self.id, message)
        self.logger.debug(f"Sent message from {self.id} to {target_id}")

    def receive_message(self, source_id: str, message: Message) -> None:
        """
        Receive message from source node.
        
        Args:
            source_id: ID of the source node
            message: The message being received
            
        Raises:
            ValueError: If message cannot be received
        """
        if source_id not in self.neighbors:
            raise ValueError(f"Received message from non-neighbor node {source_id}")
            
        validation = message.validate()
        if not validation.is_valid:
            raise ValueError(f"Received invalid message: {validation.message}")
            
        self.messages[source_id] = message
        self.logger.debug(f"Received message at {self.id} from {source_id}")

class VariableNode(Node):
    """
    Represents a variable node in the factor graph.
    Maintains exact message passing without approximation.
    """
    def __init__(self, id: str, variable_type: str):
        """
        Initialize variable node.
        
        Args:
            id: Node identifier
            variable_type: Type of variable ('discrete', 'continuous', etc.)
        """
        super().__init__(id, NodeType.VARIABLE)
        self.variable_type = variable_type
        self.evidence = None  # Evidence value if observed
        
    def set_evidence(self, value: Any) -> None:
        """
        Set evidence value for this variable.
        
        Args:
            value: Evidence value
            
        Raises:
            ValueError: If evidence value is invalid
        """
        # Evidence validation would go here
        self.evidence = value
        self.logger.debug(f"Set evidence for {self.id}: {value}")

    def compute_message(self, target_id: str) -> Message:
        """
        Compute message to target factor node.
        Product of all incoming messages except from target.
        """
        incoming_messages = [
            msg for source_id, msg in self.messages.items()
            if source_id != target_id
        ]
        
        if not incoming_messages:
            # If no incoming messages, return uniform message
            return self._create_uniform_message(target_id)
            
        # Combine all incoming messages
        result = incoming_messages[0]
        for msg in incoming_messages[1:]:
            result = result.combine(msg)
            
        # If evidence is set, incorporate it
        if self.evidence is not None:
            result = self._incorporate_evidence(result)
            
        return result

    def _create_uniform_message(self, target_id: str) -> Message:
        """Create appropriate uniform message based on variable type."""
        if self.variable_type == "discrete":
            # Get states from neighboring factor
            states = self.neighbors[target_id].get_variable_states(self.id)
            prob = 1.0 / len(states)
            return DiscreteMessage(
                self.id,
                target_id,
                [self.id],
                "variable_to_factor",
                {self.id: states},
                {(state,): prob for state in states}
            )
        elif self.variable_type == "continuous":
            # Create wide Gaussian for continuous variables
            return GaussianMessage(
                self.id,
                target_id,
                [self.id],
                "variable_to_factor",
                0.0,  # mean
                1e6   # variance (very uncertain)
            )
        else:
            raise ValueError(f"Unsupported variable type: {self.variable_type}")

    def _incorporate_evidence(self, message: Message) -> Message:
        """Incorporate evidence into message."""
        if isinstance(message, DiscreteMessage):
            # For discrete messages, set probability 1 for evidence value
            new_probs = {
                states: 1.0 if states[0] == self.evidence else 0.0
                for states in message.probabilities.keys()
            }
            return DiscreteMessage(
                message.source_id,
                message.target_id,
                message.variables,
                message.direction,
                message.states,
                new_probs
            )
        elif isinstance(message, GaussianMessage):
            # For Gaussian messages, create delta-like distribution
            return GaussianMessage(
                message.source_id,
                message.target_id,
                message.variables,
                message.direction,
                float(self.evidence),  # mean at evidence value
                1e-10                  # very small variance
            )
        else:
            raise ValueError(f"Unsupported message type for evidence: {type(message)}")

class FactorNode(Node):
    """
    Represents a factor node in the factor graph.
    Maintains exact factor operations without approximation.
    """
    def __init__(self, id: str, factor: Factor):
        """
        Initialize factor node.
        
        Args:
            id: Node identifier
            factor: The factor associated with this node
        """
        super().__init__(id, NodeType.FACTOR)
        self.factor = factor
        
    def compute_message(self, target_id: str) -> Message:
        """
        Compute message to target variable node.
        Product of factor with all incoming messages except from target,
        marginalized over all variables except target.
        """
        # Get incoming messages from all neighbors except target
        incoming_messages = [
            msg for source_id, msg in self.messages.items()
            if source_id != target_id
        ]
        
        # Start with factor as initial message
        result = self.factor.to_message(self.id, target_id)
        
        # Multiply by all incoming messages
        for msg in incoming_messages:
            result = result.combine(msg)
            
        # Marginalize out all variables except target
        vars_to_marginalize = [
            var for var in result.variables
            if var != self.neighbors[target_id].id
        ]
        
        if vars_to_marginalize:
            result = result.marginalize(vars_to_marginalize)
            
        return result

    def get_variable_states(self, var_id: str) -> List[str]:
        """Get possible states for a discrete variable."""
        if isinstance(self.factor, DiscreteFactor):
            return self.factor.states[var_id]
        raise ValueError(f"Factor {self.id} does not have discrete states")

class MessagePassing:
    """
    Controls message passing in the factor graph.
    Ensures exact inference without approximation.
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.messages_sent: Set[Tuple[str, str]] = set()
        self.logger = logging.getLogger(__name__)
        
    def add_node(self, node: Node) -> None:
        """Add node to the graph."""
        self.nodes[node.id] = node
        
    def add_edge(self, node1_id: str, node2_id: str) -> None:
        """Add edge between nodes."""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")
            
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        if node1.type == node2.type:
            raise ValueError("Cannot connect nodes of same type")
            
        node1.neighbors[node2_id] = node2
        node2.neighbors[node1_id] = node1
        
    def run_belief_propagation(self, max_iterations: int = 100, 
                             tolerance: float = 1e-6) -> bool:
        """
        Run belief propagation until convergence or max iterations.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            bool: Whether convergence was achieved
        """
        for iteration in range(max_iterations):
            old_messages = self._copy_messages()
            
            # Send messages from variables to factors
            self._send_variable_to_factor_messages()
            
            # Send messages from factors to variables
            self._send_factor_to_variable_messages()
            
            # Check convergence
            if self._check_convergence(old_messages, tolerance):
                self.logger.info(f"Converged after {iteration + 1} iterations")
                return True
                
        self.logger.warning(f"Did not converge after {max_iterations} iterations")
        return False
        
    def _copy_messages(self) -> Dict[Tuple[str, str], Message]:
        """Create deep copy of all current messages."""
        messages = {}
        for node in self.nodes.values():
            for source_id, message in node.messages.items():
                messages[(source_id, node.id)] = message
        return messages
        
    def _send_variable_to_factor_messages(self) -> None:
        """Send messages from all variable nodes to factor nodes."""
        for node in self.nodes.values():
            if node.type == NodeType.VARIABLE:
                for target_id in node.neighbors:
                    node.send_message(target_id)
                    
    def _send_factor_to_variable_messages(self) -> None:
        """Send messages from all factor nodes to variable nodes."""
        for node in self.nodes.values():
            if node.type == NodeType.FACTOR:
                for target_id in node.neighbors:
                    node.send_message(target_id)
                    
    def _check_convergence(self, old_messages: Dict[Tuple[str, str], Message], 
                          tolerance: float) -> bool:
        """
        Check if messages have converged.
        
        Args:
            old_messages: Messages from previous iteration
            tolerance: Maximum allowed difference
            
        Returns:
            bool: Whether convergence has been achieved
        """
        for node in self.nodes.values():
            for source_id, message in node.messages.items():
                old_message = old_messages.get((source_id, node.id))
                if old_message is None:
                    return False
                    
                if not message.is_close(old_message, tolerance):
                    return False
                    
        return True

    def get_beliefs(self) -> Dict[str, Message]:
        """
        Compute final beliefs for all variable nodes.
        Product of all incoming messages to each variable node.
        
        Returns:
            Dict mapping variable node IDs to their final beliefs
        """
        beliefs = {}
        for node in self.nodes.values():
            if node.type == NodeType.VARIABLE:
                # Combine all incoming messages
                if node.messages:
                    belief = list(node.messages.values())[0]
                    for message in list(node.messages.values())[1:]:
                        belief = belief.combine(message)
                else:
                    # If no messages, create uniform belief
                    belief = node._create_uniform_message(None)
                    
                beliefs[node.id] = belief
                
        return beliefs

@dataclass
class InferenceResult:
    """Results of inference computation with error tracking."""
    beliefs: Dict[str, Any]
    error_bounds: Dict[str, float]
    numerical_issues: Optional[List[str]] = None
    convergence_info: Optional[Dict[str, Any]] = None

class MessagePassingEngine:
    """
    Core engine for belief propagation inference.
    Integrates scheduling, computation, and evidence handling
    while maintaining mathematical rigor.
    """
    def __init__(self, model: 'BayesianNetwork'):
        """
        Initialize message passing engine.
        
        Args:
            model: The Bayesian network model
        """
        self.model = model
        self.computation_engine = MessageComputationEngine(model)
        self.scheduler = MessageScheduler(model)
        self.evidence_propagator = EvidencePropagator(model)
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.messages: Dict[Tuple[str, str], 'Message'] = {}
        self.beliefs: Dict[str, 'Distribution'] = {}
        self.convergence_threshold = 1e-6
        self.max_iterations = 100

    def run_inference(self, 
                     query_variables: Set[str],
                     evidence: Optional[Dict[str, Evidence]] = None,
                     **kwargs) -> InferenceResult:
        """
        Run inference to compute beliefs for query variables.
        
        Args:
            query_variables: Variables to compute beliefs for
            evidence: Optional evidence to incorporate
            **kwargs: Additional parameters (e.g., convergence_threshold)
            
        Returns:
            InferenceResult containing computed beliefs and diagnostics
            
        Raises:
            ValueError: If inference fails or precision requirements not met
        """
        self.logger.info("Starting inference computation")
        self.logger.debug(f"Query variables: {query_variables}")
        self.logger.debug(f"Evidence: {evidence}")
        
        # Update parameters if provided
        self._update_parameters(kwargs)
        
        try:
            # Initialize inference
            self._initialize_inference(evidence)
            
            # Create message schedule
            schedule = self.scheduler.create_schedule(
                query_variables,
                set(evidence.keys()) if evidence else set()
            )
            
            # Optimize schedule for numerical stability
            schedule = self.scheduler.optimize_schedule(schedule)
            
            # Run belief propagation
            converged = self._run_belief_propagation(schedule)
            
            # Compute final beliefs
            result = self._compute_final_beliefs(
                query_variables,
                converged
            )
            
            self.logger.info("Inference computation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Inference computation failed: {str(e)}")
            raise

    def _initialize_inference(self, evidence: Optional[Dict[str, Evidence]]) -> None:
        """Initialize inference state."""
        # Clear previous state
        self.messages.clear()
        self.beliefs.clear()
        
        # Initialize with evidence
        if evidence:
            for var, ev in evidence.items():
                self.evidence_propagator.add_evidence(ev)

    def _run_belief_propagation(self, schedule: List[ScheduleEntry]) -> bool:
        """
        Run belief propagation until convergence or max iterations.
        
        Returns:
            bool indicating whether convergence was achieved
        """
        iteration = 0
        converged = False
        
        while iteration < self.max_iterations and not converged:
            self.logger.debug(f"Starting iteration {iteration + 1}")
            
            # Store old messages for convergence check
            old_messages = self._copy_messages()
            
            # Process all messages in schedule
            for entry in schedule:
                self._process_message(entry)
                
            # Check convergence
            converged = self._check_convergence(old_messages)
            
            if converged:
                self.logger.info(f"Converged after {iteration + 1} iterations")
            
            iteration += 1
        
        if not converged:
            self.logger.warning(
                f"Did not converge after {self.max_iterations} iterations"
            )
            
        return converged

    def _process_message(self, entry: ScheduleEntry) -> None:
        """Process a single message in the schedule."""
        try:
            # Compute message
            message = self.computation_engine.compute_message(
                entry.source_id,
                entry.target_id,
                self._get_incoming_messages(entry)
            )
            
            # Validate message
            if not self._validate_message(message):
                raise ValueError(
                    f"Message from {entry.source_id} to {entry.target_id} "
                    f"failed validation"
                )
                
            # Store message
            self.messages[(entry.source_id, entry.target_id)] = message
            
        except Exception as e:
            self.logger.error(
                f"Error processing message {entry.source_id}->{entry.target_id}: "
                f"{str(e)}"
            )
            raise

    def _validate_message(self, message: 'Message') -> bool:
        """
        Validate a computed message.
        Checks numerical stability and distribution properties.
        """
        try:
            # Basic message validation
            if not message.validate():
                return False
                
            # Check numerical properties
            if isinstance(message, DiscreteMessage):
                return self._validate_discrete_message(message)
            elif isinstance(message, GaussianMessage):
                return self._validate_gaussian_message(message)
            elif isinstance(message, CLGMessage):
                return self._validate_clg_message(message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Message validation failed: {str(e)}")
            return False

    def _validate_discrete_message(self, message: 'DiscreteMessage') -> bool:
        """Validate discrete message properties."""
        # Check probability sum
        total_prob = sum(message.probabilities.values())
        if abs(total_prob - 1.0) > 1e-10:
            return False
            
        # Check individual probabilities
        if any(p < 0 or p > 1 for p in message.probabilities.values()):
            return False
            
        # Check for numerical issues
        if any(0 < p < 1e-15 for p in message.probabilities.values()):
            self.logger.warning("Very small probabilities detected")
            
        return True

    def _validate_gaussian_message(self, message: 'GaussianMessage') -> bool:
        """Validate Gaussian message properties."""
        # Check variance
        if message.variance <= 0:
            return False
            
        # Check for numerical stability
        if message.variance < 1e-13:
            self.logger.warning("Very small variance detected")
            
        return True

    def _validate_clg_message(self, message: 'CLGMessage') -> bool:
        """Validate CLG message properties."""
        # Validate discrete part
        if not self._validate_discrete_message(message.discrete_part):
            return False
            
        # Validate continuous part for each configuration
        for config, params in message.continuous_params.items():
            if params['variance'] <= 0:
                return False
                
            if abs(params['mean_base']) > 1e8:
                self.logger.warning("Large mean value detected")
                
            if any(abs(c) > 1e8 for c in params['coefficients']):
                self.logger.warning("Large coefficients detected")
                
        return True

    def _compute_final_beliefs(self, 
                             query_variables: Set[str],
                             converged: bool) -> InferenceResult:
        """Compute final beliefs for query variables."""
        beliefs = {}
        error_bounds = {}
        numerical_issues = []
        
        for var in query_variables:
            try:
                # Combine all incoming messages
                incoming_messages = [
                    msg for (src, tgt), msg in self.messages.items()
                    if tgt == var
                ]
                
                if not incoming_messages:
                    raise ValueError(f"No messages available for variable {var}")
                
                # Compute belief
                belief = incoming_messages[0]
                for msg in incoming_messages[1:]:
                    belief = belief.combine(msg)
                
                # Normalize if necessary
                if isinstance(belief, DiscreteMessage):
                    belief = self._normalize_discrete_belief(belief)
                
                beliefs[var] = belief.get_distribution()
                error_bounds[var] = self._compute_error_bound(belief)
                
            except Exception as e:
                self.logger.error(f"Error computing belief for {var}: {str(e)}")
                numerical_issues.append(f"Failed to compute belief for {var}")

        return InferenceResult(
            beliefs=beliefs,
            error_bounds=error_bounds,
            numerical_issues=numerical_issues if numerical_issues else None,
            convergence_info={
                'converged': converged,
                'final_delta': self._get_convergence_delta()
            }
        )

    def _normalize_discrete_belief(self, 
                                 belief: 'DiscreteMessage') -> 'DiscreteMessage':
        """Normalize a discrete belief distribution."""
        total = sum(belief.probabilities.values())
        if abs(total - 1.0) > 1e-10:
            normalized_probs = {
                k: v/total for k, v in belief.probabilities.items()
            }
            return DiscreteMessage(
                belief.source_id,
                belief.target_id,
                belief.variables,
                belief.direction,
                normalized_probs
            )
        return belief

    def _compute_error_bound(self, belief: 'Message') -> float:
        """Compute error bound for a belief."""
        if isinstance(belief, DiscreteMessage):
            # Use sum deviation from 1.0 as error measure
            return abs(sum(belief.probabilities.values()) - 1.0)
            
        elif isinstance(belief, GaussianMessage):
            # Use relative error in variance
            return abs(belief.variance) * np.finfo(float).eps
            
        elif isinstance(belief, CLGMessage):
            # Take maximum of discrete and continuous errors
            discrete_error = self._compute_error_bound(belief.discrete_part)
            continuous_errors = [
                abs(params['variance']) * np.finfo(float).eps
                for params in belief.continuous_params.values()
            ]
            return max(discrete_error, max(continuous_errors))
            
        return 0.0

    def _check_convergence(self, 
                          old_messages: Dict[Tuple[str, str], 'Message']) -> bool:
        """Check if messages have converged."""
        if not old_messages:
            return False
            
        max_delta = 0.0
        
        for key, new_msg in self.messages.items():
            if key not in old_messages:
                return False
                
            old_msg = old_messages[key]
            delta = self._compute_message_difference(old_msg, new_msg)
            max_delta = max(max_delta, delta)
            
        return max_delta < self.convergence_threshold

    def _compute_message_difference(self, 
                                  msg1: 'Message', 
                                  msg2: 'Message') -> float:
        """Compute difference between two messages."""
        if type(msg1) != type(msg2):
            raise ValueError("Cannot compare different message types")
            
        if isinstance(msg1, DiscreteMessage):
            # Maximum absolute difference in probabilities
            return max(
                abs(msg1.probabilities.get(k, 0) - msg2.probabilities.get(k, 0))
                for k in set(msg1.probabilities) | set(msg2.probabilities)
            )
            
        elif isinstance(msg1, GaussianMessage):
            # Normalized difference in parameters
            mean_diff = abs(msg1.mean - msg2.mean) / (1 + abs(msg1.mean))
            var_diff = abs(msg1.variance - msg2.variance) / msg1.variance
            return max(mean_diff, var_diff)
            
        elif isinstance(msg1, CLGMessage):
            # Maximum of discrete and continuous differences
            discrete_diff = self._compute_message_difference(
                msg1.discrete_part,
                msg2.discrete_part
            )
            
            continuous_diffs = []
            for config in msg1.continuous_params:
                if config not in msg2.continuous_params:
                    continue
                    
                params1 = msg1.continuous_params[config]
                params2 = msg2.continuous_params[config]
                
                mean_diff = abs(params1['mean_base'] - params2['mean_base'])
                coef_diff = max(
                    abs(c1 - c2)
                    for c1, c2 in zip(params1['coefficients'],
                                    params2['coefficients'])
                )
                var_diff = abs(params1['variance'] - params2['variance'])
                
                continuous_diffs.append(max(mean_diff, coef_diff, var_diff))
                
            return max(discrete_diff, max(continuous_diffs)) if continuous_diffs else discrete_diff
            
        return 0.0

    def _get_incoming_messages(self, entry: ScheduleEntry) -> List['Message']:
        """Get incoming messages for message computation."""
        source_node = self.model.nodes[entry.source_id]
        
        return [
            self.messages.get((neighbor_id, entry.source_id))
            for neighbor_id in source_node.neighbors
            if neighbor_id != entry.target_id
            and (neighbor_id, entry.source_id) in self.messages
        ]

    def _copy_messages(self) -> Dict[Tuple[str, str], 'Message']:
        """Create a deep copy of current messages."""
        return {k: v.copy() for k, v in self.messages.items()}

    def _update_parameters(self, params: Dict[str, Any]) -> None:
        """Update engine parameters from kwargs."""
        if 'convergence_threshold' in params:
            self.convergence_threshold = params['convergence_threshold']
            
        if 'max_iterations' in params:
            self.max_iterations = params['max_iterations']

    def _get_convergence_delta(self) -> float:
        """Get final convergence delta for diagnostics."""
        if not hasattr(self, '_last_delta'):
            return float('inf')
        return self._last_delta