# src/inference_engine/message_passing.py

from typing import Dict, Set, List, Optional, Tuple, Any
from dataclasses import dataclass
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