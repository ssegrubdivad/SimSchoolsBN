# src/inference_engine/integration.py

from typing import Dict, Set, List, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from .message_passing import MessagePassingEngine
from ..query_interface.query_processor import QueryProcessor
from .messages.base import ValidationResult

from src.network_structure.bayesian_network import BayesianNetwork
from src.query_interface.query_processor import QueryProcessor
from src.inference_engine.message_passing import MessagePassingEngine
# from src.validation import ValidationResult

@dataclass
class InferenceRequest:
    """Represents a request for inference computation."""
    query_variables: Set[str]
    evidence: Optional[Dict[str, Any]] = None
    inference_type: str = "exact"  # 'exact' or 'approximate'
    options: Optional[Dict[str, Any]] = None

@dataclass
class InferenceResponse:
    """Results of inference computation with metadata."""
    beliefs: Dict[str, Any]
    error_bounds: Dict[str, float]
    computation_time: float
    timestamp: datetime
    query_info: InferenceRequest
    numerical_issues: Optional[List[str]] = None
    validation_results: Optional[Dict[str, ValidationResult]] = None

class InferenceIntegrator:
    """
    Integrates new inference engine with existing codebase.
    Maintains exact computation requirements while providing
    seamless interface with existing components.
    """
    def __init__(self, model: BayesianNetwork):
        """
        Initialize integrator with model.
        
        Args:
            model: The Bayesian network model
        """
        self.model = model
        self.message_engine = MessagePassingEngine(model)
        self.query_processor = QueryProcessor(model)
        self.logger = logging.getLogger(__name__)
        
        # Validation tracking
        self.validation_results: Dict[str, ValidationResult] = {}
        
    def process_query(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process an inference request.
        
        Args:
            request: The inference request
            
        Returns:
            InferenceResponse containing results and metadata
            
        Raises:
            ValueError: If query is invalid or computation fails
        """
        start_time = datetime.now()
        self.logger.info(f"Processing inference request for variables: {request.query_variables}")
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Prepare evidence
            evidence = self._prepare_evidence(request.evidence)
            
            # Run inference
            if request.inference_type == "exact":
                result = self._run_exact_inference(request.query_variables, evidence)
            else:
                raise ValueError(f"Unsupported inference type: {request.inference_type}")
            
            # Create response
            computation_time = (datetime.now() - start_time).total_seconds()
            
            response = InferenceResponse(
                beliefs=result.beliefs,
                error_bounds=result.error_bounds,
                computation_time=computation_time,
                timestamp=datetime.now(),
                query_info=request,
                numerical_issues=result.numerical_issues,
                validation_results=self.validation_results
            )
            
            self.logger.info("Inference computation completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing inference request: {str(e)}")
            raise

    def _validate_request(self, request: InferenceRequest) -> None:
        """Validate inference request."""
        # Validate query variables
        invalid_vars = set(request.query_variables) - set(self.model.nodes.keys())
        if invalid_vars:
            raise ValueError(f"Invalid query variables: {invalid_vars}")
            
        # Validate evidence
        if request.evidence:
            self._validate_evidence(request.evidence)
            
        # Validate options
        if request.options:
            self._validate_options(request.options)

    def _validate_evidence(self, evidence: Dict[str, Any]) -> None:
        """Validate evidence values."""
        for var, value in evidence.items():
            if var not in self.model.nodes:
                raise ValueError(f"Evidence variable not in model: {var}")
                
            node = self.model.nodes[var]
            
            # Type-specific validation
            if isinstance(node, DiscreteNode):
                if value not in node.states:
                    raise ValueError(
                        f"Invalid evidence value {value} for variable {var}. "
                        f"Must be one of {node.states}"
                    )
            elif isinstance(node, ContinuousNode):
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Evidence value for continuous variable {var} "
                        f"must be numeric"
                    )
            elif isinstance(node, CLGNode):
                # Handle CLG nodes based on evidence type
                self._validate_clg_evidence(node, value)

    def _validate_clg_evidence(self, node: 'CLGNode', value: Any) -> None:
        """Validate evidence for CLG nodes."""
        if isinstance(value, (int, float)):
            # Continuous evidence for CLG node
            if not self._is_valid_continuous_value(value):
                raise ValueError(
                    f"Invalid continuous evidence value for CLG node: {value}"
                )
        elif isinstance(value, str):
            # Discrete evidence for CLG node
            if value not in node.discrete_states:
                raise ValueError(
                    f"Invalid discrete evidence value {value} for CLG node. "
                    f"Must be one of {node.discrete_states}"
                )
        else:
            raise ValueError(
                f"Invalid evidence type for CLG node: {type(value)}"
            )

    def _validate_options(self, options: Dict[str, Any]) -> None:
        """Validate inference options."""
        valid_options = {
            'convergence_threshold',
            'max_iterations',
            'numerical_precision'
        }
        
        invalid_options = set(options.keys()) - valid_options
        if invalid_options:
            raise ValueError(f"Invalid options: {invalid_options}")
            
        # Validate option values
        if 'convergence_threshold' in options:
            threshold = options['convergence_threshold']
            if not (isinstance(threshold, float) and 0 < threshold < 1):
                raise ValueError(
                    "convergence_threshold must be float between 0 and 1"
                )
                
        if 'max_iterations' in options:
            max_iter = options['max_iterations']
            if not (isinstance(max_iter, int) and max_iter > 0):
                raise ValueError("max_iterations must be positive integer")
                
        if 'numerical_precision' in options:
            precision = options['numerical_precision']
            if not (isinstance(precision, float) and precision > 0):
                raise ValueError("numerical_precision must be positive float")

    def _prepare_evidence(self, 
                         evidence: Optional[Dict[str, Any]]) -> Optional[Dict[str, 'Evidence']]:
        """Convert raw evidence to Evidence objects."""
        if not evidence:
            return None
            
        prepared_evidence = {}
        for var, value in evidence.items():
            node = self.model.nodes[var]
            
            if isinstance(node, DiscreteNode):
                prepared_evidence[var] = Evidence(
                    variable=var,
                    value=value,
                    evidence_type=EvidenceType.HARD
                )
            elif isinstance(node, ContinuousNode):
                prepared_evidence[var] = Evidence(
                    variable=var,
                    value=float(value),
                    evidence_type=EvidenceType.HARD,
                    precision=self._get_precision_requirement(node)
                )
            elif isinstance(node, CLGNode):
                prepared_evidence[var] = self._prepare_clg_evidence(node, value)
                
        return prepared_evidence

    def _prepare_clg_evidence(self, 
                            node: 'CLGNode', 
                            value: Any) -> 'Evidence':
        """Prepare evidence for CLG nodes."""
        if isinstance(value, (int, float)):
            return Evidence(
                variable=node.id,
                value=float(value),
                evidence_type=EvidenceType.HARD,
                precision=self._get_precision_requirement(node)
            )
        else:
            return Evidence(
                variable=node.id,
                value=value,
                evidence_type=EvidenceType.HARD
            )

    def _get_precision_requirement(self, node: 'Node') -> float:
        """Get precision requirement for node type."""
        if isinstance(node, DiscreteNode):
            return 1e-10  # For probability sums
        elif isinstance(node, ContinuousNode):
            return 1e-13  # For continuous variables
        elif isinstance(node, CLGNode):
            return 1e-12  # For mixed variables
        return 1e-10  # Default

    def _run_exact_inference(self,
                           query_variables: Set[str],
                           evidence: Optional[Dict[str, 'Evidence']]) -> InferenceResult:
        """Run exact inference using message passing engine."""
        return self.message_engine.run_inference(
            query_variables=query_variables,
            evidence=evidence
        )

    def validate_model(self) -> ValidationResult:
        """Validate the model for inference compatibility."""
        try:
            # Check model structure
            if not self.model.check_model():
                return ValidationResult(
                    False,
                    "Invalid model structure"
                )
                
            # Check variable types
            if not self._validate_variable_types():
                return ValidationResult(
                    False,
                    "Invalid variable type configuration"
                )
                
            # Check distribution specifications
            if not self._validate_distributions():
                return ValidationResult(
                    False,
                    "Invalid distribution specifications"
                )
                
            return ValidationResult(True, "Model validation successful")
            
        except Exception as e:
            return ValidationResult(
                False,
                f"Model validation failed: {str(e)}"
            )

    def _validate_variable_types(self) -> bool:
        """Validate variable type configurations."""
        for node in self.model.nodes.values():
            # Check variable type
            if not hasattr(node, 'variable_type'):
                return False
                
            # Check type-specific properties
            if isinstance(node, DiscreteNode):
                if not node.states:
                    return False
            elif isinstance(node, ContinuousNode):
                if not hasattr(node, 'distribution'):
                    return False
            elif isinstance(node, CLGNode):
                if not (hasattr(node, 'discrete_states') and 
                       hasattr(node, 'continuous_parents')):
                    return False
                    
        return True

    def _validate_distributions(self) -> bool:
        """Validate distribution specifications."""
        for node in self.model.nodes.values():
            if not hasattr(node, 'distribution'):
                return False
                
            distribution = node.distribution
            if distribution is None:
                return False
                
            # Validate distribution parameters
            try:
                validation = distribution.validate()
                self.validation_results[node.id] = validation
                if not validation.is_valid:
                    return False
            except Exception:
                return False
                
        return True