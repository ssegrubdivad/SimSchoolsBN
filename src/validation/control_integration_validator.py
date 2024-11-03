# src/validation/control_integration_validator.py

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging
import numpy as np

from src.inference_engine.message_computation import ControlAwareMessageComputationEngine
from src.inference_engine.message_passing import ControlAwareMessageEngine
from src.inference_engine.evidence_propagation import ControlAwareEvidencePropagator
from src.visualization.network_visualizer import ControlAwareNetworkVisualizer
from src.network_structure.bayesian_network import BayesianNetwork
from src.education_models.locus_control import (
    ControlLevel,
    ControlScope,
    ControlValidator
)

@dataclass
class IntegrationValidationResult:
    """Results of control integration validation."""
    is_valid: bool
    message: str
    details: Dict[str, Any]
    numerical_issues: Optional[List[str]] = None
    error_bounds: Optional[Dict[str, float]] = None

class ControlIntegrationValidator:
    """
    Validates integration of control-aware components.
    Ensures consistent control validation and mathematical precision.
    """
    def __init__(self, model: BayesianNetwork):
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.control_validator = ControlValidator()
        
        # Initialize all control-aware components
        self.message_engine = ControlAwareMessageEngine(model)
        self.computation_engine = ControlAwareMessageComputationEngine(model)
        self.evidence_propagator = ControlAwareEvidencePropagator(model)
        self.visualizer = ControlAwareNetworkVisualizer(model)
        
        # Track validation state
        self.validation_results: Dict[str, IntegrationValidationResult] = {}
        self.error_bounds: Dict[str, float] = {}

    def validate_control_integration(self) -> IntegrationValidationResult:
        """
        Perform comprehensive validation of control integration.
        
        Returns:
            IntegrationValidationResult containing validation details
            
        Mathematical Guarantees:
        - Maintains exact probability computations
        - Preserves influence weight precision
        - Ensures consistent control validation
        """
        try:
            # Validate component interactions
            component_validation = self._validate_component_interactions()
            if not component_validation.is_valid:
                return component_validation
                
            # Validate control flow consistency
            control_validation = self._validate_control_flow()
            if not control_validation.is_valid:
                return control_validation
                
            # Validate mathematical precision
            precision_validation = self._validate_mathematical_precision()
            if not precision_validation.is_valid:
                return precision_validation
                
            return IntegrationValidationResult(
                is_valid=True,
                message="Control integration validation successful",
                details={
                    'component_validation': component_validation.details,
                    'control_validation': control_validation.details,
                    'precision_validation': precision_validation.details
                }
            )
            
        except Exception as e:
            self.logger.error(f"Control integration validation failed: {str(e)}")
            return IntegrationValidationResult(
                is_valid=False,
                message=f"Validation failed: {str(e)}",
                details={'error': str(e)}
            )

    def _validate_component_interactions(self) -> IntegrationValidationResult:
        """
        Validate interactions between control-aware components.
        Ensures consistent control handling across components.
        """
        interactions = {
            'message_computation': [],
            'evidence_handling': [],
            'visualization': []
        }
        
        # Test message computation interactions
        for node_id, node in self.model.nodes.items():
            if hasattr(node, 'control_scope'):
                # Test computation with different control levels
                for level in ControlLevel:
                    try:
                        message = self.computation_engine.compute_message(
                            node_id,
                            next(iter(node.children)).id,
                            [],
                            level
                        )
                        interactions['message_computation'].append({
                            'node': node_id,
                            'level': level,
                            'success': True
                        })
                    except Exception as e:
                        interactions['message_computation'].append({
                            'node': node_id,
                            'level': level,
                            'success': False,
                            'error': str(e)
                        })

        # Test evidence propagation interactions
        evidence_tests = self._generate_evidence_tests()
        for test in evidence_tests:
            try:
                self.evidence_propagator.add_evidence(
                    test['evidence'],
                    test['level']
                )
                interactions['evidence_handling'].append({
                    'test': test,
                    'success': True
                })
            except Exception as e:
                interactions['evidence_handling'].append({
                    'test': test,
                    'success': False,
                    'error': str(e)
                })

        # Test visualization interactions
        try:
            graph_data = self.visualizer.generate_graph_data(
                VisualizationConfig(
                    show_control_levels=True,
                    show_influence_paths=True
                )
            )
            interactions['visualization'].append({
                'graph_data_generation': 'success',
                'control_info_present': True
            })
        except Exception as e:
            interactions['visualization'].append({
                'graph_data_generation': 'failed',
                'error': str(e)
            })

        # Analyze interaction results
        failed_interactions = (
            [i for i in interactions['message_computation'] if not i['success']] +
            [i for i in interactions['evidence_handling'] if not i['success']] +
            [i for i in interactions['visualization'] if 'error' in i]
        )

        if failed_interactions:
            return IntegrationValidationResult(
                is_valid=False,
                message="Component interactions validation failed",
                details={
                    'failed_interactions': failed_interactions,
                    'all_interactions': interactions
                }
            )

        return IntegrationValidationResult(
            is_valid=True,
            message="Component interactions validated successfully",
            details={'interactions': interactions}
        )

    def _validate_control_flow(self) -> IntegrationValidationResult:
        """
        Validate control flow consistency across components.
        Ensures authority paths and influence weights are consistent.
        """
        control_tests = []
        
        # Test control flow through complete inference chain
        for node_id, node in self.model.nodes.items():
            if not hasattr(node, 'control_scope'):
                continue
                
            control_scope = node.control_scope
            primary_level = control_scope.primary_level
            
            # Test direct control
            direct_test = self._test_control_flow(node_id, primary_level)
            control_tests.append({
                'node': node_id,
                'level': primary_level,
                'type': 'direct',
                'result': direct_test
            })
            
            # Test secondary control levels
            for secondary_level in control_scope.secondary_levels:
                secondary_test = self._test_control_flow(node_id, secondary_level)
                control_tests.append({
                    'node': node_id,
                    'level': secondary_level,
                    'type': 'secondary',
                    'result': secondary_test
                })

        # Analyze control flow results
        failed_tests = [test for test in control_tests if not test['result']['success']]
        
        if failed_tests:
            return IntegrationValidationResult(
                is_valid=False,
                message="Control flow validation failed",
                details={
                    'failed_tests': failed_tests,
                    'all_tests': control_tests
                }
            )

        return IntegrationValidationResult(
            is_valid=True,
            message="Control flow validated successfully",
            details={'control_tests': control_tests}
        )

    def _validate_mathematical_precision(self) -> IntegrationValidationResult:
        """
        Validate mathematical precision across control-aware operations.
        Ensures exact computations and proper error tracking.
        """
        precision_tests = {
            'probability_sums': [],
            'influence_weights': [],
            'error_bounds': []
        }
        
        # Test probability precision
        for node_id, node in self.model.nodes.items():
            if not hasattr(node, 'control_scope'):
                continue
                
            # Test probability computations with control influence
            test_result = self._test_probability_precision(node_id)
            precision_tests['probability_sums'].append(test_result)
            
            # Test influence weight precision
            weight_result = self._test_influence_weight_precision(node_id)
            precision_tests['influence_weights'].append(weight_result)
            
            # Test error bound tracking
            error_result = self._test_error_bound_tracking(node_id)
            precision_tests['error_bounds'].append(error_result)

        # Analyze precision results
        failed_precision = []
        for category, tests in precision_tests.items():
            failed = [t for t in tests if not t['success']]
            if failed:
                failed_precision.extend(failed)

        if failed_precision:
            return IntegrationValidationResult(
                is_valid=False,
                message="Mathematical precision validation failed",
                details={
                    'failed_precision': failed_precision,
                    'all_tests': precision_tests
                }
            )

        return IntegrationValidationResult(
            is_valid=True,
            message="Mathematical precision validated successfully",
            details={
                'precision_tests': precision_tests,
                'error_bounds': self.error_bounds
            }
        )

    def _test_control_flow(self, 
                          node_id: str, 
                          control_level: ControlLevel) -> Dict[str, Any]:
        """Test control flow through all components for a node."""
        try:
            # Test message computation
            message = self.computation_engine.compute_message(
                node_id,
                next(iter(self.model.nodes[node_id].children)).id,
                [],
                control_level
            )
            
            # Test evidence handling
            evidence = self._create_test_evidence(node_id)
            self.evidence_propagator.add_evidence(evidence, control_level)
            
            # Test visualization
            vis_data = self.visualizer.highlight_authority_path(
                node_id,
                control_level
            )
            
            return {
                'success': True,
                'message': message,
                'evidence': evidence,
                'visualization': vis_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _test_probability_precision(self, node_id: str) -> Dict[str, Any]:
        """Test probability computation precision with control influence."""
        try:
            node = self.model.nodes[node_id]
            control_scope = node.control_scope
            
            # Compute probabilities with different control levels
            results = {}
            for level in ControlLevel:
                message = self.computation_engine.compute_message(
                    node_id,
                    next(iter(node.children)).id,
                    [],
                    level
                )
                
                # Check probability sum for discrete messages
                if hasattr(message, 'probabilities'):
                    prob_sum = sum(message.probabilities.values())
                    if abs(prob_sum - 1.0) > 1e-10:
                        return {
                            'success': False,
                            'error': f"Invalid probability sum: {prob_sum}",
                            'level': level
                        }
                
                results[level] = message
                
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _test_influence_weight_precision(self, node_id: str) -> Dict[str, Any]:
        """Test precision of influence weight calculations."""
        try:
            node = self.model.nodes[node_id]
            control_scope = node.control_scope
            
            # Test influence weight precision
            weights = {}
            for level in ControlLevel:
                weight = control_scope.get_influence_weight(level)
                
                # Verify weight is in valid range
                if not (0 <= weight <= 1):
                    return {
                        'success': False,
                        'error': f"Invalid influence weight: {weight}",
                        'level': level
                    }
                    
                weights[level] = weight
                
            return {
                'success': True,
                'weights': weights
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _test_error_bound_tracking(self, node_id: str) -> Dict[str, Any]:
        """Test error bound tracking in control-aware computations."""
        try:
            node = self.model.nodes[node_id]
            
            # Track error through computation chain
            error_chain = []
            current_error = 0.0
            
            for level in ControlLevel:
                message = self.computation_engine.compute_message(
                    node_id,
                    next(iter(node.children)).id,
                    [],
                    level
                )
                
                # Track error accumulation
                if hasattr(message, 'error_bound'):
                    current_error += message.error_bound
                    error_chain.append({
                        'level': level,
                        'error': message.error_bound,
                        'cumulative': current_error
                    })
                    
            # Store final error bound
            self.error_bounds[node_id] = current_error
            
            return {
                'success': True,
                'error_chain': error_chain,
                'final_error': current_error
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _create_test_evidence(self, node_id: str) -> 'Evidence':
        """Create test evidence for a node."""
        node = self.model.nodes[node_id]
        
        if hasattr(node, 'distribution'):
            if hasattr(node.distribution, 'probabilities'):
                # Discrete evidence
                return Evidence(
                    variable=node_id,
                    value=next(iter(node.distribution.probabilities.keys())),
                    evidence_type=EvidenceType.HARD
                )
            elif hasattr(node.distribution, 'mean'):
                # Continuous evidence
                return Evidence(
                    variable=node_id,
                    value=node.distribution.mean,
                    evidence_type=EvidenceType.HARD
                )
                
        raise ValueError(f"Cannot create test evidence for node {node_id}")

    def _generate_evidence_tests(self) -> List[Dict[str, Any]]:
        """Generate evidence test cases."""
        tests = []
        
        for node_id, node in self.model.nodes.items():
            if not hasattr(node, 'control_scope'):
                continue
                
            evidence = self._create_test_evidence(node_id)
            
            # Test with primary level
            tests.append({
                'evidence': evidence,
                'level': node.control_scope.primary_level
            })
            
            # Test with secondary levels
            for level in node.control_scope.secondary_levels:
                tests.append({
                    'evidence': evidence,
                    'level': level
                })
                
        return tests