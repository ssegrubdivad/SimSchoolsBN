# tests/test_inference_engine/test_integration.py

import pytest
import numpy as np
from typing import Dict, List, Any
from src.inference_engine.message_passing import MessagePassingEngine
from src.inference_engine.messages import MessageType
from src.network_structure.bayesian_network import BayesianNetwork
from ..test_framework import PrecisionTestCase

class IntegrationTestCases:
    """Test cases for end-to-end integration."""

    @pytest.fixture
    def integration_cases(self) -> List[PrecisionTestCase]:
        """Test cases for complete inference process."""
        return [
            # Mixed network inference
            PrecisionTestCase(
                name="mixed_network_inference",
                input_data={
                    'network': create_mixed_test_network(),
                    'query_variables': {'X3'},
                    'evidence': {
                        'X1': {'type': 'discrete', 'value': 'T'},
                        'X2': {'type': 'continuous', 'value': 1.0}
                    }
                },
                expected_result={
                    'X3': {
                        'discrete': {'high': 0.7, 'low': 0.3},
                        'continuous': {
                            'high': {'mean': 2.0, 'variance': 1.0},
                            'low': {'mean': 0.5, 'variance': 2.0}
                        }
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'end_to_end_precision': True,
                    'mixed_type_handling': True
                },
                preconditions=["Valid mixed network"],
                postconditions=["Valid inference results"]
            ),

            # Complex evidence handling
            PrecisionTestCase(
                name="complex_evidence_inference",
                input_data={
                    'network': create_complex_test_network(),
                    'query_variables': {'X4', 'X5'},
                    'evidence': {
                        'X1': {'type': 'discrete', 'value': 'T'},
                        'X2': {'type': 'continuous', 'value': 1.0},
                        'X3': {'type': 'clg', 'value': {'state': 'high', 'value': 2.0}}
                    }
                },
                expected_result={
                    'X4': {
                        'mean': 2.5,
                        'variance': 1.5
                    },
                    'X5': {
                        'discrete': {'A': 0.6, 'B': 0.4}
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'evidence_handling': True,
                    'precision_maintenance': True
                },
                preconditions=["Multiple evidence types"],
                postconditions=["Correct evidence incorporation"]
            ),

            # Large network test
            PrecisionTestCase(
                name="large_network_inference",
                input_data={
                    'network': create_large_test_network(),
                    'query_variables': {'X10', 'X20'},
                    'evidence': {
                        'X1': {'type': 'discrete', 'value': 'A'},
                        'X5': {'type': 'continuous', 'value': 0.5},
                        'X15': {'type': 'discrete', 'value': 'B'}
                    }
                },
                expected_result={
                    'X10': {
                        'mean': 1.5,
                        'variance': 0.5
                    },
                    'X20': {
                        'discrete': {'A': 0.3, 'B': 0.4, 'C': 0.3}
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'scalability': True,
                    'precision_maintenance': True,
                    'performance': True
                },
                preconditions=["Large network structure"],
                postconditions=["Accurate results on large scale"]
            ),

            # Temporal inference test
            PrecisionTestCase(
                name="temporal_inference",
                input_data={
                    'network': create_temporal_test_network(),
                    'query_variables': {'X_t1', 'X_t2'},
                    'evidence': {
                        'X_t0': {'type': 'continuous', 'value': 0.0}
                    },
                    'time_steps': 2
                },
                expected_result={
                    'X_t1': {'mean': 0.5, 'variance': 1.0},
                    'X_t2': {'mean': 1.0, 'variance': 1.5}
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'temporal_consistency': True,
                    'precision_over_time': True
                },
                preconditions=["Valid temporal network"],
                postconditions=["Correct temporal propagation"]
            )
        ]

    # Helper functions for creating test networks
    def create_mixed_test_network(self) -> BayesianNetwork:
        """Create a test network with mixed variable types."""
        network = BayesianNetwork("test_mixed")
        # Add nodes and edges for mixed network
        # [Implementation details]
        return network

    def create_complex_test_network(self) -> BayesianNetwork:
        """Create a test network for complex evidence cases."""
        network = BayesianNetwork("test_complex")
        # Add nodes and edges for complex network
        # [Implementation details]
        return network

    def create_large_test_network(self) -> BayesianNetwork:
        """Create a large test network."""
        network = BayesianNetwork("test_large")
        # Add nodes and edges for large network
        # [Implementation details]
        return network

    def create_temporal_test_network(self) -> BayesianNetwork:
        """Create a temporal test network."""
        network = BayesianNetwork("test_temporal")
        # Add nodes and edges for temporal network
        # [Implementation details]
        return network

    def test_mixed_network_inference(self, integration_cases):
        """Test inference on mixed networks."""
        case = integration_cases[0]  # mixed_network_inference case
        engine = MessagePassingEngine(case.input_data['network'])
        result = engine.run_inference(
            query_variables=case.input_data['query_variables'],
            evidence=case.input_data['evidence']
        )
        assert self._compare_results(result, case.expected_result, case.error_bound)

    def test_complex_evidence_handling(self, integration_cases):
        """Test handling of complex evidence."""
        case = integration_cases[1]  # complex_evidence_inference case
        engine = MessagePassingEngine(case.input_data['network'])
        result = engine.run_inference(
            query_variables=case.input_data['query_variables'],
            evidence=case.input_data['evidence']
        )
        assert self._compare_results(result, case.expected_result, case.error_bound)

    def test_large_network_inference(self, integration_cases):
        """Test inference on large networks."""
        case = integration_cases[2]  # large_network_inference case
        engine = MessagePassingEngine(case.input_data['network'])
        result = engine.run_inference(
            query_variables=case.input_data['query_variables'],
            evidence=case.input_data['evidence']
        )
        assert self._compare_results(result, case.expected_result, case.error_bound)

    def test_temporal_inference(self, integration_cases):
        """Test temporal inference."""
        case = integration_cases[3]  # temporal_inference case
        engine = MessagePassingEngine(case.input_data['network'])
        result = engine.run_inference(
            query_variables=case.input_data['query_variables'],
            evidence=case.input_data['evidence'],
            time_steps=case.input_data['time_steps']
        )
        assert self._compare_results(result, case.expected_result, case.error_bound)

    def _compare_results(self, actual: Dict, expected: Dict, tolerance: float) -> bool:
        """Compare actual and expected results within tolerance."""
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return abs(actual - expected) < tolerance
        
        for key in expected:
            if key not in actual:
                return False
            if isinstance(expected[key], dict):
                if not self._compare_results(actual[key], expected[key], tolerance):
                    return False
            else:
                if abs(actual[key] - expected[key]) >= tolerance:
                    return False
        return True