# tests/test_inference_engine/test_messages.py

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any

class MessagePassingTestCases:
    """Test cases for message passing computations."""
    
    @pytest.fixture
    def discrete_message_cases(self) -> List[PrecisionTestCase]:
        """Test cases for discrete message computation."""
        return [
            # Basic discrete message
            PrecisionTestCase(
                name="basic_discrete_message",
                input_data={
                    'source_node': {
                        'id': 'X1',
                        'type': 'discrete',
                        'states': ['T', 'F']
                    },
                    'incoming_messages': [
                        {
                            'type': 'discrete',
                            'probabilities': {'T': 0.7, 'F': 0.3}
                        }
                    ]
                },
                expected_result={
                    'T': 0.7,
                    'F': 0.3
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'probability_sum': 1.0,
                    'min_probability': 0.0,
                    'max_probability': 1.0
                },
                preconditions=["Valid probability distribution"],
                postconditions=["Sum to 1.0 within error bound"]
            ),
            
            # Multiple incoming messages
            PrecisionTestCase(
                name="multiple_discrete_messages",
                input_data={
                    'source_node': {
                        'id': 'X1',
                        'type': 'discrete',
                        'states': ['T', 'F']
                    },
                    'incoming_messages': [
                        {'type': 'discrete', 'probabilities': {'T': 0.8, 'F': 0.2}},
                        {'type': 'discrete', 'probabilities': {'T': 0.6, 'F': 0.4}}
                    ]
                },
                expected_result={
                    'T': 0.75,  # Normalized product
                    'F': 0.25
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'probability_sum': 1.0,
                    'precision_maintenance': True
                },
                preconditions=["Multiple valid messages"],
                postconditions=["Correct combination"]
            ),
            
            # Edge case: Very small probabilities
            PrecisionTestCase(
                name="small_probability_handling",
                input_data={
                    'source_node': {
                        'id': 'X1',
                        'type': 'discrete',
                        'states': ['T', 'F']
                    },
                    'incoming_messages': [
                        {'type': 'discrete', 'probabilities': {'T': 0.999999, 'F': 0.000001}}
                    ]
                },
                expected_result={
                    'T': 0.999999,
                    'F': 0.000001
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'small_value_handling': True,
                    'precision_maintenance': True
                },
                preconditions=["Contains small probabilities"],
                postconditions=["Maintains numerical stability"]
            )
        ]

    @pytest.fixture
    def gaussian_message_cases(self) -> List[PrecisionTestCase]:
        """Test cases for Gaussian message computation."""
        return [
            # Basic Gaussian message
            PrecisionTestCase(
                name="basic_gaussian_message",
                input_data={
                    'source_node': {
                        'id': 'X2',
                        'type': 'continuous',
                        'distribution': 'gaussian'
                    },
                    'incoming_messages': [
                        {
                            'type': 'gaussian',
                            'mean': 0.0,
                            'variance': 1.0
                        }
                    ]
                },
                expected_result={
                    'mean': 0.0,
                    'variance': 1.0
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'positive_variance': True,
                    'precision_maintenance': True
                },
                preconditions=["Valid Gaussian parameters"],
                postconditions=["Valid Gaussian result"]
            ),
            
            # Precision-weighted combination
            PrecisionTestCase(
                name="gaussian_precision_combination",
                input_data={
                    'source_node': {
                        'id': 'X2',
                        'type': 'continuous',
                        'distribution': 'gaussian'
                    },
                    'incoming_messages': [
                        {'type': 'gaussian', 'mean': 1.0, 'variance': 4.0},
                        {'type': 'gaussian', 'mean': 2.0, 'variance': 1.0}
                    ]
                },
                expected_result={
                    'mean': 1.8,  # Precision-weighted mean
                    'variance': 0.8  # Combined variance
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'precision_weighting': True,
                    'variance_combination': True
                },
                preconditions=["Multiple Gaussian messages"],
                postconditions=["Correct precision-weighted combination"]
            ),
            
            # Numerical stability case
            PrecisionTestCase(
                name="gaussian_stability",
                input_data={
                    'source_node': {
                        'id': 'X2',
                        'type': 'continuous',
                        'distribution': 'gaussian'
                    },
                    'incoming_messages': [
                        {'type': 'gaussian', 'mean': 1e6, 'variance': 1e-6}
                    ]
                },
                expected_result={
                    'mean': 1e6,
                    'variance': 1e-6
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'large_number_handling': True,
                    'small_variance_handling': True
                },
                preconditions=["Extreme parameter values"],
                postconditions=["Maintains numerical stability"]
            )
        ]

    @pytest.fixture
    def clg_message_cases(self) -> List[PrecisionTestCase]:
        """Test cases for CLG message computation."""
        return [
            # Basic CLG message
            PrecisionTestCase(
                name="basic_clg_message",
                input_data={
                    'source_node': {
                        'id': 'X3',
                        'type': 'clg',
                        'discrete_states': ['high', 'low'],
                        'continuous_parents': ['Y']
                    },
                    'incoming_messages': [
                        {
                            'type': 'clg',
                            'discrete_probabilities': {'high': 0.7, 'low': 0.3},
                            'continuous_parameters': {
                                'high': {'mean_base': 1.0, 'coefficients': [1.0], 'variance': 1.0},
                                'low': {'mean_base': 0.0, 'coefficients': [0.5], 'variance': 2.0}
                            }
                        }
                    ]
                },
                expected_result={
                    'discrete': {'high': 0.7, 'low': 0.3},
                    'continuous': {
                        'high': {'mean_base': 1.0, 'coefficients': [1.0], 'variance': 1.0},
                        'low': {'mean_base': 0.0, 'coefficients': [0.5], 'variance': 2.0}
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'discrete_sum': 1.0,
                    'positive_variance': True,
                    'coefficient_precision': True
                },
                preconditions=["Valid CLG parameters"],
                postconditions=["Valid CLG result"]
            ),
            
            # Complex CLG combination
            PrecisionTestCase(
                name="clg_combination",
                input_data={
                    'source_node': {
                        'id': 'X3',
                        'type': 'clg',
                        'discrete_states': ['high', 'low'],
                        'continuous_parents': ['Y']
                    },
                    'incoming_messages': [
                        {
                            'type': 'clg',
                            'discrete_probabilities': {'high': 0.6, 'low': 0.4},
                            'continuous_parameters': {
                                'high': {'mean_base': 1.0, 'coefficients': [1.0], 'variance': 2.0},
                                'low': {'mean_base': 0.0, 'coefficients': [0.5], 'variance': 1.0}
                            }
                        },
                        {
                            'type': 'clg',
                            'discrete_probabilities': {'high': 0.8, 'low': 0.2},
                            'continuous_parameters': {
                                'high': {'mean_base': 2.0, 'coefficients': [0.5], 'variance': 1.0},
                                'low': {'mean_base': 1.0, 'coefficients': [1.0], 'variance': 2.0}
                            }
                        }
                    ]
                },
                expected_result={
                    'discrete': {'high': 0.75, 'low': 0.25},  # Normalized product
                    'continuous': {
                        'high': {'mean_base': 1.67, 'coefficients': [0.67], 'variance': 0.67},
                        'low': {'mean_base': 0.33, 'coefficients': [0.67], 'variance': 0.67}
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'discrete_combination': True,
                    'continuous_combination': True,
                    'precision_maintenance': True
                },
                preconditions=["Multiple CLG messages"],
                postconditions=["Correct combined result"]
            )
        ]