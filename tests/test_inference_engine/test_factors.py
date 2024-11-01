# tests/test_inference_engine/test_factors.py

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any

from src.probability_distribution.factors import (
    Factor,
    DiscreteFactor,
    GaussianFactor,
    TruncatedGaussianFactor,
    CLGFactor,
    MultivariateGaussianFactor
)
from ..test_framework import PrecisionTestCase, NumericalTestFramework

class TestFactorOperations:
    """Test suite for factor operations with strict precision requirements."""
    
    @pytest.fixture
    def numerical_framework(self):
        return NumericalTestFramework()

    @pytest.fixture
    def precision_cases(self) -> List[PrecisionTestCase]:
        """Create test cases for factor operations."""
        return [
            # Discrete × Discrete
            PrecisionTestCase(
                name="discrete_multiplication",
                input_data={
                    'factor1': DiscreteFactor(
                        name="F1",
                        variables=["X"],
                        states={"X": ["0", "1"]},
                        probabilities={(("0",)): 0.3, (("1",)): 0.7}
                    ),
                    'factor2': DiscreteFactor(
                        name="F2",
                        variables=["X"],
                        states={"X": ["0", "1"]},
                        probabilities={(("0",)): 0.4, (("1",)): 0.6}
                    )
                },
                expected_result={
                    (("0",)): 0.12,  # 0.3 * 0.4
                    (("1",)): 0.42   # 0.7 * 0.6
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'probability_sum': True,
                    'non_negative': True
                },
                preconditions=["Valid probability distributions"],
                postconditions=["Valid probability distribution"]
            ),

            # Gaussian × Gaussian
            PrecisionTestCase(
                name="gaussian_multiplication",
                input_data={
                    'factor1': GaussianFactor(
                        name="F1",
                        variables=["X"],
                        parameters={
                            'mean': 0.0,
                            'variance': 1.0
                        }
                    ),
                    'factor2': GaussianFactor(
                        name="F2",
                        variables=["X"],
                        parameters={
                            'mean': 1.0,
                            'variance': 2.0
                        }
                    )
                },
                expected_result={
                    'mean': 0.6666666667,  # Precision-weighted mean
                    'variance': 0.6666666667  # Combined variance
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'positive_variance': True,
                    'precision_form': True
                },
                preconditions=["Valid Gaussian parameters"],
                postconditions=["Valid Gaussian distribution"]
            ),

            # CLG Factor Test
            PrecisionTestCase(
                name="clg_multiplication",
                input_data={
                    'factor1': CLGFactor(
                        name="F1",
                        continuous_var="Y",
                        continuous_parents=["X"],
                        discrete_parents=["D"],
                        discrete_states={"D": ["0", "1"]},
                        parameters={
                            ("0",): {
                                'mean_base': 0.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            },
                            ("1",): {
                                'mean_base': 1.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            }
                        }
                    ),
                    'factor2': CLGFactor(
                        name="F2",
                        continuous_var="Y",
                        continuous_parents=["X"],
                        discrete_parents=["D"],
                        discrete_states={"D": ["0", "1"]},
                        parameters={
                            ("0",): {
                                'mean_base': 0.0,
                                'coefficients': [0.5],
                                'variance': 2.0
                            },
                            ("1",): {
                                'mean_base': 1.0,
                                'coefficients': [0.5],
                                'variance': 2.0
                            }
                        }
                    )
                },
                expected_result={
                    ("0",): {
                        'mean_base': 0.0,
                        'coefficients': [0.8333333333],
                        'variance': 0.6666666667
                    },
                    ("1",): {
                        'mean_base': 1.0,
                        'coefficients': [0.8333333333],
                        'variance': 0.6666666667
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'positive_variance': True,
                    'coefficient_precision': True,
                    'discrete_consistency': True
                },
                preconditions=["Valid CLG parameters"],
                postconditions=["Valid CLG distribution"]
            )
        ]

    def test_discrete_multiplication(self, numerical_framework, precision_cases):
        """Test discrete factor multiplication with precision requirements."""
        case = precision_cases[0]  # discrete_multiplication case
        factor1, factor2 = case.input_data['factor1'], case.input_data['factor2']
        
        result = factor1.multiply(factor2)
        
        # Check probability sum
        prob_sum = sum(result._probability_table.values())
        assert abs(prob_sum - 1.0) < case.error_bound
        
        # Check individual probabilities
        for states, prob in result._probability_table.items():
            assert abs(prob - case.expected_result[states]) < case.error_bound
            assert prob >= 0  # Non-negativity

    def test_gaussian_multiplication(self, numerical_framework, precision_cases):
        """Test Gaussian factor multiplication with precision requirements."""
        case = precision_cases[1]  # gaussian_multiplication case
        factor1, factor2 = case.input_data['factor1'], case.input_data['factor2']
        
        result = factor1.multiply(factor2)
        
        # Check mean
        assert abs(result.parameters['mean'] - 
                  case.expected_result['mean']) < case.error_bound
        
        # Check variance
        assert result.parameters['variance'] > 0  # Positive variance
        assert abs(result.parameters['variance'] - 
                  case.expected_result['variance']) < case.error_bound

    def test_clg_multiplication(self, numerical_framework, precision_cases):
        """Test CLG factor multiplication with precision requirements."""
        case = precision_cases[2]  # clg_multiplication case
        factor1, factor2 = case.input_data['factor1'], case.input_data['factor2']
        
        result = factor1.multiply(factor2)
        
        # Check parameters for each discrete state
        for config in case.expected_result:
            expected = case.expected_result[config]
            actual = result.parameters[config]
            
            # Check mean base
            assert abs(actual['mean_base'] - 
                      expected['mean_base']) < case.error_bound
            
            # Check coefficients
            for c1, c2 in zip(actual['coefficients'], expected['coefficients']):
                assert abs(c1 - c2) < case.error_bound
            
            # Check variance
            assert actual['variance'] > 0  # Positive variance
            assert abs(actual['variance'] - 
                      expected['variance']) < case.error_bound

    def test_factor_error_propagation(self, numerical_framework):
        """Test error bound propagation in factor operations."""
        # Create factors with known error bounds
        factor1 = DiscreteFactor(
            name="F1",
            variables=["X"],
            states={"X": ["0", "1"]},
            probabilities={(("0",)): 0.3, (("1",)): 0.7}
        )
        factor1._error_bound = 1e-10
        
        factor2 = DiscreteFactor(
            name="F2",
            variables=["X"],
            states={"X": ["0", "1"]},
            probabilities={(("0",)): 0.4, (("1",)): 0.6}
        )
        factor2._error_bound = 1e-10
        
        result = factor1.multiply(factor2)
        
        # Error should accumulate but remain bounded
        assert result._error_bound <= factor1._error_bound + factor2._error_bound + 1e-15

    @pytest.fixture
    def marginalization_cases(self) -> List[PrecisionTestCase]:
        """Create test cases for factor marginalization."""
        return [
            # Discrete Marginalization
            PrecisionTestCase(
                name="discrete_marginalization",
                input_data={
                    'factor': DiscreteFactor(
                        name="F1",
                        variables=["X", "Y"],
                        states={
                            "X": ["0", "1"],
                            "Y": ["0", "1"]
                        },
                        probabilities={
                            (("0", "0")): 0.2,
                            (("0", "1")): 0.3,
                            (("1", "0")): 0.1,
                            (("1", "1")): 0.4
                        }
                    ),
                    'variables_to_marginalize': ["Y"]
                },
                expected_result={
                    (("0",)): 0.5,  # 0.2 + 0.3
                    (("1",)): 0.5   # 0.1 + 0.4
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'probability_sum': True,
                    'non_negative': True,
                    'marginalization_consistency': True
                },
                preconditions=["Valid joint distribution"],
                postconditions=["Valid marginal distribution"]
            ),

            # Gaussian Marginalization
            PrecisionTestCase(
                name="gaussian_marginalization",
                input_data={
                    'factor': MultivariateGaussianFactor(
                        name="F1",
                        variables=["X", "Y"],
                        parameters={
                            'mean': np.array([1.0, 2.0]),
                            'covariance': np.array([
                                [2.0, 0.5],
                                [0.5, 1.0]
                            ])
                        }
                    ),
                    'variables_to_marginalize': ["Y"]
                },
                expected_result={
                    'mean': np.array([1.0]),
                    'covariance': np.array([[2.0]])
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'positive_definite': True,
                    'symmetry': True,
                    'precision_maintenance': True
                },
                preconditions=["Valid multivariate Gaussian"],
                postconditions=["Valid marginal Gaussian"]
            ),

            # CLG Marginalization
            PrecisionTestCase(
                name="clg_marginalization",
                input_data={
                    'factor': CLGFactor(
                        name="F1",
                        continuous_var="Y",
                        continuous_parents=["X1", "X2"],
                        discrete_parents=["D"],
                        discrete_states={"D": ["0", "1"]},
                        parameters={
                            ("0",): {
                                'mean_base': 0.0,
                                'coefficients': [1.0, 0.5],
                                'variance': 1.0
                            },
                            ("1",): {
                                'mean_base': 1.0,
                                'coefficients': [1.0, 0.5],
                                'variance': 1.0
                            }
                        }
                    ),
                    'variables_to_marginalize': ["X2"]
                },
                expected_result={
                    ("0",): {
                        'mean_base': 0.0,
                        'coefficients': [1.0],
                        'variance': 1.25  # Original + marginalized coefficient effect
                    },
                    ("1",): {
                        'mean_base': 1.0,
                        'coefficients': [1.0],
                        'variance': 1.25
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'positive_variance': True,
                    'coefficient_consistency': True,
                    'marginalization_validity': True
                },
                preconditions=["Valid CLG distribution"],
                postconditions=["Valid marginalized CLG"]
            )
        ]

    def test_discrete_marginalization(self, numerical_framework, marginalization_cases):
        """Test discrete factor marginalization with precision requirements."""
        case = marginalization_cases[0]
        factor = case.input_data['factor']
        vars_to_marginalize = case.input_data['variables_to_marginalize']
        
        result = factor.marginalize(vars_to_marginalize)
        
        # Check probability sum
        prob_sum = sum(result._probability_table.values())
        assert abs(prob_sum - 1.0) < case.error_bound
        
        # Check individual probabilities
        for states, prob in result._probability_table.items():
            assert abs(prob - case.expected_result[states]) < case.error_bound
            assert prob >= 0  # Non-negativity

    def test_gaussian_marginalization(self, numerical_framework, marginalization_cases):
        """Test Gaussian factor marginalization with precision requirements."""
        case = marginalization_cases[1]
        factor = case.input_data['factor']
        vars_to_marginalize = case.input_data['variables_to_marginalize']
        
        result = factor.marginalize(vars_to_marginalize)
        
        # Check mean
        assert np.allclose(
            result.parameters['mean'],
            case.expected_result['mean'],
            atol=case.error_bound
        )
        
        # Check covariance
        assert np.allclose(
            result.parameters['covariance'],
            case.expected_result['covariance'],
            atol=case.error_bound
        )
        
        # Verify positive definiteness
        eigvals = np.linalg.eigvals(result.parameters['covariance'])
        assert np.all(eigvals > -case.error_bound)

    def test_clg_marginalization(self, numerical_framework, marginalization_cases):
        """Test CLG factor marginalization with precision requirements."""
        case = marginalization_cases[2]
        factor = case.input_data['factor']
        vars_to_marginalize = case.input_data['variables_to_marginalize']
        
        result = factor.marginalize(vars_to_marginalize)
        
        # Check parameters for each discrete state
        for config in case.expected_result:
            expected = case.expected_result[config]
            actual = result.parameters[config]
            
            # Check mean base
            assert abs(actual['mean_base'] - 
                      expected['mean_base']) < case.error_bound
            
            # Check coefficients
            assert len(actual['coefficients']) == len(expected['coefficients'])
            for c1, c2 in zip(actual['coefficients'], expected['coefficients']):
                assert abs(c1 - c2) < case.error_bound
            
            # Check variance
            assert actual['variance'] > 0
            assert abs(actual['variance'] - 
                      expected['variance']) < case.error_bound

    def test_marginalization_error_propagation(self, numerical_framework):
        """Test error bound propagation in marginalization operations."""
        # Create factor with known error bound
        factor = MultivariateGaussianFactor(
            name="F1",
            variables=["X", "Y"],
            parameters={
                'mean': np.array([1.0, 2.0]),
                'covariance': np.array([
                    [2.0, 0.5],
                    [0.5, 1.0]
                ])
            }
        )
        factor._error_bound = 1e-10
        
        result = factor.marginalize(["Y"])
        
        # Error should remain bounded after marginalization
        assert result._error_bound <= factor._error_bound + 1e-15
        
        # Matrix operations should maintain precision
        assert np.linalg.cond(result.parameters['covariance']) < 1e13

    @pytest.fixture
    def mixed_combination_cases(self) -> List[PrecisionTestCase]:
        """Create test cases for mixed factor type combinations."""
        return [
            # Discrete × CLG
            PrecisionTestCase(
                name="discrete_clg_combination",
                input_data={
                    'discrete_factor': DiscreteFactor(
                        name="F1",
                        variables=["D"],
                        states={"D": ["0", "1"]},
                        probabilities={(("0",)): 0.3, (("1",)): 0.7}
                    ),
                    'clg_factor': CLGFactor(
                        name="F2",
                        continuous_var="Y",
                        continuous_parents=["X"],
                        discrete_parents=["D"],
                        discrete_states={"D": ["0", "1"]},
                        parameters={
                            ("0",): {
                                'mean_base': 0.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            },
                            ("1",): {
                                'mean_base': 1.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            }
                        }
                    )
                },
                expected_result={
                    ("0",): {
                        'mean_base': 0.0,
                        'coefficients': [1.0],
                        'variance': 1.0,
                        'weight': 0.3
                    },
                    ("1",): {
                        'mean_base': 1.0,
                        'coefficients': [1.0],
                        'variance': 1.0,
                        'weight': 0.7
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'discrete_consistency': True,
                    'continuous_validity': True,
                    'weight_normalization': True
                },
                preconditions=["Valid discrete and CLG factors"],
                postconditions=["Valid weighted CLG distribution"]
            ),

            # Gaussian × CLG
            PrecisionTestCase(
                name="gaussian_clg_combination",
                input_data={
                    'gaussian_factor': GaussianFactor(
                        name="F1",
                        variables=["X"],
                        parameters={
                            'mean': 0.0,
                            'variance': 2.0
                        }
                    ),
                    'clg_factor': CLGFactor(
                        name="F2",
                        continuous_var="Y",
                        continuous_parents=["X"],
                        discrete_parents=["D"],
                        discrete_states={"D": ["0", "1"]},
                        parameters={
                            ("0",): {
                                'mean_base': 0.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            },
                            ("1",): {
                                'mean_base': 1.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            }
                        }
                    )
                },
                expected_result={
                    ("0",): {
                        'mean_base': 0.0,
                        'coefficients': [0.6666666667],  # Precision-weighted
                        'variance': 0.6666666667
                    },
                    ("1",): {
                        'mean_base': 1.0,
                        'coefficients': [0.6666666667],
                        'variance': 0.6666666667
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'coefficient_precision': True,
                    'variance_validity': True,
                    'discrete_preservation': True
                },
                preconditions=["Valid Gaussian and CLG factors"],
                postconditions=["Valid combined CLG distribution"]
            ),

            # Edge Case: Near-Deterministic CLG
            PrecisionTestCase(
                name="near_deterministic_clg",
                input_data={
                    'factor1': CLGFactor(
                        name="F1",
                        continuous_var="Y",
                        continuous_parents=["X"],
                        discrete_parents=["D"],
                        discrete_states={"D": ["0", "1"]},
                        parameters={
                            ("0",): {
                                'mean_base': 0.0,
                                'coefficients': [1.0],
                                'variance': 1e-8
                            },
                            ("1",): {
                                'mean_base': 1.0,
                                'coefficients': [1.0],
                                'variance': 1e-8
                            }
                        }
                    ),
                    'factor2': CLGFactor(
                        name="F2",
                        continuous_var="Y",
                        continuous_parents=["X"],
                        discrete_parents=["D"],
                        discrete_states={"D": ["0", "1"]},
                        parameters={
                            ("0",): {
                                'mean_base': 0.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            },
                            ("1",): {
                                'mean_base': 1.0,
                                'coefficients': [1.0],
                                'variance': 1.0
                            }
                        }
                    )
                },
                expected_result={
                    ("0",): {
                        'mean_base': 0.0,
                        'coefficients': [1.0],
                        'variance': 1e-8  # Near-deterministic dominates
                    },
                    ("1",): {
                        'mean_base': 1.0,
                        'coefficients': [1.0],
                        'variance': 1e-8
                    }
                },
                error_bound=1e-10,
                required_precision=1e-10,
                numerical_requirements={
                    'numerical_stability': True,
                    'precision_maintenance': True,
                    'variance_validity': True
                },
                preconditions=["Valid CLG factors with extreme variance"],
                postconditions=["Valid near-deterministic CLG"]
            )
        ]

    def test_discrete_clg_combination(self, numerical_framework, mixed_combination_cases):
        """Test combination of discrete and CLG factors."""
        case = mixed_combination_cases[0]
        discrete_factor = case.input_data['discrete_factor']
        clg_factor = case.input_data['clg_factor']
        
        result = discrete_factor.multiply(clg_factor)
        
        # Check discrete weights
        weight_sum = 0.0
        for config in case.expected_result:
            actual = result.parameters[config]
            expected = case.expected_result[config]
            weight_sum += actual['weight']
            
            assert abs(actual['weight'] - expected['weight']) < case.error_bound
            
        # Verify weight normalization
        assert abs(weight_sum - 1.0) < case.error_bound
        
        # Check continuous parameters
        for config in case.expected_result:
            actual = result.parameters[config]
            expected = case.expected_result[config]
            
            assert abs(actual['mean_base'] - expected['mean_base']) < case.error_bound
            assert abs(actual['variance'] - expected['variance']) < case.error_bound
            
            for c1, c2 in zip(actual['coefficients'], expected['coefficients']):
                assert abs(c1 - c2) < case.error_bound

    def test_gaussian_clg_combination(self, numerical_framework, mixed_combination_cases):
        """Test combination of Gaussian and CLG factors."""
        case = mixed_combination_cases[1]
        gaussian_factor = case.input_data['gaussian_factor']
        clg_factor = case.input_data['clg_factor']
        
        result = gaussian_factor.multiply(clg_factor)
        
        # Check parameters for each discrete state
        for config in case.expected_result:
            actual = result.parameters[config]
            expected = case.expected_result[config]
            
            assert abs(actual['mean_base'] - expected['mean_base']) < case.error_bound
            assert actual['variance'] > 0
            assert abs(actual['variance'] - expected['variance']) < case.error_bound
            
            for c1, c2 in zip(actual['coefficients'], expected['coefficients']):
                assert abs(c1 - c2) < case.error_bound

    def test_near_deterministic_clg(self, numerical_framework, mixed_combination_cases):
        """Test combination of CLG factors with extreme variances."""
        case = mixed_combination_cases[2]
        factor1 = case.input_data['factor1']
        factor2 = case.input_data['factor2']
        
        result = factor1.multiply(factor2)
        
        # Verify numerical stability
        assert result._numerical_issues is None or len(result._numerical_issues) == 0
        
        # Check parameters
        for config in case.expected_result:
            actual = result.parameters[config]
            expected = case.expected_result[config]
            
            assert abs(actual['mean_base'] - expected['mean_base']) < case.error_bound
            assert actual['variance'] > 0
            assert abs(actual['variance'] - expected['variance']) < case.error_bound
            assert actual['variance'] < 1e-7  # Verify near-deterministic behavior

            for c1, c2 in zip(actual['coefficients'], expected['coefficients']):
                assert abs(c1 - c2) < case.error_bound