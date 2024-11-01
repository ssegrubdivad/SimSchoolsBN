# tests/test_inference_engine/framework.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from scipy.stats import truncnorm
import logging
from dataclasses import dataclass
from numpy.linalg import LinAlgError
import pytest

@dataclass
class TestCase:
    """Represents a test case with expected results and error bounds."""
    name: str
    input_data: Dict[str, Any]
    expected_result: Any
    error_bound: float
    numerical_requirements: Dict[str, float]
    preconditions: List[str]
    postconditions: List[str]

class PrecisionTestCase(TestCase):
    """Test case with specific precision requirements."""
    def __init__(self, *args, required_precision: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_precision = required_precision
        self.actual_precision: Optional[float] = None

class NumericalTestFramework:
    """
    Framework for testing numerical computations with strict precision requirements.
    Ensures mathematical guarantees are maintained.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.precision_threshold = 1e-10
        self.test_results: Dict[str, TestResult] = {}

    def validate_precision(self, 
                         computed: Union[float, np.ndarray], 
                         expected: Union[float, np.ndarray],
                         test_case: PrecisionTestCase) -> bool:
        """
        Validate computation meets precision requirements.
        
        Args:
            computed: Computed value
            expected: Expected value
            test_case: Test case with precision requirements
            
        Returns:
            bool: Whether precision requirements are met
            
        Raises:
            ValueError: If precision requirements are not met
        """
        if isinstance(computed, np.ndarray):
            error = np.max(np.abs(computed - expected))
        else:
            error = abs(computed - expected)
            
        test_case.actual_precision = error
        
        if error > test_case.required_precision:
            self.logger.error(
                f"Precision requirement not met: error={error}, "
                f"required={test_case.required_precision}"
            )
            return False
            
        return True

    def validate_probability_sum(self, probabilities: Dict[str, float]) -> bool:
        """
        Validate probability distribution sums to 1.
        
        Args:
            probabilities: Dictionary of probabilities
            
        Returns:
            bool: Whether sum is valid
        """
        total = sum(probabilities.values())
        return abs(total - 1.0) < self.precision_threshold

    def validate_covariance_matrix(self, matrix: np.ndarray) -> bool:
        """
        Validate covariance matrix properties.
        
        Args:
            matrix: Covariance matrix
            
        Returns:
            bool: Whether matrix is valid
        """
        # Check symmetry
        if not np.allclose(matrix, matrix.T):
            return False
            
        # Check positive definiteness
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

class TestResult:
    """Detailed test result with precision information."""
    def __init__(self, 
                 test_case: TestCase,
                 passed: bool,
                 actual_result: Any,
                 precision_info: Dict[str, float],
                 error_info: Optional[str] = None):
        self.test_case = test_case
        self.passed = passed
        self.actual_result = actual_result
        self.precision_info = precision_info
        self.error_info = error_info

class MessagePassingTestSuite:
    """
    Test suite for message passing operations.
    Ensures exact computation requirements are met.
    """
    def __init__(self, framework: NumericalTestFramework):
        self.framework = framework
        self.test_cases: List[TestCase] = []

    def test_discrete_message(self, test_case: TestCase):
        """Test discrete message computation."""
        result = compute_discrete_message(**test_case.input_data)
        
        # Validate probabilities
        assert self.framework.validate_probability_sum(result.probabilities)
        
        # Check individual probabilities
        for state, prob in result.probabilities.items():
            expected = test_case.expected_result[state]
            assert abs(prob - expected) < test_case.error_bound
            
        return TestResult(
            test_case=test_case,
            passed=True,
            actual_result=result,
            precision_info={'max_error': max(
                abs(p - test_case.expected_result[s])
                for s, p in result.probabilities.items()
            )}
        )

    def test_gaussian_message(self, test_case: PrecisionTestCase):
        """Test Gaussian message computation."""
        result = compute_gaussian_message(**test_case.input_data)
        
        # Validate mean
        assert abs(result.mean - test_case.expected_result['mean']) < test_case.error_bound
        
        # Validate variance
        assert result.variance > 0
        assert abs(
            result.variance - test_case.expected_result['variance']
        ) < test_case.error_bound
        
        return TestResult(
            test_case=test_case,
            passed=True,
            actual_result=result,
            precision_info={
                'mean_error': abs(result.mean - test_case.expected_result['mean']),
                'variance_error': abs(result.variance - test_case.expected_result['variance'])
            }
        )

    def test_clg_message(self, test_case: PrecisionTestCase):
        """Test CLG message computation."""
        result = compute_clg_message(**test_case.input_data)
        
        # Validate discrete part
        assert self.framework.validate_probability_sum(result.discrete_probabilities)
        
        # Validate continuous part
        for config, params in result.continuous_parameters.items():
            expected = test_case.expected_result['continuous'][config]
            assert abs(params['mean'] - expected['mean']) < test_case.error_bound
            assert params['variance'] > 0
            assert abs(
                params['variance'] - expected['variance']
            ) < test_case.error_bound
            
        return TestResult(
            test_case=test_case,
            passed=True,
            actual_result=result,
            precision_info={
                'discrete_error': max(
                    abs(p - test_case.expected_result['discrete'][s])
                    for s, p in result.discrete_probabilities.items()
                ),
                'continuous_error': max(
                    abs(params['mean'] - test_case.expected_result['continuous'][c]['mean'])
                    for c, params in result.continuous_parameters.items()
                )
            }
        )

class SchedulingTestSuite:
    """
    Test suite for message scheduling.
    Ensures correct message ordering and dependencies.
    """
    def __init__(self, framework: NumericalTestFramework):
        self.framework = framework

    def test_schedule_creation(self, test_case: TestCase):
        """Test schedule creation."""
        schedule = create_message_schedule(**test_case.input_data)
        
        # Validate dependencies
        for entry in schedule:
            assert all(
                dep in [e.message_id for e in schedule[:schedule.index(entry)]]
                for dep in entry.dependencies
            )
            
        return TestResult(
            test_case=test_case,
            passed=True,
            actual_result=schedule,
            precision_info={}
        )

    def test_schedule_optimization(self, test_case: TestCase):
        """Test schedule optimization."""
        optimized = optimize_message_schedule(**test_case.input_data)
        
        # Validate optimization maintains dependencies
        original = test_case.input_data['schedule']
        for i, entry in enumerate(optimized):
            deps = entry.dependencies
            dep_positions = [
                optimized.index(e) for e in optimized 
                if e.message_id in deps
            ]
            assert all(pos < i for pos in dep_positions)
            
        return TestResult(
            test_case=test_case,
            passed=True,
            actual_result=optimized,
            precision_info={}
        )

class EvidenceTestSuite:
    """
    Test suite for evidence handling.
    Ensures exact evidence incorporation.
    """
    def __init__(self, framework: NumericalTestFramework):
        self.framework = framework

    def test_evidence_incorporation(self, test_case: PrecisionTestCase):
        """Test evidence incorporation."""
        result = incorporate_evidence(**test_case.input_data)
        
        # Validate evidence type
        if test_case.input_data['evidence_type'] == 'discrete':
            assert self.framework.validate_probability_sum(result.probabilities)
        elif test_case.input_data['evidence_type'] == 'continuous':
            assert result.variance > 0
            
        return TestResult(
            test_case=test_case,
            passed=True,
            actual_result=result,
            precision_info={'error': test_case.actual_precision}
        )

class IntegrationTestSuite:
    """
    Test suite for component integration.
    Ensures correct interaction between components.
    """
    def __init__(self, framework: NumericalTestFramework):
        self.framework = framework

    def test_end_to_end_inference(self, test_case: TestCase):
        """Test complete inference process."""
        result = run_inference(**test_case.input_data)
        
        # Validate results
        for var, belief in result.beliefs.items():
            expected = test_case.expected_result[var]
            if isinstance(belief, dict):  # Discrete
                assert self.framework.validate_probability_sum(belief)
            elif isinstance(belief, tuple):  # Gaussian
                mean, variance = belief
                assert abs(mean - expected[0]) < test_case.error_bound
                assert variance > 0
                
        return TestResult(
            test_case=test_case,
            passed=True,
            actual_result=result,
            precision_info={
                'max_error': max(
                    result.error_bounds.values()
                )
            }
        )

# Pytest fixtures and utility functions
@pytest.fixture
def test_framework():
    """Provide test framework fixture."""
    return NumericalTestFramework()

@pytest.fixture
def test_suites():
    """Provide test suites fixture."""
    return create_test_suites()

def create_test_suites() -> Dict[str, Any]:
    """Create all test suites."""
    framework = NumericalTestFramework()
    
    return {
        'message_passing': MessagePassingTestSuite(framework),
        'scheduling': SchedulingTestSuite(framework),
        'evidence': EvidenceTestSuite(framework),
        'integration': IntegrationTestSuite(framework)
    }

# Example test case generation
def create_message_test_case() -> PrecisionTestCase:
    """Create test case for message computation."""
    return PrecisionTestCase(
        name="simple_discrete_message",
        input_data={
            'source_node': create_test_node(),
            'incoming_messages': create_test_messages()
        },
        expected_result={
            'state1': 0.7,
            'state2': 0.3
        },
        error_bound=1e-10,
        required_precision=1e-10,
        numerical_requirements={
            'probability_sum': 1e-10,
            'minimum_probability': 0.0
        },
        preconditions=[
            "All incoming messages validated",
            "Node types compatible"
        ],
        postconditions=[
            "Result is valid probability distribution",
            "Error bounds maintained"
        ]
    )


