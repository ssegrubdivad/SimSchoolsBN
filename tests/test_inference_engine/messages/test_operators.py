# tests/test_inference_engine/messages/test_operators.py

import pytest
import numpy as np

from typing import Dict, Tuple

from src.inference_engine.messages.operators import (
    MessageOperator,
    DiscreteMessageOperator,
    GaussianMessageOperator,
    CLGMessageOperator,
    MultivariateGaussianMessageOperator
)
from src.inference_engine.messages import (
    DiscreteMessage,
    GaussianMessage,
    TruncatedGaussianMessage,
    CLGMessage,
    MultivariateGaussianMessage
)

from src.inference_engine.message_computation import (
    DiscreteMessageOperator,
    GaussianMessageOperator,
    CLGMessageOperator,
    MultivariateGaussianMessageOperator,
    ComputationResult
)
from src.probability_distribution import (
    DiscreteDistribution,
    GaussianDistribution,
    CLGDistribution,
    MultivariateGaussianDistribution
)

class TestDiscreteMessageOperator:
    """Tests for discrete message operations."""
    
    @pytest.fixture
    def operator(self):
        return DiscreteMessageOperator()

    @pytest.fixture
    def simple_messages(self) -> Tuple[DiscreteDistribution, DiscreteDistribution]:
        """Create simple discrete messages for testing."""
        msg1 = DiscreteDistribution(
            variables=["X"],
            states={"X": ["0", "1"]},
            probabilities={("0",): 0.3, ("1",): 0.7}
        )
        msg2 = DiscreteDistribution(
            variables=["Y"],
            states={"Y": ["0", "1"]},
            probabilities={("0",): 0.6, ("1",): 0.4}
        )
        return msg1, msg2

    def test_combine_exact_computation(self, operator, simple_messages):
        """Test exact probability computation in message combination."""
        msg1, msg2 = simple_messages
        result = operator.combine(msg1, msg2)
        
        # Check probability sum
        total_prob = sum(result.value.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # Check individual probabilities
        expected = {
            ("0", "0"): 0.3 * 0.6,
            ("0", "1"): 0.3 * 0.4,
            ("1", "0"): 0.7 * 0.6,
            ("1", "1"): 0.7 * 0.4
        }
        normalized = {k: v/sum(expected.values()) for k, v in expected.items()}
        
        for states, prob in result.value.items():
            assert abs(prob - normalized[states]) < result.error_bound

    def test_combine_numerical_stability(self, operator):
        """Test numerical stability with very small probabilities."""
        msg1 = DiscreteDistribution(
            variables=["X"],
            states={"X": ["0", "1"]},
            probabilities={("0",): 1e-7, ("1",): 1-1e-7}
        )
        msg2 = DiscreteDistribution(
            variables=["Y"],
            states={"Y": ["0", "1"]},
            probabilities={("0",): 1e-7, ("1",): 1-1e-7}
        )
        
        result = operator.combine(msg1, msg2)
        
        # Check that very small probabilities are handled correctly
        assert all(p >= 0 for p in result.value.values())
        assert abs(sum(result.value.values()) - 1.0) < 1e-10
        assert result.numerical_issues  # Should report numerical issues

    def test_marginalization_exact(self, operator, simple_messages):
        """Test exact marginalization."""
        msg1, _ = simple_messages
        result = operator.marginalize(msg1, ["X"])
        
        # Check probability sum
        assert abs(sum(result.value.values()) - 1.0) < 1e-10
        
        # Check error bound
        assert result.error_bound < 1e-10

    def test_error_bounds(self, operator, simple_messages):
        """Test that error bounds are correctly computed and maintained."""
        msg1, msg2 = simple_messages
        result = operator.combine(msg1, msg2)
        
        # Error bound should be small for well-conditioned problems
        assert result.error_bound < 1e-10
        
        # Validate computation results
        validation = operator.validate_computation(result)
        assert validation.is_valid
        assert not validation.details.get('numerical_issues')


class TestGaussianMessageOperator:
    """Tests for Gaussian message operations."""
    
    @pytest.fixture
    def operator(self):
        return GaussianMessageOperator()

    @pytest.fixture
    def simple_messages(self) -> Tuple[GaussianDistribution, GaussianDistribution]:
        """Create simple Gaussian messages for testing."""
        msg1 = GaussianDistribution(
            variables=["X"],
            mean=0.0,
            variance=1.0
        )
        msg2 = GaussianDistribution(
            variables=["X"],
            mean=1.0,
            variance=2.0
        )
        return msg1, msg2

    def test_combine_exact_computation(self, operator, simple_messages):
        """Test exact parameter computation in Gaussian combination."""
        msg1, msg2 = simple_messages
        result = operator.combine(msg1, msg2)
        
        # Compute expected values
        expected_precision = 1/1.0 + 1/2.0
        expected_variance = 1/expected_precision
        expected_mean = (0.0/1.0 + 1.0/2.0) * expected_variance
        
        # Check parameters
        assert abs(result.value["variance"] - expected_variance) < 1e-10
        assert abs(result.value["mean"] - expected_mean) < 1e-10

    def test_combine_numerical_stability(self, operator):
        """Test numerical stability with poorly conditioned variances."""
        msg1 = GaussianDistribution(
            variables=["X"],
            mean=0.0,
            variance=1e-8
        )
        msg2 = GaussianDistribution(
            variables=["X"],
            mean=0.0,
            variance=1e8
        )
        
        result = operator.combine(msg1, msg2)
        
        # Should handle large condition numbers
        assert result.numerical_issues
        assert result.error_bound > 0
        assert result.value["variance"] > 0

    def test_marginalization(self, operator, simple_messages):
        """Test Gaussian marginalization."""
        msg1, _ = simple_messages
        result = operator.marginalize(msg1, ["X"])
        
        # Marginalization of single variable should give constant message
        assert np.isinf(result.value["variance"])
        assert result.error_bound == 0.0

    def test_precision_maintenance(self, operator):
        """Test maintenance of numerical precision in calculations."""
        msg1 = GaussianDistribution(
            variables=["X"],
            mean=1234567.89,
            variance=0.0001
        )
        msg2 = GaussianDistribution(
            variables=["X"],
            mean=1234567.89,
            variance=0.0001
        )
        
        result = operator.combine(msg1, msg2)
        
        # Should maintain precision even with large means
        assert abs(result.value["mean"] - 1234567.89) < 1e-8
        assert result.value["variance"] < 0.0001


class TestCLGMessageOperator:
    """Tests for CLG message operations."""
    
    @pytest.fixture
    def operator(self):
        return CLGMessageOperator()

    @pytest.fixture
    def simple_messages(self) -> Tuple[CLGDistribution, CLGDistribution]:
        """Create simple CLG messages for testing."""
        msg1 = CLGDistribution(
            variables=["X", "Y"],
            discrete_variables=["X"],
            continuous_variables=["Y"],
            discrete_states={"X": ["0", "1"]},
            parameters={
                ("0",): {"mean_base": 0.0, "coefficients": [1.0], "variance": 1.0},
                ("1",): {"mean_base": 1.0, "coefficients": [1.0], "variance": 1.0}
            }
        )
        msg2 = CLGDistribution(
            variables=["X", "Y"],
            discrete_variables=["X"],
            continuous_variables=["Y"],
            discrete_states={"X": ["0", "1"]},
            parameters={
                ("0",): {"mean_base": 0.0, "coefficients": [1.0], "variance": 2.0},
                ("1",): {"mean_base": 1.0, "coefficients": [1.0], "variance": 2.0}
            }
        )
        return msg1, msg2

    def test_combine_exact_computation(self, operator, simple_messages):
        """Test exact computation in CLG combination."""
        msg1, msg2 = simple_messages
        result = operator.combine(msg1, msg2)
        
        # Check discrete probabilities
        discrete_probs = result.value['discrete']
        assert abs(sum(discrete_probs.values()) - 1.0) < 1e-10
        
        # Check continuous parameters
        for config in result.value['continuous']:
            params = result.value['continuous'][config]
            assert 'mean_base' in params
            assert 'coefficients' in params
            assert 'variance' in params
            assert params['variance'] > 0

    def test_combine_numerical_stability(self, operator):
        """Test numerical stability with extreme parameters."""
        msg1 = CLGDistribution(
            variables=["X", "Y"],
            discrete_variables=["X"],
            continuous_variables=["Y"],
            discrete_states={"X": ["0", "1"]},
            parameters={
                ("0",): {"mean_base": 0.0, "coefficients": [1e-8], "variance": 1e-8},
                ("1",): {"mean_base": 0.0, "coefficients": [1e-8], "variance": 1e-8}
            }
        )
        msg2 = CLGDistribution(
            variables=["X", "Y"],
            discrete_variables=["X"],
            continuous_variables=["Y"],
            discrete_states={"X": ["0", "1"]},
            parameters={
                ("0",): {"mean_base": 0.0, "coefficients": [1e8], "variance": 1e8},
                ("1",): {"mean_base": 0.0, "coefficients": [1e8], "variance": 1e8}
            }
        )
        
        result = operator.combine(msg1, msg2)
        
        # Should handle extreme parameters
        assert result.numerical_issues
        assert result.error_bound > 0
        
        # Parameters should still be valid
        for config in result.value['continuous']:
            assert result.value['continuous'][config]['variance'] > 0

    def test_marginalization_discrete(self, operator, simple_messages):
        """Test marginalization of discrete variables."""
        msg1, _ = simple_messages
        result = operator.marginalize(msg1, ["X"])
        
        # Should result in a Gaussian message
        assert len(result.value['discrete']) == 1
        assert 'continuous' in result.value
        assert result.error_bound >= 0

    def test_marginalization_continuous(self, operator, simple_messages):
        """Test marginalization of continuous variables."""
        msg1, _ = simple_messages
        result = operator.marginalize(msg1, ["Y"])
        
        # Should result in a discrete message
        assert 'discrete' in result.value
        assert abs(sum(result.value['discrete'].values()) - 1.0) < 1e-10


class TestMultivariateGaussianMessageOperator:
    """Tests for multivariate Gaussian message operations."""
    
    @pytest.fixture
    def operator(self):
        return MultivariateGaussianMessageOperator()

    @pytest.fixture
    def simple_messages(self) -> Tuple[MultivariateGaussianDistribution, MultivariateGaussianDistribution]:
        """Create simple multivariate Gaussian messages for testing."""
        msg1 = MultivariateGaussianDistribution(
            variables=["X", "Y"],
            mean=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 0.5], [0.5, 2.0]])
        )
        msg2 = MultivariateGaussianDistribution(
            variables=["X", "Y"],
            mean=np.array([1.0, 1.0]),
            covariance=np.array([[2.0, 0.0], [0.0, 1.0]])
        )
        return msg1, msg2

    def test_combine_exact_computation(self, operator, simple_messages):
        """Test exact computation in multivariate Gaussian combination."""
        msg1, msg2 = simple_messages
        result = operator.combine(msg1, msg2)
        
        # Check positive definiteness
        try:
            np.linalg.cholesky(result.value['covariance'])
        except np.linalg.LinAlgError:
            pytest.fail("Combined covariance matrix is not positive definite")
        
        # Check dimensions
        assert result.value['mean'].shape == (2,)
        assert result.value['covariance'].shape == (2, 2)
        
        # Check symmetry
        assert np.allclose(
            result.value['covariance'],
            result.value['covariance'].T,
            atol=1e-10
        )

    def test_combine_numerical_stability(self, operator):
        """Test numerical stability with poorly conditioned matrices."""
        msg1 = MultivariateGaussianDistribution(
            variables=["X", "Y"],
            mean=np.array([0.0, 0.0]),
            covariance=np.array([[1e-8, 0], [0, 1e-8]])
        )
        msg2 = MultivariateGaussianDistribution(
            variables=["X", "Y"],
            mean=np.array([0.0, 0.0]),
            covariance=np.array([[1e8, 0], [0, 1e8]])
        )
        
        result = operator.combine(msg1, msg2)
        
        # Should handle poor conditioning
        assert result.numerical_issues
        assert result.error_bound > 0
        
        # Result should still be valid
        assert np.all(np.linalg.eigvals(result.value['covariance']) > 0)

    def test_marginalization(self, operator, simple_messages):
        """Test multivariate Gaussian marginalization."""
        msg1, _ = simple_messages
        result = operator.marginalize(msg1, ["Y"])
        
        # Check dimensions
        assert result.value['mean'].shape == (1,)
        assert result.value['covariance'].shape == (1, 1)
        
        # Check positive definiteness
        assert result.value['covariance'][0, 0] > 0

    def test_precision_form_stability(self, operator, simple_messages):
        """Test stability of precision form calculations."""
        msg1, msg2 = simple_messages
        
        # Modify covariance to be nearly singular
        msg1.covariance[0, 1] = 0.99 * np.sqrt(msg1.covariance[0, 0] * msg1.covariance[1, 1])
        msg1.covariance[1, 0] = msg1.covariance[0, 1]
        
        result = operator.combine(msg1, msg2)
        
        # Should handle near-singular matrices
        assert result.numerical_issues
        assert np.all(np.linalg.eigvals(result.value['covariance']) > 0)