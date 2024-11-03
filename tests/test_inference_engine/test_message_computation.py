# tests/test_inference_engine/test_message_computation.py

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any

from src.inference_engine.message_computation import (
    ControlAwareMessageComputationEngine,
    ControlAwareComputationResult
)
from src.education_models.locus_control import (
    ControlLevel,
    ControlScope,
    ControlledVariable
)
from ..test_framework import PrecisionTestCase

class TestControlAwareMessageComputation:
    """
    Test suite for control-aware message computation.
    Verifies mathematical guarantees and control influence.
    """
    
    @pytest.fixture
    def controlled_network(self):
        """Create test network with control levels."""
        network = create_test_network()
        
        # Add control scopes to variables
        network.nodes["StudentPerformance"].control_scope = ControlScope(
            primary_level=ControlLevel.STUDENT,
            secondary_levels={ControlLevel.TEACHER},
            influence_weight=0.8,
            requires_coordination=False
        )
        
        network.nodes["TeacherQuality"].control_scope = ControlScope(
            primary_level=ControlLevel.TEACHER,
            secondary_levels={ControlLevel.SCHOOL_ADMIN},
            influence_weight=0.9,
            requires_coordination=True
        )
        
        network.nodes["SchoolResources"].control_scope = ControlScope(
            primary_level=ControlLevel.SCHOOL_ADMIN,
            secondary_levels={ControlLevel.DISTRICT_ADMIN},
            influence_weight=0.7,
            requires_coordination=True
        )
        
        return network

    @pytest.fixture
    def computation_engine(self, controlled_network):
        """Create computation engine with control awareness."""
        return ControlAwareMessageComputationEngine(controlled_network)

    def test_discrete_message_control(self, computation_engine):
        """Test control influence on discrete message computation."""
        # Create test messages
        messages = [
            DiscreteMessage(
                source_id="TeacherQuality",
                target_id="StudentPerformance",
                variables=["TeacherQuality"],
                states={"TeacherQuality": ["low", "medium", "high"]},
                probabilities={
                    ("low",): 0.2,
                    ("medium",): 0.5,
                    ("high",): 0.3
                }
            )
        ]
        
        # Compute with teacher control level
        result = computation_engine.compute_message(
            source_id="TeacherQuality",
            target_id="StudentPerformance",
            incoming_messages=messages,
            control_level=ControlLevel.TEACHER
        )
        
        # Verify probabilities maintain sum with influence weight
        probs = result.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-10
        
        # Verify influence weight effect
        weight = 0.9  # Teacher primary level weight
        uniform_weight = (1 - weight) / 3  # Three states
        
        for state in ["low", "medium", "high"]:
            expected = weight * messages[0].probabilities[(state,)] + uniform_weight
            assert abs(probs[(state,)] - expected) < 1e-10

    def test_gaussian_message_control(self, computation_engine):
        """Test control influence on Gaussian message computation."""
        # Create test messages
        messages = [
            GaussianMessage(
                source_id="SchoolResources",
                target_id="StudentPerformance",
                variables=["SchoolResources"],
                mean=1000000.0,  # $1M mean
                variance=100000.0
            )
        ]
        
        # Compute with school admin control level
        result = computation_engine.compute_message(
            source_id="SchoolResources",
            target_id="StudentPerformance",
            incoming_messages=messages,
            control_level=ControlLevel.SCHOOL_ADMIN
        )
        
        # Verify mean preservation
        assert abs(result.mean - messages[0].mean) < 1e-10
        
        # Verify variance scaling by influence weight
        weight = 0.7  # School admin primary level weight
        expected_variance = messages[0].variance / weight
        assert abs(result.variance - expected_variance) < 1e-10
        
        # Verify distribution validity
        assert result.variance > 0

    def test_clg_message_control(self, computation_engine):
        """Test control influence on CLG message computation."""
        # Create test messages
        messages = [
            CLGMessage(
                source_id="StudentPerformance",
                target_id="TeacherQuality",
                continuous_var="StudentPerformance",
                continuous_parents=["SchoolResources"],
                discrete_parents=["TeacherQuality"],
                parameters={
                    ("low",): {
                        'mean_base': 70.0,
                        'coefficients': [0.5],
                        'variance': 100.0
                    },
                    ("medium",): {
                        'mean_base': 80.0,
                        'coefficients': [0.5],
                        'variance': 80.0
                    },
                    ("high",): {
                        'mean_base': 90.0,
                        'coefficients': [0.5],
                        'variance': 60.0
                    }
                }
            )
        ]
        
        # Compute with student control level
        result = computation_engine.compute_message(
            source_id="StudentPerformance",
            target_id="TeacherQuality",
            incoming_messages=messages,
            control_level=ControlLevel.STUDENT
        )
        
        # Verify parameters for each discrete state
        weight = 0.8  # Student primary level weight
        
        for state in ["low", "medium", "high"]:
            # Mean base should be preserved
            assert abs(
                result.parameters[(state,)]['mean_base'] - 
                messages[0].parameters[(state,)]['mean_base']
            ) < 1e-10
            
            # Coefficients should be scaled
            assert abs(
                result.parameters[(state,)]['coefficients'][0] - 
                weight * messages[0].parameters[(state,)]['coefficients'][0]
            ) < 1e-10
            
            # Variance should be scaled inversely
            expected_variance = messages[0].parameters[(state,)]['variance'] / weight
            assert abs(
                result.parameters[(state,)]['variance'] - 
                expected_variance
            ) < 1e-10
            
            # Verify distribution validity
            assert result.parameters[(state,)]['variance'] > 0

    def test_control_authority_validation(self, computation_engine):
        """Test validation of control level authority."""
        messages = [create_test_message()]
        
        # Test valid authority
        try:
            computation_engine.compute_message(
                source_id="StudentPerformance",
                target_id="TeacherQuality",
                incoming_messages=messages,
                control_level=ControlLevel.STUDENT
            )
        except ValueError:
            pytest.fail("Unexpected authority validation error")
            
        # Test invalid authority
        with pytest.raises(ValueError, match="lacks authority"):
            computation_engine.compute_message(
                source_id="SchoolResources",
                target_id="TeacherQuality",
                incoming_messages=messages,
                control_level=ControlLevel.STUDENT
            )

    def test_mixed_control_computation(self, computation_engine):
        """Test computation with mixed control levels."""
        # Create test messages with different control levels
        messages = [
            create_student_message(ControlLevel.STUDENT),
            create_teacher_message(ControlLevel.TEACHER),
            create_admin_message(ControlLevel.SCHOOL_ADMIN)
        ]
        
        # Compute with district admin control level
        result = computation_engine.compute_message(
            source_id="SchoolResources",
            target_id="StudentPerformance",
            incoming_messages=messages,
            control_level=ControlLevel.DISTRICT_ADMIN
        )
        
        # Verify result maintains mathematical properties
        validation = computation_engine.validate_computation(result)
        assert validation.is_valid, validation.message

    def test_error_propagation_with_control(self, computation_engine):
        """Test error bound tracking with control influence."""
        messages = [create_test_message_with_error()]
        
        # Compute with control influence
        result = computation_engine.compute_message(
            source_id="StudentPerformance",
            target_id="TeacherQuality",
            incoming_messages=messages,
            control_level=ControlLevel.TEACHER
        )
        
        # Verify error bound includes control influence
        assert isinstance(result, ControlAwareComputationResult)
        assert result.error_bound > messages[0].error_bound
        assert result.error_bound < messages[0].error_bound + 1e-15

    @pytest.mark.parametrize("control_level,expected_weight", [
        (ControlLevel.STUDENT, 0.8),
        (ControlLevel.TEACHER, 0.9),
        (ControlLevel.SCHOOL_ADMIN, 0.7),
        (ControlLevel.DISTRICT_ADMIN, 1.0)
    ])
    def test_influence_weight_application(self, computation_engine, 
                                        control_level, expected_weight):
        """Test correct application of influence weights."""
        messages = [create_test_message()]
        
        result = computation_engine.compute_message(
            source_id="StudentPerformance",
            target_id="TeacherQuality",
            incoming_messages=messages,
            control_level=control_level
        )
        
        # Verify influence weight application
        assert isinstance(result, ControlAwareComputationResult)
        assert abs(result.influence_weight - expected_weight) < 1e-10

    def test_numerical_stability_with_control(self, computation_engine):
        """Test numerical stability of control-influenced computation."""
        # Create messages with extreme values
        messages = [create_extreme_value_message()]
        
        result = computation_engine.compute_message(
            source_id="SchoolResources",
            target_id="StudentPerformance",
            incoming_messages=messages,
            control_level=ControlLevel.SCHOOL_ADMIN
        )
        
        # Verify numerical stability
        assert result.numerical_issues is None or len(result.numerical_issues) == 0
        if isinstance(result, GaussianMessage):
            assert result.variance > 1e-13
        elif isinstance(result, CLGMessage):
            for params in result.parameters.values():
                assert params['variance'] > 1e-13

# Helper functions for creating test messages and networks
def create_test_network() -> 'BayesianNetwork':
    """
    Create test network with educational variables.
    
    Returns:
        BayesianNetwork with variables:
        - StudentPerformance (continuous)
        - TeacherQuality (discrete)
        - SchoolResources (continuous)
    """
    network = BayesianNetwork("TestNetwork")
    
    # Add nodes
    student_perf = Node(
        id="StudentPerformance",
        name="Student Performance",
        variable_type="continuous"
    )
    teacher_qual = Node(
        id="TeacherQuality",
        name="Teacher Quality",
        variable_type="discrete"
    )
    teacher_qual.add_states(["low", "medium", "high"])
    
    school_res = Node(
        id="SchoolResources",
        name="School Resources",
        variable_type="continuous"
    )
    
    network.add_node(student_perf)
    network.add_node(teacher_qual)
    network.add_node(school_res)
    
    # Add edges
    network.add_edge(teacher_qual.id, student_perf.id)
    network.add_edge(school_res.id, student_perf.id)
    
    return network

def create_test_message() -> Message:
    """
    Create standard test message with typical values.
    
    Returns:
        CLGMessage with reasonable educational parameters
    """
    return CLGMessage(
        source_id="StudentPerformance",
        target_id="TeacherQuality",
        continuous_var="StudentPerformance",
        continuous_parents=["SchoolResources"],
        discrete_parents=["TeacherQuality"],
        parameters={
            ("low",): {
                'mean_base': 70.0,
                'coefficients': [0.0001],  # Small coefficient for large resource values
                'variance': 100.0
            },
            ("medium",): {
                'mean_base': 80.0,
                'coefficients': [0.0001],
                'variance': 80.0
            },
            ("high",): {
                'mean_base': 90.0,
                'coefficients': [0.0001],
                'variance': 60.0
            }
        }
    )

def create_student_message(control_level: ControlLevel) -> Message:
    """
    Create test message for student performance.
    
    Args:
        control_level: Control level for message
        
    Returns:
        GaussianMessage representing student performance
    """
    return GaussianMessage(
        source_id="StudentPerformance",
        target_id="TeacherQuality",
        variables=["StudentPerformance"],
        mean=85.0,  # Typical grade
        variance=100.0,  # Reasonable variance for grades
        control_level=control_level
    )

def create_teacher_message(control_level: ControlLevel) -> Message:
    """
    Create test message for teacher quality.
    
    Args:
        control_level: Control level for message
        
    Returns:
        DiscreteMessage representing teacher quality
    """
    return DiscreteMessage(
        source_id="TeacherQuality",
        target_id="StudentPerformance",
        variables=["TeacherQuality"],
        states={"TeacherQuality": ["low", "medium", "high"]},
        probabilities={
            ("low",): 0.2,
            ("medium",): 0.5,
            ("high",): 0.3
        },
        control_level=control_level
    )

def create_admin_message(control_level: ControlLevel) -> Message:
    """
    Create test message for school resources.
    
    Args:
        control_level: Control level for message
        
    Returns:
        GaussianMessage representing school resources
    """
    return GaussianMessage(
        source_id="SchoolResources",
        target_id="StudentPerformance",
        variables=["SchoolResources"],
        mean=1000000.0,  # $1M mean budget
        variance=100000000.0,  # Reasonable variance for budget
        control_level=control_level
    )

def create_test_message_with_error() -> Message:
    """
    Create test message with known error bound.
    
    Returns:
        Message with explicit error bound for testing
    """
    message = create_test_message()
    message._error_bound = 1e-10  # Set explicit error bound
    return message

def create_extreme_value_message() -> Message:
    """
    Create test message with extreme values to test numerical stability.
    
    Returns:
        Message with extreme but valid values
    """
    return CLGMessage(
        source_id="SchoolResources",
        target_id="StudentPerformance",
        continuous_var="SchoolResources",
        continuous_parents=[],
        discrete_parents=["TeacherQuality"],
        parameters={
            ("low",): {
                'mean_base': 1e7,  # Very large budget
                'coefficients': [],
                'variance': 1e-8  # Very small variance
            },
            ("medium",): {
                'mean_base': 1e7,
                'coefficients': [],
                'variance': 1e-8
            },
            ("high",): {
                'mean_base': 1e7,
                'coefficients': [],
                'variance': 1e-8
            }
        }
    )

def create_test_clg_sequence() -> List[Message]:
    """
    Create sequence of CLG messages for testing error propagation.
    
    Returns:
        List of CLG messages with known relationships
    """
    messages = []
    # Resource allocation message
    messages.append(CLGMessage(
        source_id="SchoolResources",
        target_id="TeacherQuality",
        continuous_var="SchoolResources",
        continuous_parents=[],
        discrete_parents=["TeacherQuality"],
        parameters={
            ("low",): {
                'mean_base': 800000.0,  # $800K for low quality
                'coefficients': [],
                'variance': 100000.0
            },
            ("medium",): {
                'mean_base': 1000000.0,  # $1M for medium quality
                'coefficients': [],
                'variance': 100000.0
            },
            ("high",): {
                'mean_base': 1200000.0,  # $1.2M for high quality
                'coefficients': [],
                'variance': 100000.0
            }
        }
    ))
    
    # Performance prediction message
    messages.append(CLGMessage(
        source_id="StudentPerformance",
        target_id="TeacherQuality",
        continuous_var="StudentPerformance",
        continuous_parents=["SchoolResources"],
        discrete_parents=["TeacherQuality"],
        parameters={
            ("low",): {
                'mean_base': 70.0,
                'coefficients': [0.00001],  # Small effect of resources
                'variance': 100.0
            },
            ("medium",): {
                'mean_base': 80.0,
                'coefficients': [0.00001],
                'variance': 80.0
            },
            ("high",): {
                'mean_base': 90.0,
                'coefficients': [0.00001],
                'variance': 60.0
            }
        }
    ))
    
    return messages

def verify_distribution_validity(distribution: Distribution) -> bool:
    """
    Verify mathematical validity of distribution.
    
    Args:
        distribution: Distribution to verify
        
    Returns:
        bool indicating whether distribution is valid
    
    Checks:
    - Discrete: probability sum = 1
    - Gaussian: variance > 0
    - CLG: all parameters valid
    """
    if isinstance(distribution, DiscreteDistribution):
        return abs(sum(distribution.probabilities.values()) - 1.0) < 1e-10
        
    elif isinstance(distribution, GaussianDistribution):
        return distribution.variance > 0
        
    elif isinstance(distribution, CLGDistribution):
        for params in distribution.parameters.values():
            if params['variance'] <= 0:
                return False
            if len(params['coefficients']) != len(distribution.continuous_parents):
                return False
        return True
        
    return False

def verify_control_influence(result: ControlAwareComputationResult,
                           original: Message,
                           control_level: ControlLevel) -> bool:
    """
    Verify correct application of control influence.
    
    Args:
        result: Computation result to verify
        original: Original message before control influence
        control_level: Applied control level
        
    Returns:
        bool indicating whether control influence was correctly applied
    """
    if not isinstance(result, ControlAwareComputationResult):
        return False
        
    if result.control_level != control_level:
        return False
        
    # Verify influence weight is in valid range
    if result.influence_weight < 0 or result.influence_weight > 1:
        return False
        
    # Verify authority path if present
    if result.authority_path:
        if result.authority_path[0] != control_level:
            return False
        if not all(a.value < b.value for a, b in 
                  zip(result.authority_path, result.authority_path[1:])):
            return False
            
    return True

def verify_error_bounds(result: ControlAwareComputationResult,
                       original_bounds: List[float]) -> bool:
    """
    Verify error bound tracking in computation result.
    
    Args:
        result: Computation result to verify
        original_bounds: Original error bounds from input messages
        
    Returns:
        bool indicating whether error bounds are valid
    """
    if not hasattr(result, 'error_bound'):
        return False
        
    # Error should be bounded by sum of input errors plus small constant
    max_input_error = max(original_bounds)
    return (result.error_bound >= max_input_error and
            result.error_bound <= max_input_error + 1e-15)