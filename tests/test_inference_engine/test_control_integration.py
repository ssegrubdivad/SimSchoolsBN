# tests/test_inference_engine/test_control_integration.py

import pytest
import numpy as np
from typing import Dict, List

from src.validation.control_integration_validator import (
    ControlIntegrationValidator,
    IntegrationValidationResult
)
from src.network_structure.bayesian_network import BayesianNetwork
from src.network_structure.node import Node
from src.probability_distribution import (
    DiscreteDistribution,
    GaussianDistribution,
    CLGDistribution
)
from src.education_models.locus_control import (
    ControlLevel,
    ControlScope
)

class TestControlIntegrationValidator:
    """Test suite for control integration validation."""

    @pytest.fixture
    def test_network(self) -> BayesianNetwork:
        """Create test network with control-aware nodes."""
        network = BayesianNetwork("test_network")
        
        # Create nodes with different control levels
        student_node = Node(
            id="student_performance",
            name="Student Performance",
            variable_type="continuous"
        )
        student_node.control_scope = ControlScope(
            primary_level=ControlLevel.STUDENT,
            secondary_levels={ControlLevel.TEACHER},
            influence_weight=1.0,
            requires_coordination=False
        )
        
        teacher_node = Node(
            id="teacher_quality",
            name="Teacher Quality",
            variable_type="discrete"
        )
        teacher_node.control_scope = ControlScope(
            primary_level=ControlLevel.TEACHER,
            secondary_levels={ControlLevel.SCHOOL_ADMIN},
            influence_weight=0.8,
            requires_coordination=True
        )
        
        admin_node = Node(
            id="resource_allocation",
            name="Resource Allocation",
            variable_type="clg"
        )
        admin_node.control_scope = ControlScope(
            primary_level=ControlLevel.SCHOOL_ADMIN,
            secondary_levels={ControlLevel.DISTRICT_ADMIN},
            influence_weight=0.7,
            requires_coordination=True
        )
        
        # Add nodes to network
        network.add_node(student_node)
        network.add_node(teacher_node)
        network.add_node(admin_node)
        
        # Add edges
        network.add_edge(teacher_node.id, student_node.id)
        network.add_edge(admin_node.id, teacher_node.id)
        
        return network

    @pytest.fixture
    def validator(self, test_network) -> ControlIntegrationValidator:
        """Create validator instance with test network."""
        return ControlIntegrationValidator(test_network)

    def test_component_interaction_validation(self, validator):
        """Test validation of component interactions."""
        result = validator._validate_component_interactions()
        assert result.is_valid, f"Component interaction validation failed: {result.message}"
        
        # Verify all interaction categories were tested
        assert 'message_computation' in result.details['interactions']
        assert 'evidence_handling' in result.details['interactions']
        assert 'visualization' in result.details['interactions']
        
        # Verify successful interactions
        for category, tests in result.details['interactions'].items():
            failed = [t for t in tests if not t.get('success', False)]
            assert not failed, f"Failed {category} tests: {failed}"

    def test_control_flow_validation(self, validator):
        """Test validation of control flow consistency."""
        result = validator._validate_control_flow()
        assert result.is_valid, f"Control flow validation failed: {result.message}"
        
        # Verify all control levels tested
        control_tests = result.details['control_tests']
        levels_tested = {test['level'] for test in control_tests}
        assert levels_tested == set(ControlLevel)
        
        # Verify direct and secondary control
        test_types = {test['type'] for test in control_tests}
        assert 'direct' in test_types
        assert 'secondary' in test_types

    def test_mathematical_precision_validation(self, validator):
        """Test validation of mathematical precision."""
        result = validator._validate_mathematical_precision()
        assert result.is_valid, f"Mathematical precision validation failed: {result.message}"
        
        # Verify precision categories
        precision_tests = result.details['precision_tests']
        assert 'probability_sums' in precision_tests
        assert 'influence_weights' in precision_tests
        assert 'error_bounds' in precision_tests
        
        # Check probability sums
        for test in precision_tests['probability_sums']:
            if test.get('success'):
                probs = test.get('results', {})
                for message in probs.values():
                    if hasattr(message, 'probabilities'):
                        prob_sum = sum(message.probabilities.values())
                        assert abs(prob_sum - 1.0) < 1e-10

    def test_influence_weight_precision(self, validator):
        """Test precision of influence weight calculations."""
        result = validator._test_influence_weight_precision("teacher_quality")
        assert result['success'], f"Influence weight test failed: {result.get('error')}"
        
        # Verify weight properties
        weights = result['weights']
        for level, weight in weights.items():
            assert 0 <= weight <= 1, f"Invalid weight {weight} for level {level}"
            
        # Verify primary level has full weight
        teacher_node = validator.model.nodes["teacher_quality"]
        primary_weight = weights[teacher_node.control_scope.primary_level]
        assert abs(primary_weight - teacher_node.control_scope.influence_weight) < 1e-10

    def test_error_bound_tracking(self, validator):
        """Test error bound tracking through computation chain."""
        result = validator._test_error_bound_tracking("student_performance")
        assert result['success'], f"Error bound tracking failed: {result.get('error')}"
        
        # Verify error chain properties
        error_chain = result['error_chain']
        assert len(error_chain) > 0
        
        # Verify error accumulation
        previous_error = 0.0
        for step in error_chain:
            assert step['cumulative'] >= previous_error
            previous_error = step['cumulative']
            
        # Verify final error bound
        assert result['final_error'] == previous_error

    def test_complete_validation(self, validator):
        """Test complete validation process."""
        result = validator.validate_control_integration()
        assert result.is_valid, f"Complete validation failed: {result.message}"
        
        # Verify all validation aspects
        assert 'component_validation' in result.details
        assert 'control_validation' in result.details
        assert 'precision_validation' in result.details
        
        # Verify no numerical issues
        assert not result.numerical_issues

    def test_invalid_control_flow(self, validator):
        """Test detection of invalid control flow."""
        # Create invalid control relationship
        student_node = validator.model.nodes["student_performance"]
        student_node.control_scope.secondary_levels.add(ControlLevel.DISTRICT_ADMIN)
        
        result = validator._validate_control_flow()
        assert not result.is_valid
        assert "Control flow validation failed" in result.message
        
        # Verify specific failure details
        failed_tests = result.details['failed_tests']
        assert any(test['level'] == ControlLevel.DISTRICT_ADMIN for test in failed_tests)

    def test_precision_violation(self, validator):
        """Test detection of precision violations."""
        # Create precision violation by manipulating probabilities
        teacher_node = validator.model.nodes["teacher_quality"]
        teacher_node.distribution = DiscreteDistribution(
            variable=teacher_node.id,
            parents=[],
            states=["low", "medium", "high"],
            probabilities={
                ("low",): 0.3,
                ("medium",): 0.3,
                ("high",): 0.3
            }  # Sum < 1.0
        )
        
        result = validator._validate_mathematical_precision()
        assert not result.is_valid
        assert "Mathematical precision validation failed" in result.message

    def test_evidence_validation(self, validator):
        """Test evidence handling validation."""
        evidence_tests = validator._generate_evidence_tests()
        
        # Verify test coverage
        nodes_tested = {test['evidence'].variable for test in evidence_tests}
        assert nodes_tested == set(validator.model.nodes.keys())
        
        # Verify evidence properties
        for test in evidence_tests:
            evidence = test['evidence']
            level = test['level']
            node = validator.model.nodes[evidence.variable]
            
            # Verify evidence type matches node type
            if node.variable_type == "discrete":
                assert isinstance(evidence.value, str)
            elif node.variable_type in ["continuous", "clg"]:
                assert isinstance(evidence.value, (int, float))

    def test_influence_path_validity(self, validator):
        """Test validation of influence paths."""
        admin_node = validator.model.nodes["resource_allocation"]
        student_node = validator.model.nodes["student_performance"]
        
        # Test influence path from admin to student
        vis_data = validator.visualizer.highlight_authority_path(
            student_node.id,
            ControlLevel.SCHOOL_ADMIN
        )
        
        # Verify path properties
        assert 'authorityPath' in vis_data
        path = vis_data['authorityPath']
        assert path[0] == ControlLevel.SCHOOL_ADMIN.name
        assert path[-1] == ControlLevel.STUDENT.name
        
        # Verify path validity
        for i in range(len(path) - 1):
            level1 = ControlLevel[path[i]]
            level2 = ControlLevel[path[i + 1]]
            assert validator.control_validator.validate_control_path(level1, level2, "influence")