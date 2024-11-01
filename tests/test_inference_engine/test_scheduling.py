# tests/test_inference_engine/test_scheduling.py

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any

class SchedulingTestCases:
    """Test cases for message scheduling."""
    
    @pytest.fixture
    def scheduling_cases(self) -> List[TestCase]:
        """Test cases for schedule creation and optimization."""
        return [
            # Basic schedule
            TestCase(
                name="basic_schedule",
                input_data={
                    'graph': {
                        'nodes': ['X1', 'X2', 'X3'],
                        'edges': [('X1', 'X2'), ('X2', 'X3')]
                    },
                    'query_nodes': {'X3'},
                    'evidence_nodes': {'X1'}
                },
                expected_result=[
                    ('X1', 'X2'),
                    ('X2', 'X3')
                ],
                error_bound=0,  # Discrete ordering
                numerical_requirements={},
                preconditions=["Valid graph structure"],
                postconditions=["Valid message order"]
            ),
            
            # Complex dependencies
            TestCase(
                name="complex_dependencies",
                input_data={
                    'graph': {
                        'nodes': ['X1', 'X2', 'X3', 'X4'],
                        'edges': [
                            ('X1', 'X2'), ('X1', 'X3'),
                            ('X2', 'X4'), ('X3', 'X4')
                        ]
                    },
                    'query_nodes': {'X4'},
                    'evidence_nodes': {'X1'}
                },
                expected_result=[
                    ('X1', 'X2'),
                    ('X1', 'X3'),
                    ('X2', 'X4'),
                    ('X3', 'X4')
                ],
                error_bound=0,
                numerical_requirements={
                    'dependency_preservation': True
                },
                preconditions=["Complex dependency structure"],
                postconditions=["All dependencies satisfied"]
            )
        ]
