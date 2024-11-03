# tests/test_inference_engine/test_performance/test_control_performance.py

import pytest
import time
import numpy as np
from typing import Dict, List, Set
import logging
from dataclasses import dataclass

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
from src.validation.control_integration_validator import ControlIntegrationValidator

@dataclass
class PerformanceMetrics:
    """Track performance metrics for operations."""
    operation: str
    execution_time: float
    memory_usage: int
    message_count: int
    scale_factor: int
    control_levels: int

class ControlPerformanceValidator:
    """Validates performance of control-aware operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: List[PerformanceMetrics] = []
        
    def test_scaling_performance(self, 
                               scale_factors: List[int], 
                               max_control_levels: int = 5) -> Dict[str, List[PerformanceMetrics]]:
        """
        Test performance scaling with network size and control levels.
        
        Args:
            scale_factors: List of network size multipliers
            max_control_levels: Maximum number of control levels to test
            
        Returns:
            Dict mapping operation types to performance metrics
        """
        results = {
            'message_computation': [],
            'evidence_propagation': [],
            'visualization': [],
            'authority_validation': []
        }
        
        for scale in scale_factors:
            for num_levels in range(2, max_control_levels + 1):
                # Create test network of appropriate size
                network = self._create_test_network(scale, num_levels)
                validator = ControlIntegrationValidator(network)
                
                # Test message computation
                metrics = self._test_message_computation(validator, scale, num_levels)
                results['message_computation'].append(metrics)
                
                # Test evidence propagation
                metrics = self._test_evidence_propagation(validator, scale, num_levels)
                results['evidence_propagation'].append(metrics)
                
                # Test visualization
                metrics = self._test_visualization(validator, scale, num_levels)
                results['visualization'].append(metrics)
                
                # Test authority validation
                metrics = self._test_authority_validation(validator, scale, num_levels)
                results['authority_validation'].append(metrics)
                
        return results

    def _create_test_network(self, 
                            scale: int, 
                            num_levels: int) -> BayesianNetwork:
        """Create test network with specified scale and control levels."""
        network = BayesianNetwork(f"test_network_s{scale}_l{num_levels}")
        
        # Create control level hierarchy
        control_levels = list(ControlLevel)[:num_levels]
        
        # Create nodes for each level
        nodes_per_level = 10 * scale
        for level in control_levels:
            for i in range(nodes_per_level):
                node = self._create_test_node(f"{level.name}_{i}", level)
                network.add_node(node)
        
        # Create edges between levels
        self._create_test_edges(network, control_levels, nodes_per_level)
        
        return network

    def _create_test_node(self, 
                         node_id: str, 
                         control_level: ControlLevel) -> Node:
        """Create test node with appropriate distribution and control scope."""
        # Vary node types based on control level
        if control_level == ControlLevel.STUDENT:
            var_type = "continuous"
            distribution = GaussianDistribution(
                variable=node_id,
                parents=[],
                mean=70.0,
                variance=100.0
            )
        elif control_level == ControlLevel.TEACHER:
            var_type = "discrete"
            distribution = DiscreteDistribution(
                variable=node_id,
                parents=[],
                states=["low", "medium", "high"],
                probabilities={
                    ("low",): 0.3,
                    ("medium",): 0.4,
                    ("high",): 0.3
                }
            )
        else:
            var_type = "clg"
            distribution = CLGDistribution(
                variable=node_id,
                continuous_parents=[],
                discrete_parents=[],
                parameters={}
            )
            
        node = Node(node_id, node_id, var_type)
        node.distribution = distribution
        
        # Create control scope with secondary levels
        secondary_levels = set()
        level_list = list(ControlLevel)
        current_idx = level_list.index(control_level)
        if current_idx > 0:
            secondary_levels.add(level_list[current_idx - 1])
        if current_idx < len(level_list) - 1:
            secondary_levels.add(level_list[current_idx + 1])
            
        node.control_scope = ControlScope(
            primary_level=control_level,
            secondary_levels=secondary_levels,
            influence_weight=0.8,
            requires_coordination=len(secondary_levels) > 0
        )
        
        return node

    def _create_test_edges(self,
                          network: BayesianNetwork,
                          control_levels: List[ControlLevel],
                          nodes_per_level: int) -> None:
        """Create edges between nodes respecting control hierarchy."""
        for i in range(len(control_levels) - 1):
            current_level = control_levels[i]
            next_level = control_levels[i + 1]
            
            # Create edges from higher to lower control levels
            for j in range(nodes_per_level):
                parent_id = f"{current_level.name}_{j}"
                child_id = f"{next_level.name}_{j % nodes_per_level}"
                network.add_edge(parent_id, child_id)

    def _test_message_computation(self,
                                validator: ControlIntegrationValidator,
                                scale: int,
                                num_levels: int) -> PerformanceMetrics:
        """Test message computation performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        message_count = 0
        
        # Test message computation for each control level
        for node_id in validator.model.nodes:
            node = validator.model.nodes[node_id]
            if not hasattr(node, 'control_scope'):
                continue
                
            # Compute messages with different control levels
            for level in ControlLevel:
                try:
                    validator.computation_engine.compute_message(
                        node_id,
                        next(iter(node.children)).id if node.children else None,
                        [],
                        level
                    )
                    message_count += 1
                except Exception as e:
                    self.logger.warning(f"Message computation failed: {str(e)}")
                    
        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - memory_start
        
        return PerformanceMetrics(
            operation="message_computation",
            execution_time=execution_time,
            memory_usage=memory_usage,
            message_count=message_count,
            scale_factor=scale,
            control_levels=num_levels
        )

    def _test_evidence_propagation(self,
                                 validator: ControlIntegrationValidator,
                                 scale: int,
                                 num_levels: int) -> PerformanceMetrics:
        """Test evidence propagation performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        message_count = 0
        
        # Generate and process evidence for each control level
        evidence_tests = validator._generate_evidence_tests()
        
        for test in evidence_tests:
            try:
                validator.evidence_propagator.add_evidence(
                    test['evidence'],
                    test['level']
                )
                message_count += 1
            except Exception as e:
                self.logger.warning(f"Evidence propagation failed: {str(e)}")
                
        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - memory_start
        
        return PerformanceMetrics(
            operation="evidence_propagation",
            execution_time=execution_time,
            memory_usage=memory_usage,
            message_count=message_count,
            scale_factor=scale,
            control_levels=num_levels
        )

    def _test_visualization(self,
                          validator: ControlIntegrationValidator,
                          scale: int,
                          num_levels: int) -> PerformanceMetrics:
        """Test visualization performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        message_count = 0
        
        try:
            # Generate visualization with control information
            validator.visualizer.generate_graph_data(
                VisualizationConfig(
                    show_control_levels=True,
                    show_influence_paths=True,
                    highlight_authority_paths=True
                )
            )
            message_count += 1
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {str(e)}")
            
        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - memory_start
        
        return PerformanceMetrics(
            operation="visualization",
            execution_time=execution_time,
            memory_usage=memory_usage,
            message_count=message_count,
            scale_factor=scale,
            control_levels=num_levels
        )

    def _test_authority_validation(self,
                                 validator: ControlIntegrationValidator,
                                 scale: int,
                                 num_levels: int) -> PerformanceMetrics:
        """Test authority validation performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        message_count = 0
        
        control_levels = list(ControlLevel)[:num_levels]
        
        # Test authority path validation between all level pairs
        for i, level1 in enumerate(control_levels):
            for j, level2 in enumerate(control_levels):
                if i != j:
                    try:
                        validator.control_validator.validate_control_path(
                            level1,
                            level2,
                            "computation"
                        )
                        message_count += 1
                    except Exception as e:
                        self.logger.warning(f"Authority validation failed: {str(e)}")
                        
        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - memory_start
        
        return PerformanceMetrics(
            operation="authority_validation",
            execution_time=execution_time,
            memory_usage=memory_usage,
            message_count=message_count,
            scale_factor=scale,
            control_levels=num_levels
        )

    def _get_memory_usage(self) -> int:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss

    def analyze_results(self, 
                       results: Dict[str, List[PerformanceMetrics]]) -> Dict[str, Any]:
        """
        Analyze performance test results.
        
        Returns:
            Dict containing performance analysis metrics.
        """
        analysis = {}
        
        for operation, metrics in results.items():
            # Analyze scaling behavior
            scale_factors = sorted(set(m.scale_factor for m in metrics))
            execution_times = []
            memory_usages = []
            
            for scale in scale_factors:
                scale_metrics = [m for m in metrics if m.scale_factor == scale]
                avg_time = np.mean([m.execution_time for m in scale_metrics])
                avg_memory = np.mean([m.memory_usage for m in scale_metrics])
                execution_times.append(avg_time)
                memory_usages.append(avg_memory)
                
            # Compute scaling factors
            time_scaling = np.polyfit(np.log(scale_factors), np.log(execution_times), 1)[0]
            memory_scaling = np.polyfit(np.log(scale_factors), np.log(memory_usages), 1)[0]
            
            analysis[operation] = {
                'time_complexity': f"O(n^{time_scaling:.2f})",
                'memory_complexity': f"O(n^{memory_scaling:.2f})",
                'max_time': max(execution_times),
                'max_memory': max(memory_usages),
                'scale_factors': scale_factors,
                'execution_times': execution_times,
                'memory_usages': memory_usages
            }
            
        return analysis

def test_control_performance():
    """Test control-aware component performance."""
    validator = ControlPerformanceValidator()
    
    # Test with different scale factors
    scale_factors = [1, 2, 4, 8, 16]
    results = validator.test_scaling_performance(scale_factors)
    
    # Analyze results
    analysis = validator.analyze_results(results)
    
    # Verify performance requirements
    for operation, metrics in analysis.items():
        # Time complexity should be reasonable
        time_power = float(metrics['time_complexity'].split('^')[1][:-1])
        assert time_power < 3.0, f"Time complexity too high for {operation}"
        
        # Memory scaling should be reasonable
        memory_power = float(metrics['memory_complexity'].split('^')[1][:-1])
        assert memory_power < 2.0, f"Memory scaling too high for {operation}"
        
        # Maximum execution time should be reasonable
        assert metrics['max_time'] < 10.0, f"Execution time too high for {operation}"
        
        # Maximum memory usage should be reasonable
        assert metrics['max_memory'] < 1e9, f"Memory usage too high for {operation}"