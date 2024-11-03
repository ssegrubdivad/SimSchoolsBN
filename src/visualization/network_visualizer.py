# src/visualization/network_visualizer.py

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import logging
import json

from src.visualization.network_visualizer import NetworkVisualizer
from src.inference_engine.message_passing import InferenceResult
from src.network_structure.bayesian_network import BayesianNetwork

from ..education_models.locus_control import (
    ControlLevel,
    ControlScope,
    ControlValidator,
    ControlledVariable
)

@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    show_probabilities: bool = True
    show_error_bounds: bool = True
    show_confidence: bool = True
    color_scheme: str = "educational"  # educational/standard/custom
    layout_type: str = "hierarchical"  # hierarchical/radial/force-directed
    show_control_levels: bool = True
    show_influence_paths: bool = True
    highlight_authority_paths: bool = True

class NetworkVisualizer:
    """Base network visualization functionality."""
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network

    def generate_graph_data(self):
        nodes = []
        links = []

        for node_id, node in self.bayesian_network.nodes.items():
            nodes.append({"id": node_id, "name": node.name})

        for edge in self.bayesian_network.edges:
            links.append({"source": edge.parent.id, "target": edge.child.id})

        return {"nodes": nodes, "links": links}

    def generate_html(self):
        graph_data = self.generate_graph_data()
        return json.dumps(graph_data)

class VisualizationIntegrator:
    """
    Integrates inference results with visualization system.
    Maintains exact representation of computational results.
    """
    def __init__(self, model: BayesianNetwork):
        """
        Initialize visualization integrator.
        
        Args:
            model: The Bayesian network model
        """
        self.model = model
        self.network_visualizer = NetworkVisualizer(model)
        self.logger = logging.getLogger(__name__)
        
        # Track visualization state
        self.current_results: Optional[InferenceResult] = None
        self.current_config: Optional[VisualizationConfig] = None

    def create_visualization(self, 
                           inference_result: InferenceResult,
                           config: Optional[VisualizationConfig] = None) -> Dict[str, Any]:
        """
        Create visualization from inference results.
        
        Args:
            inference_result: Results from inference computation
            config: Optional visualization configuration
            
        Returns:
            Visualization data structure
            
        Raises:
            ValueError: If visualization cannot be created
        """
        try:
            self.current_results = inference_result
            self.current_config = config or VisualizationConfig()
            
            # Create base network visualization
            vis_data = self.network_visualizer.generate_graph_data()
            
            # Enhance with inference results
            self._enhance_visualization(vis_data)
            
            # Add error bounds and confidence information
            if self.current_config.show_error_bounds:
                self._add_error_information(vis_data)
                
            # Apply educational color scheme if specified
            if self.current_config.color_scheme == "educational":
                self._apply_educational_colors(vis_data)
                
            return vis_data
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            raise

    def _enhance_visualization(self, vis_data: Dict[str, Any]) -> None:
        """
        Enhance visualization with inference results.
        Maintains exact representation of probabilities.
        """
        for node in vis_data['nodes']:
            node_id = node['id']
            if node_id in self.current_results.beliefs:
                belief = self.current_results.beliefs[node_id]
                
                # Add belief information based on variable type
                if isinstance(belief, dict):  # Discrete
                    self._add_discrete_belief(node, belief)
                elif isinstance(belief, tuple):  # Gaussian
                    self._add_continuous_belief(node, belief)
                elif isinstance(belief, dict) and 'type' in belief:  # CLG
                    self._add_clg_belief(node, belief)

    def _add_discrete_belief(self, 
                           node: Dict[str, Any],
                           belief: Dict[str, float]) -> None:
        """Add discrete belief information to node."""
        if self.current_config.show_probabilities:
            # Format probabilities with exact precision
            node['probabilities'] = {
                state: f"{prob:.10g}"  # Use g format for clean representation
                for state, prob in belief.items()
            }
            
            # Add mode (most probable state)
            mode_state = max(belief.items(), key=lambda x: x[1])[0]
            node['mode'] = mode_state
            
            # Add entropy if confidence display is enabled
            if self.current_config.show_confidence:
                entropy = -sum(p * np.log2(p) for p in belief.values() if p > 0)
                node['entropy'] = entropy

    def _add_continuous_belief(self,
                             node: Dict[str, Any],
                             belief: tuple) -> None:
        """Add continuous belief information to node."""
        mean, variance = belief
        
        if self.current_config.show_probabilities:
            # Store exact mean and variance
            node['mean'] = f"{mean:.10g}"
            node['variance'] = f"{variance:.10g}"
            
            # Add confidence interval if requested
            if self.current_config.show_confidence:
                std_dev = np.sqrt(variance)
                node['confidence_interval'] = {
                    'lower': f"{(mean - 2*std_dev):.10g}",
                    'upper': f"{(mean + 2*std_dev):.10g}"
                }

    def _add_clg_belief(self,
                       node: Dict[str, Any],
                       belief: Dict[str, Any]) -> None:
        """Add CLG belief information to node."""
        if self.current_config.show_probabilities:
            # Add discrete component
            if 'discrete' in belief:
                node['discrete_probabilities'] = {
                    state: f"{prob:.10g}"
                    for state, prob in belief['discrete'].items()
                }
                
            # Add continuous component
            if 'continuous' in belief:
                node['continuous_parameters'] = {
                    config: {
                        'mean': f"{params['mean']:.10g}",
                        'variance': f"{params['variance']:.10g}"
                    }
                    for config, params in belief['continuous'].items()
                }

    def _add_error_information(self, vis_data: Dict[str, Any]) -> None:
        """Add error bound information to visualization."""
        for node in vis_data['nodes']:
            node_id = node['id']
            if node_id in self.current_results.error_bounds:
                error_bound = self.current_results.error_bounds[node_id]
                node['error_bound'] = f"{error_bound:.10g}"
                
                # Add numerical issues if any
                if (self.current_results.numerical_issues and 
                    node_id in self.current_results.numerical_issues):
                    node['numerical_issues'] = self.current_results.numerical_issues[node_id]

    def _apply_educational_colors(self, vis_data: Dict[str, Any]) -> None:
        """Apply education-specific color scheme."""
        for node in vis_data['nodes']:
            node_type = self.model.nodes[node['id']].variable_type
            
            if node_type == 'discrete':
                if 'probabilities' in node:
                    # Color based on certainty
                    max_prob = max(float(p) for p in node['probabilities'].values())
                    node['color'] = self._get_certainty_color(max_prob)
                    
            elif node_type == 'continuous':
                if 'variance' in node:
                    # Color based on precision
                    variance = float(node['variance'])
                    node['color'] = self._get_precision_color(variance)
                    
            elif node_type == 'clg':
                # Use special color for CLG nodes
                node['color'] = "#9370DB"  # Medium purple

    def _get_certainty_color(self, probability: float) -> str:
        """Get color based on probability (certainty)."""
        # Use educational color scheme
        if probability > 0.9:
            return "#2E8B57"  # Sea Green (High certainty)
        elif probability > 0.7:
            return "#4682B4"  # Steel Blue (Good certainty)
        elif probability > 0.5:
            return "#DAA520"  # Goldenrod (Moderate certainty)
        else:
            return "#CD853F"  # Peru (Low certainty)

    def _get_precision_color(self, variance: float) -> str:
        """Get color based on variance (precision)."""
        # Use educational color scheme
        if variance < 0.01:
            return "#2E8B57"  # Sea Green (High precision)
        elif variance < 0.1:
            return "#4682B4"  # Steel Blue (Good precision)
        elif variance < 1.0:
            return "#DAA520"  # Goldenrod (Moderate precision)
        else:
            return "#CD853F"  # Peru (Low precision)

    def create_detailed_view(self, 
                           node_id: str) -> Dict[str, Any]:
        """
        Create detailed view for a specific node.
        
        Args:
            node_id: ID of node to detail
            
        Returns:
            Detailed visualization data for node
        """
        if not self.current_results or node_id not in self.current_results.beliefs:
            raise ValueError(f"No results available for node {node_id}")
            
        node = self.model.nodes[node_id]
        belief = self.current_results.beliefs[node_id]
        
        detailed_data = {
            'node_id': node_id,
            'variable_type': node.variable_type,
            'belief': self._format_belief_for_detail(belief),
            'error_bound': self.current_results.error_bounds[node_id]
        }
        
        # Add relationships
        detailed_data['parents'] = [p.id for p in node.parents]
        detailed_data['children'] = [c.id for c in node.children]
        
        # Add educational context if available
        if hasattr(node, 'educational_context'):
            detailed_data['educational_context'] = node.educational_context
            
        return detailed_data

    def _format_belief_for_detail(self,
                                belief: Union[Dict[str, float], tuple, Dict[str, Any]]) -> Dict[str, Any]:
        """Format belief for detailed view."""
        if isinstance(belief, dict) and 'type' not in belief:  # Discrete
            return {
                'type': 'discrete',
                'probabilities': {
                    state: f"{prob:.10g}"
                    for state, prob in belief.items()
                }
            }
        elif isinstance(belief, tuple):  # Gaussian
            mean, variance = belief
            return {
                'type': 'continuous',
                'mean': f"{mean:.10g}",
                'variance': f"{variance:.10g}",
                'std_dev': f"{np.sqrt(variance):.10g}"
            }
        else:  # CLG
            return {
                'type': 'clg',
                'discrete': belief.get('discrete', {}),
                'continuous': belief.get('continuous', {})
            }

class ControlAwareNetworkVisualizer(NetworkVisualizer):
    """
    Enhanced network visualization with control level awareness.
    Maintains exact representation of network structure and control relationships.
    """
    def __init__(self, bayesian_network):
        super().__init__(bayesian_network)
        self.control_validator = ControlValidator()
        self.variable_controls: Dict[str, ControlledVariable] = {}
        self._initialize_control_mappings()
        self.logger = logging.getLogger(__name__)
        
        # Control level color scheme
        self.control_colors = {
            ControlLevel.STUDENT: "#4CAF50",      # Green
            ControlLevel.TEACHER: "#2196F3",      # Blue
            ControlLevel.PARENT: "#FFC107",       # Amber
            ControlLevel.SCHOOL_ADMIN: "#9C27B0",  # Purple
            ControlLevel.DISTRICT_ADMIN: "#F44336" # Red
        }

    def _initialize_control_mappings(self) -> None:
        """Initialize control mappings for visualization."""
        for node_id, node in self.bayesian_network.nodes.items():
            if hasattr(node, 'control_scope'):
                self.variable_controls[node_id] = ControlledVariable(
                    name=node_id,
                    control_scope=node.control_scope,
                    validator=self.control_validator
                )

    def generate_graph_data(self, 
                          config: Optional[VisualizationConfig] = None) -> Dict[str, Any]:
        """
        Generate visualization data with control information.
        
        Args:
            config: Optional visualization configuration
            
        Returns:
            Dict containing graph visualization data with control enhancements
        """
        config = config or VisualizationConfig()
        
        # Get base graph data
        graph_data = super().generate_graph_data()
        
        if config.show_control_levels:
            self._add_control_information(graph_data)
            
        if config.show_influence_paths:
            self._add_influence_paths(graph_data)
            
        return graph_data

    def _add_control_information(self, graph_data: Dict[str, Any]) -> None:
        """
        Add control level information to nodes and edges.
        Preserves exact network structure while adding control visualization.
        """
        # Enhance nodes with control information
        for node in graph_data['nodes']:
            node_id = node['id']
            if node_id in self.variable_controls:
                controlled_var = self.variable_controls[node_id]
                control_scope = controlled_var.control_scope
                
                # Add control level information
                node['controlLevel'] = control_scope.primary_level.name
                node['secondaryLevels'] = [
                    level.name for level in control_scope.secondary_levels
                ]
                node['influenceWeight'] = control_scope.influence_weight
                node['requiresCoordination'] = control_scope.requires_coordination
                
                # Add control level color
                node['color'] = self.control_colors[control_scope.primary_level]
                
                # Add border style for coordination requirement
                if control_scope.requires_coordination:
                    node['borderStyle'] = 'dashed'
                    node['borderWidth'] = 3

        # Enhance edges with control relationship information
        for edge in graph_data['links']:
            source_id = edge['source']
            target_id = edge['target']
            
            if (source_id in self.variable_controls and 
                target_id in self.variable_controls):
                source_level = self.variable_controls[source_id].control_scope.primary_level
                target_level = self.variable_controls[target_id].control_scope.primary_level
                
                # Add control relationship type
                if source_level.value < target_level.value:
                    edge['controlType'] = 'upward'
                    edge['strokeDasharray'] = '5,5'
                elif source_level.value > target_level.value:
                    edge['controlType'] = 'downward'
                    edge['strokeWidth'] = 2
                else:
                    edge['controlType'] = 'same_level'

    def _add_influence_paths(self, graph_data: Dict[str, Any]) -> None:
        """
        Add influence path information to visualization.
        Shows how control levels influence each other.
        """
        influence_paths = []
        
        # Generate influence paths between control levels
        for source_node in graph_data['nodes']:
            if 'controlLevel' not in source_node:
                continue
                
            for target_node in graph_data['nodes']:
                if 'controlLevel' not in target_node:
                    continue
                    
                source_level = ControlLevel[source_node['controlLevel']]
                target_level = ControlLevel[target_node['controlLevel']]
                
                # Get influence path if it exists
                path = self.control_validator.get_influence_path(
                    source_level,
                    target_level
                )
                
                if path:
                    influence_paths.append({
                        'path': [level.name for level in path],
                        'source': source_node['id'],
                        'target': target_node['id'],
                        'strength': self._calculate_influence_strength(path)
                    })
        
        graph_data['influencePaths'] = influence_paths

    def _calculate_influence_strength(self, 
                                    path: List[ControlLevel]) -> float:
        """
        Calculate cumulative influence strength along a path.
        Maintains exact influence weight calculations.
        """
        strength = 1.0
        for i in range(len(path) - 1):
            current = path[i]
            next_level = path[i + 1]
            
            # Get variables controlled by current level
            current_vars = [
                var for var, ctrl in self.variable_controls.items()
                if ctrl.control_scope.primary_level == current
            ]
            
            # Calculate average influence weight
            if current_vars:
                weights = [
                    self.variable_controls[var].get_influence_weight(next_level)
                    for var in current_vars
                ]
                strength *= sum(weights) / len(weights)
            
        return strength

    def highlight_authority_path(self, 
                               start_node: str,
                               control_level: ControlLevel) -> Dict[str, Any]:
        """
        Generate visualization data highlighting specific authority path.
        
        Args:
            start_node: Starting node ID
            control_level: Control level to highlight path from
            
        Returns:
            Dict containing highlighted path information
        """
        if start_node not in self.variable_controls:
            return {}
            
        controlled_var = self.variable_controls[start_node]
        target_level = controlled_var.control_scope.primary_level
        
        # Get authority path
        path = self.control_validator.get_influence_path(
            control_level,
            target_level
        )
        
        if not path:
            return {}
            
        # Create highlighted path data
        highlighted_nodes = set()
        highlighted_edges = set()
        
        for i in range(len(path) - 1):
            current_level = path[i]
            next_level = path[i + 1]
            
            # Find nodes at these levels
            current_nodes = {
                node_id for node_id, ctrl in self.variable_controls.items()
                if ctrl.control_scope.primary_level == current_level
            }
            next_nodes = {
                node_id for node_id, ctrl in self.variable_controls.items()
                if ctrl.control_scope.primary_level == next_level
            }
            
            highlighted_nodes.update(current_nodes)
            if i == len(path) - 2:  # Add final level nodes
                highlighted_nodes.update(next_nodes)
                
            # Add edges between levels
            for curr in current_nodes:
                for next_node in next_nodes:
                    edge = (curr, next_node)
                    if self._edge_exists(edge):
                        highlighted_edges.add(edge)
        
        return {
            'highlightedNodes': list(highlighted_nodes),
            'highlightedEdges': list(highlighted_edges),
            'authorityPath': [level.name for level in path]
        }

    def _edge_exists(self, edge: Tuple[str, str]) -> bool:
        """Check if edge exists in network."""
        return any(
            e.parent.id == edge[0] and e.child.id == edge[1]
            for e in self.bayesian_network.edges
        )

    def generate_control_level_summary(self) -> Dict[str, Any]:
        """
        Generate summary of control level distribution.
        
        Returns:
            Dict containing control level statistics
        """
        summary = {level: [] for level in ControlLevel}
        
        for var_id, ctrl in self.variable_controls.items():
            summary[ctrl.control_scope.primary_level].append({
                'variable': var_id,
                'influence_weight': ctrl.control_scope.influence_weight,
                'requires_coordination': ctrl.control_scope.requires_coordination,
                'secondary_levels': [
                    level.name for level in ctrl.control_scope.secondary_levels
                ]
            })
            
        return {
            level.name: {
                'variables': variables,
                'count': len(variables)
            }
            for level, variables in summary.items()
        }