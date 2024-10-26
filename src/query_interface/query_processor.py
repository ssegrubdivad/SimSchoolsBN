# src/query_interface/query_processor.py

from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from typing import List, Dict, Any, Tuple
import logging
import numpy as np

from src.network_structure import BayesianNetwork
from src.inference_engine.junction_tree import JunctionTree

class QueryProcessor:
    """
    Processes queries for Bayesian Networks, ensuring strict data integrity.
    Never creates or modifies probability distributions - requires complete
    user specification before performing any inference.

    Note on Implementation:
    This class currently uses a basic implementation for MAP and MPE queries.
    While sufficient for current educational models, more complex or larger models
    in the future may benefit from specialized estimator methods.

    Future Consideration:
    If educational contexts prove to involve more complex or larger networks,
    consider implementing or integrating specialized estimator methods for
    improved performance and accuracy in MAP and MPE queries.
    """
    def __init__(self, model: BayesianNetwork):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.current_algorithm = "variable_elimination"
        self.variable_elimination = None
        self.junction_tree = None
        self.initialized = False
        # Validate distributions before attempting to initialize inference engines
        validation_result = self._validate_distributions()
        if not validation_result['is_valid']:
            self.logger.error(f"Distribution validation failed: {validation_result['message']}")
            self.validation_errors = validation_result
        else:
            self.missing_distributions = self._get_missing_distributions()
            if not self.missing_distributions:
                self._initialize_inference_engines()

    def _validate_distributions(self) -> Dict[str, Any]:
        """
        Validates all distributions in the network to ensure they are properly specified
        and compatible with inference operations.
        
        Returns:
            Dict containing:
                is_valid (bool): Whether all distributions are valid
                message (str): Description of validation result
                errors (List[Dict]): List of specific validation errors
        """
        errors = []
        
        # First pass: check for missing distributions
        missing_distributions = []
        for node_id, node in self.model.nodes.items():
            self.logger.debug(f"Validating distribution for node {node_id}")
            if node.distribution is None:
                errors.append({
                    'node': node_id,
                    'error_type': 'missing_distribution',
                    'message': f"Node {node_id} has no probability distribution specified"
                })
                continue
                
            try:
                params = node.distribution.get_parameters()
                self.logger.debug(f"Node {node_id} parameters: {params}")
            except Exception as e:
                self.logger.error(f"Error getting parameters for node {node_id}: {str(e)}")
                errors.append({
                    'node': node_id,
                    'error_type': 'parameter_error',
                    'message': f"Error getting parameters for node {node_id}: {str(e)}"
                })
        
        if missing_distributions:
            return {
                'is_valid': False,
                'message': "Distribution validation failed with the following errors:\n" + 
                          "\n".join([f"- Node {node_id} has no probability distribution specified" 
                                    for node_id in missing_distributions]),
                'errors': [{'node': node_id, 'error_type': 'missing_distribution'} 
                          for node_id in missing_distributions],
                'type': 'missing_distributions'  # Add this to help client distinguish error types
            }
        
        # Second pass: validate distribution parameters
        for node_id, node in self.model.nodes.items():
            try:
                params = node.distribution.get_parameters()
                
                if node.variable_type == 'discrete':
                    if not node.states:
                        errors.append({
                            'node': node_id,
                            'error_type': 'missing_states',
                            'message': f"- Discrete node {node_id} has no states defined"
                        })
                    
                elif node.variable_type == 'continuous':
                    self.logger.debug(f"\nValidating continuous node: {node_id}")
                    dist_type = getattr(node.distribution, 'distribution_type', None)
                    self.logger.debug(f"Distribution type: {dist_type}")
                    
                    params = node.distribution.get_parameters()
                    self.logger.debug(f"Full parameters from distribution: {params}")
                    
                    if dist_type == 'gaussian':
                        required_params = ['mean', 'variance']
                    elif dist_type == 'truncated_gaussian':
                        required_params = ['mean', 'variance', 'lower', 'upper']
                    elif dist_type == 'clg':
                        required_params = ['mean_base', 'coefficients', 'variance', 'continuous_parents']
                    else:
                        self.logger.debug(f"Unknown distribution type: {dist_type}")
                        errors.append({
                            'node': node_id,
                            'error_type': 'unknown_distribution',
                            'message': f"- Node {node_id} has unknown or unspecified distribution type"
                        })
                        continue
                    
                    distribution_params = params.get('parameters', {}).get((), {})
                    self.logger.debug(f"Extracted distribution parameters: {distribution_params}")
                    
                    for param in required_params:
                        if param not in distribution_params:
                            self.logger.debug(f"Missing required parameter: {param}")
                            errors.append({
                                'node': node_id,
                                'error_type': 'missing_parameter',
                                'message': f"- Node {node_id} missing required parameter: {param}"
                            })
                
            except Exception as e:
                errors.append({
                    'node': node_id,
                    'error_type': 'validation_error',
                    'message': f"- Error validating distribution for node {node_id}: {str(e)}"
                })

        if errors:
            error_messages = "\n".join([error['message'] for error in errors])
            return {
                'is_valid': False,
                'message': f"Distribution validation failed with the following errors:\n{error_messages}",
                'errors': errors,
                'type': 'parameter_validation'  # Add this to help client distinguish error types
            }
        
        return {
            'is_valid': True,
            'message': "All distributions validated successfully",
            'errors': [],
            'type': 'success'
        }

    def _get_missing_distributions(self) -> List[str]:
        """
        Identify nodes that lack probability distributions.
        Returns a list of node IDs that need distributions specified.
        """
        return [node.id for node in self.model.nodes.values() 
                if node.distribution is None]

    def _initialize_inference_engines(self) -> bool:
        """
        Initialize inference engines only if all distributions are specified.
        Never creates default distributions.
        """
        missing = self._get_missing_distributions()
        if missing:
            missing_nodes = ", ".join(missing)
            self.logger.warning(f"Cannot initialize inference engines - missing distributions for: {missing_nodes}")
            return False

        try:
            pgmpy_model = self._convert_to_pgmpy_model()
            self.variable_elimination = VariableElimination(pgmpy_model)
            self.junction_tree = JunctionTree(self.model)
            self.initialized = True
            self.logger.info("Inference engines initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engines: {str(e)}")
            return False

    def _convert_to_pgmpy_model(self):
        pgmpy_model = PgmpyBayesianNetwork()
        for node in self.model.nodes.values():
            pgmpy_model.add_node(node.id)
        for edge in self.model.edges:
            pgmpy_model.add_edge(edge.parent.id, edge.child.id)
        for node in self.model.nodes.values():
            cpd = self._create_placeholder_cpd(node)
            pgmpy_model.add_cpds(cpd)
        return pgmpy_model

    def _create_placeholder_cpd(self, node):
        variable_card = len(node.states)
        parent_cards = [len(parent.states) for parent in node.parents]
        
        if node.distribution:
            # If the node has a distribution, use it to create the CPD
            values = node.distribution.get_parameters()['table']
            if isinstance(values, dict):
                # Convert dict to numpy array
                values = np.array(list(values.values())).T
        else:
            # If no distribution is available, create a uniform distribution
            if parent_cards:
                values = np.ones([variable_card, np.prod(parent_cards, dtype=int)]) / variable_card
            else:
                values = np.ones([variable_card, 1]) / variable_card
        
        evidence = [parent.id for parent in node.parents]
        
        return TabularCPD(
            variable=node.id,
            variable_card=variable_card,
            values=values,
            evidence=evidence,
            evidence_card=parent_cards
        )
    
    def process_query(self, query_type: str, query_vars: List[str], 
                 evidence: Dict[str, Any], 
                 interventions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query, ensuring all required distributions are present and valid.
        Raises ValueError if distributions are missing.
        
        Args:
            query_type: Type of query ('marginal', 'conditional', 'interventional', 'map', 'mpe')
            query_vars: List of variables to query
            evidence: Dictionary of evidence variables and their values
            interventions: Optional dictionary of intervention variables and their values

        Returns:
            Dict[str, Any]: Query results

        Raises:
            ValueError: If distributions are missing or query type is invalid
        """
        # Check if we have validation errors
        if hasattr(self, 'validation_errors'):
            raise ValueError(self.validation_errors['message'])

        # Check for missing distributions
        missing = self._get_missing_distributions()
        if missing:
            missing_nodes = ", ".join(missing)
            raise ValueError(
                f"Cannot process query - missing probability distributions for nodes: "
                f"{missing_nodes}. Please upload complete CPT specifications first."
            )

        # Initialize inference engines if needed
        if not self.initialized and not self._initialize_inference_engines():
            raise ValueError(
                "Failed to initialize inference engines. Please ensure all "
                "probability distributions are properly specified."
            )

        # Validate inference engine selection
        if self.current_algorithm == "variable_elimination":
            inference_engine = self.variable_elimination
        elif self.current_algorithm == "junction_tree":
            inference_engine = self.junction_tree
        else:
            raise ValueError(f"Invalid inference algorithm: {self.current_algorithm}")

        # Process query based on type
        try:
            if query_type == 'marginal':
                return self._marginal_query(inference_engine, query_vars, evidence)
            elif query_type == 'conditional':
                return self._conditional_query(inference_engine, query_vars, evidence)
            elif query_type == 'interventional':
                return self._interventional_query(inference_engine, query_vars, evidence, interventions)
            elif query_type == 'map':
                return self._map_query(inference_engine, query_vars, evidence)
            elif query_type == 'mpe':
                return self._mpe_query(inference_engine, evidence)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    def _map_query(self, query_vars: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a Maximum a Posteriori (MAP) query using Junction Tree
        """
        return self.junction_tree.map_query(query_vars, evidence)

    def _mpe_query(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a Most Probable Explanation (MPE) query using Junction Tree
        """
        return self.junction_tree.mpe_query(evidence)

    def temporal_query(self, query_vars: List[str], time_steps: int, evidence: Dict[str, Any] = None) -> Dict[str, List[float]]:
        if not isinstance(self.model, DynamicBayesianNetwork):
            raise ValueError("Temporal queries are only supported for Dynamic Bayesian Networks")
        
        results = {}
        for t in range(time_steps):
            time_slice_evidence = self._get_time_slice_evidence(evidence, t)
            time_slice_query_vars = [f"{var}_{t}" for var in query_vars]
            time_slice_result = self._marginal_query(time_slice_query_vars, time_slice_evidence)
            for var, probs in time_slice_result.items():
                if var not in results:
                    results[var] = []
                results[var].append(probs)
        
        return results

    def _get_time_slice_evidence(self, evidence: Dict[str, Any], t: int) -> Dict[str, Any]:
        if evidence is None:
            return {}
        return {f"{var}_{t}": value for var, value in evidence.items()}

    def _optimize_query(self, query_vars: List[str]) -> List[str]:
        # Simple optimization: order variables by their number of values
        return sorted(query_vars, key=lambda v: len(self.model.get_cpds(v).values))

    def _marginal_query(self, inference_engine, query_vars: List[str], evidence: Dict[str, Any]) -> Dict[str, List[float]]:
        result = inference_engine.query(variables=query_vars, evidence=evidence)
        return {var: result[var].tolist() for var in query_vars}

    # TODO: Implement method to switch between inference algorithms
    # def set_inference_algorithm(self, algorithm: str):
    #     if algorithm == "variable_elimination":
    #         self.inference_engine = VariableElimination(self._convert_to_pgmpy_model())
    #     elif algorithm == "belief_propagation":
    #         self.inference_engine = BeliefPropagation(self._convert_to_pgmpy_model())
    #     else:
    #         raise ValueError(f"Unsupported inference algorithm: {algorithm}")


    def _conditional_query(self, query_vars: List[str], evidence: Dict[str, Any]) -> Dict[str, List[float]]:
        return self._marginal_query(query_vars, evidence)

    def _interventional_query(self, query_vars: List[str], evidence: Dict[str, Any], interventions: Dict[str, Any]) -> Dict[str, List[float]]:
        # Combine evidence and interventions
        combined_evidence = {**evidence, **interventions}
        return self._marginal_query(query_vars, combined_evidence)

    def _perform_intervention(self, interventions: Dict[str, Any]) -> BayesianNetwork:
        intervened_model = self.model.copy()
        
        for node, value in interventions.items():
            # Remove all incoming edges to the intervened node
            for parent in list(intervened_model.get_parents(node)):
                intervened_model.remove_edge(parent, node)
            
            # Set the CPD of the intervened node to a deterministic distribution
            cpd = TabularCPD(node, 2, [[0, 1] if value else [1, 0]])
            intervened_model.add_cpds(cpd)
        
        return intervened_model

    def set_inference_algorithm(self, algorithm: str):
        if algorithm not in ["variable_elimination", "junction_tree"]:
            raise ValueError(f"Unsupported inference algorithm: {algorithm}")
        self.current_algorithm = algorithm
        self.logger.info(f"Inference algorithm set to: {algorithm}")