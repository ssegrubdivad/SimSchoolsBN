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
    Processes queries for Bayesian Networks in educational contexts.

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
        self.variable_elimination = None
        self.junction_tree = None
        self._initialize_inference_engines()

    def _initialize_inference_engine(self):
        try:
            self.variable_elimination = VariableElimination(self._convert_to_pgmpy_model())
            self.junction_tree = JunctionTree(self.model)
            self.logger.info("Inference engines initialized successfully")
            
            # TODO: Consider implementing BeliefPropagation as an alternative inference method
            # self.belief_propagation = BeliefPropagation(pgmpy_model)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engines: {str(e)}")
            raise

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
        # Create a placeholder CPD with uniform distribution
        variable_card = len(node.states) if node.states else 2
        parent_cards = [len(parent.states) if parent.states else 2 for parent in node.parents]
        
        # Create a 2D array of uniform probabilities
        if parent_cards:
            cpd_values = np.ones([variable_card, np.prod(parent_cards, dtype=int)]) / variable_card
        else:
            cpd_values = np.ones([variable_card, 1]) / variable_card
        
        evidence = [parent.id for parent in node.parents]
        
        return TabularCPD(
            variable=node.id,
            variable_card=variable_card,
            values=cpd_values,
            evidence=evidence,
            evidence_card=parent_cards
        )
    
    def process_query(self, query_type: str, query_vars: List[str], evidence: Dict[str, Any], interventions: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.variable_elimination or not self.junction_tree:
            raise ValueError("Inference engines not initialized. Unable to process query.")

        try:
            if query_type == 'marginal':
                return self._marginal_query(query_vars, evidence)
            elif query_type == 'conditional':
                return self._conditional_query(query_vars, evidence)
            elif query_type == 'interventional':
                return self._interventional_query(query_vars, evidence, interventions)
            elif query_type == 'map':
                return self._map_query(query_vars, evidence)
            elif query_type == 'mpe':
                return self._mpe_query(evidence)
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

    def _marginal_query(self, query_vars: List[str], evidence: Dict[str, Any]) -> Dict[str, List[float]]:
        result = self.junction_tree.query(variables=query_vars, evidence=evidence)
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