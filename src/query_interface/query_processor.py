# src/query_interface/query_processor.py

from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from typing import List, Dict, Any, Tuple
import logging
import numpy as np

from src.network_structure import BayesianNetwork

class QueryProcessor:
    def __init__(self, model: BayesianNetwork):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.inference_engine = None
        self._initialize_inference_engine()

    def _initialize_inference_engine(self):
        try:
            # Convert our custom BayesianNetwork to pgmpy's BayesianNetwork
            pgmpy_model = PgmpyBayesianNetwork()
            for node in self.model.nodes.values():
                pgmpy_model.add_node(node.id)
            for edge in self.model.edges:
                pgmpy_model.add_edge(edge.parent.id, edge.child.id)

            # Add placeholder CPDs
            for node in self.model.nodes.values():
                cpd = self._create_placeholder_cpd(node)
                pgmpy_model.add_cpds(cpd)

            self.inference_engine = VariableElimination(pgmpy_model)
            self.logger.info("Inference engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engine: {str(e)}")
            raise

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
    
    def process_query(self, query_type: str, query_vars: List[str], evidence: Dict[str, Any], interventions: Dict[str, Any] = None) -> Dict[str, List[float]]:
        if not self.inference_engine:
            raise ValueError("Inference engine not initialized. Unable to process query.")

        try:
            if query_type == 'marginal':
                return self._marginal_query(query_vars, evidence)
            elif query_type == 'conditional':
                return self._conditional_query(query_vars, evidence)
            elif query_type == 'interventional':
                return self._interventional_query(query_vars, evidence, interventions)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    def _optimize_query(self, query_vars: List[str]) -> List[str]:
        # Simple optimization: order variables by their number of values
        return sorted(query_vars, key=lambda v: len(self.model.get_cpds(v).values))

    def _marginal_query(self, query_vars: List[str], evidence: Dict[str, Any]) -> Dict[str, List[float]]:
        result = self.inference_engine.query(variables=query_vars, evidence=evidence)
        return {var: result[var].values.tolist() for var in query_vars}

    def _conditional_query(self, query_vars: List[str], evidence: Dict[str, Any]) -> Dict[str, List[float]]:
        # For now, conditional query is the same as marginal query
        return self._marginal_query(query_vars, evidence)

    def _interventional_query(self, query_vars: List[str], evidence: Dict[str, Any], interventions: Dict[str, Any]) -> Dict[str, List[float]]:
        # Implement interventional query logic here
        # For now, we'll just return a marginal query
        return self._marginal_query(query_vars, evidence)

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