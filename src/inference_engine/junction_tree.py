# src/inference_engine/junction_tree.py

import numpy as np
from typing import Dict, List, Any
from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

from src.network_structure.bayesian_network import BayesianNetwork
from src.network_structure.node import Node

class JunctionTree:
    def __init__(self, model: BayesianNetwork):
        self.model = model
        self.pgmpy_model = self._convert_to_pgmpy_model()
        self.belief_propagation = BeliefPropagation(self.pgmpy_model)

    def _convert_to_pgmpy_model(self) -> PgmpyBayesianNetwork:
        pgmpy_model = PgmpyBayesianNetwork()
        for node in self.model.nodes.values():
            pgmpy_model.add_node(node.id)
        for edge in self.model.edges:
            pgmpy_model.add_edge(edge.parent.id, edge.child.id)
        for node in self.model.nodes.values():
            cpd = self._create_cpd(node)
            pgmpy_model.add_cpds(cpd)
        return pgmpy_model

    def _create_cpd(self, node: Node) -> TabularCPD:
        variable_card = len(node.states)
        parent_cards = [len(parent.states) for parent in node.parents]
        
        if node.distribution:
            # If the node has a distribution, use it to create the CPD
            values = node.distribution.get_parameters()['probabilities']
        else:
            # If no distribution is available, create a uniform distribution
            if parent_cards:
                values = np.ones([variable_card, np.prod(parent_cards)]) / variable_card
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

    def query(self, variables: List[str], evidence: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """
        Perform inference using the Junction Tree algorithm.

        Args:
            variables (List[str]): List of variable names to query.
            evidence (Dict[str, Any], optional): Evidence to condition on.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping variable names to their probability distributions.
        """
        result = self.belief_propagation.query(variables, evidence=evidence)
        return {var: result[var].values for var in variables}

    def map_query(self, variables: List[str], evidence: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a Maximum a Posteriori (MAP) query using the Junction Tree algorithm.

        Args:
            variables (List[str]): List of variable names to query.
            evidence (Dict[str, Any], optional): Evidence to condition on.

        Returns:
            Dict[str, Any]: A dictionary mapping variable names to their MAP values.
        """
        result = self.belief_propagation.map_query(variables, evidence=evidence)
        return result

    def mpe_query(self, evidence: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a Most Probable Explanation (MPE) query using the Junction Tree algorithm.

        Args:
            evidence (Dict[str, Any], optional): Evidence to condition on.

        Returns:
            Dict[str, Any]: A dictionary representing the most probable explanation.
        """
        non_evidence_vars = [node for node in self.model.nodes if node not in evidence]
        return self.map_query(non_evidence_vars, evidence)