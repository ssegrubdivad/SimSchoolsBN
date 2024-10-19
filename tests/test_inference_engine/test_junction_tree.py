# tests/test_inference_engine/test_junction_tree.py

import pytest
import numpy as np
from src.inference_engine.junction_tree import JunctionTree
from src.network_structure.bayesian_network import BayesianNetwork
from src.network_structure.node import Node

@pytest.fixture
def simple_network():
    network = BayesianNetwork("Simple Network")
    a = Node("A", "Node A", "discrete")
    b = Node("B", "Node B", "discrete")
    c = Node("C", "Node C", "discrete")
    
    a.add_states(["T", "F"])
    b.add_states(["T", "F"])
    c.add_states(["T", "F"])
    
    network.add_node(a)
    network.add_node(b)
    network.add_node(c)
    network.add_edge("A", "B")
    network.add_edge("A", "C")
    
    # Set some dummy probabilities
    a.distribution.set_parameters({"probabilities": np.array([[0.6], [0.4]])})
    b.distribution.set_parameters({"probabilities": np.array([[0.7, 0.3], [0.2, 0.8]])})
    c.distribution.set_parameters({"probabilities": np.array([[0.5, 0.5], [0.1, 0.9]])})
    
    return network

def test_junction_tree_initialization(simple_network):
    jt = JunctionTree(simple_network)
    assert jt.model == simple_network
    assert jt.pgmpy_model is not None
    assert jt.belief_propagation is not None

def test_junction_tree_query(simple_network):
    jt = JunctionTree(simple_network)
    result = jt.query(["A"])
    assert "A" in result
    assert np.allclose(result["A"], [0.6, 0.4], atol=1e-5)

def test_junction_tree_map_query(simple_network):
    jt = JunctionTree(simple_network)
    result = jt.map_query(["A", "B"], evidence={"C": "T"})
    assert "A" in result and "B" in result
    assert result["A"] in ["T", "F"]
    assert result["B"] in ["T", "F"]

def test_junction_tree_mpe_query(simple_network):
    jt = JunctionTree(simple_network)
    result = jt.mpe_query(evidence={"A": "T"})
    assert "A" in result and "B" in result and "C" in result
    assert result["A"] == "T"
    assert result["B"] in ["T", "F"]
    assert result["C"] in ["T", "F"]