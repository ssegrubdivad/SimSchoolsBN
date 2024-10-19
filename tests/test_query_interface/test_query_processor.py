# tests/test_query_interface/test_query_processor.py

import pytest
from src.query_interface.query_processor import QueryProcessor
from src.network_structure.bayesian_network import BayesianNetwork
from src.network_structure.node import Node

@pytest.fixture
def sample_network():
    network = BayesianNetwork("Test Network")
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
    a.distribution.set_parameters({"probabilities": [[0.6], [0.4]]})
    b.distribution.set_parameters({"probabilities": [[0.7, 0.3], [0.2, 0.8]]})
    c.distribution.set_parameters({"probabilities": [[0.5, 0.5], [0.1, 0.9]]})
    
    return network

@pytest.fixture
def query_processor(sample_network):
    return QueryProcessor(sample_network)

def test_marginal_query(query_processor):
    result = query_processor.process_query("marginal", ["B"], {})
    assert "B" in result
    assert len(result["B"]) == 2

def test_conditional_query(query_processor):
    result = query_processor.process_query("conditional", ["B"], {"A": "T"})
    assert "B" in result
    assert len(result["B"]) == 2

def test_interventional_query(query_processor):
    result = query_processor.process_query("interventional", ["C"], {"A": "T"}, {"B": "F"})
    assert "C" in result
    assert len(result["C"]) == 2

def test_map_query(query_processor):
    result = query_processor.process_query("map", ["B", "C"], {"A": "T"})
    assert "B" in result and "C" in result
    assert result["B"] in ["T", "F"]
    assert result["C"] in ["T", "F"]

def test_mpe_query(query_processor):
    result = query_processor.process_query("mpe", [], {"A": "T"})
    assert "A" in result and "B" in result and "C" in result
    assert result["A"] == "T"
    assert result["B"] in ["T", "F"]
    assert result["C"] in ["T", "F"]

def test_set_inference_algorithm(query_processor):
    query_processor.set_inference_algorithm("junction_tree")
    assert query_processor.current_algorithm == "junction_tree"
    
    with pytest.raises(ValueError):
        query_processor.set_inference_algorithm("invalid_algorithm")

def test_temporal_query(query_processor):
    with pytest.raises(ValueError):
        # Should raise an error because our sample network is not a Dynamic Bayesian Network
        query_processor.temporal_query(["B"], 3, {"A": "T"})

def test_invalid_query_type(query_processor):
    with pytest.raises(ValueError):
        query_processor.process_query("invalid_type", ["B"], {})