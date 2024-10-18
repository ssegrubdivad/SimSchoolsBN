# tests/test_query_interface/test_query_processor.py

import pytest
from src.query_interface.query_processor import QueryProcessor
from src.network_structure import BayesianNetwork, Node, Edge

@pytest.fixture
def sample_network():
    network = BayesianNetwork("Test Network")
    node_a = Node("A", "Node A", "discrete")
    node_b = Node("B", "Node B", "discrete")
    node_c = Node("C", "Node C", "discrete")
    network.add_node(node_a)
    network.add_node(node_b)
    network.add_node(node_c)
    network.add_edge("A", "B")
    network.add_edge("A", "C")
    return network

@pytest.fixture
def query_processor(sample_network):
    return QueryProcessor(sample_network)

def test_marginal_query(query_processor):
    result = query_processor.process_query("marginal", ["B"], {})
    assert "B" in result
    assert isinstance(result["B"], list)

def test_conditional_query(query_processor):
    result = query_processor.process_query("conditional", ["B"], {"A": "value1"})
    assert "B" in result
    assert isinstance(result["B"], list)

def test_interventional_query(query_processor):
    result = query_processor.process_query("interventional", ["C"], {"A": "value1"}, {"B": "value2"})
    assert "C" in result
    assert isinstance(result["C"], list)

def test_map_query(query_processor):
    result = query_processor.process_query("map", ["B", "C"], {"A": "value1"})
    assert isinstance(result, dict)
    assert "B" in result
    assert "C" in result

def test_mpe_query(query_processor):
    result = query_processor.process_query("mpe", [], {"A": "value1"})
    assert isinstance(result, dict)
    assert "A" in result
    assert "B" in result
    assert "C" in result

def test_temporal_query(query_processor):
    with pytest.raises(ValueError):
        # Should raise an error because our sample network is not a Dynamic Bayesian Network
        query_processor.temporal_query(["B"], 3, {"A": "value1"})

def test_invalid_query_type(query_processor):
    with pytest.raises(ValueError):
        query_processor.process_query("invalid_type", ["B"], {})