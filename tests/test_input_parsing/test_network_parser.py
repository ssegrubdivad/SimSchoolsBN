# tests/test_input_parsing/test_network_parser.py

import pytest
import os
from src.input_parsing import NetworkParser
from src.network_structure import BayesianNetwork, DynamicBayesianNetwork

@pytest.fixture
def network_parser():
    return NetworkParser()

@pytest.fixture
def sample_bns_content():
    return """
    # Bayesian Network Structure File (.bns)
    META network_name TestNetwork
    META author John Doe
    META date_created 2023-06-15

    NODE A NodeA discrete
      DESCRIPTION Node A description
      STATES state1,state2

    NODE B NodeB continuous
      DESCRIPTION Node B description
      RANGE 0,100

    EDGE A B

    DBN_TIMESLICE
      INTRA_SLICE_EDGES
        EDGE A B
      INTER_SLICE_EDGES
        EDGE A_t A_t+1
    """

def test_parse_metadata(network_parser, sample_bns_content, tmp_path):
    file_path = tmp_path / "test_network.bns"
    file_path.write_text(sample_bns_content)
    
    network_parser.parse_file(str(file_path))
    metadata = network_parser.get_metadata()
    
    assert metadata['network_name'] == 'TestNetwork'
    assert metadata['author'] == 'John Doe'
    assert metadata['date_created'] == '2023-06-15'

def test_parse_nodes(network_parser, sample_bns_content, tmp_path):
    file_path = tmp_path / "test_network.bns"
    file_path.write_text(sample_bns_content)
    
    network_parser.parse_file(str(file_path))
    nodes = network_parser.get_nodes()
    
    assert len(nodes) == 2
    assert 'A' in nodes and 'B' in nodes
    assert nodes['A']['name'] == 'NodeA' and nodes['A']['type'] == 'discrete'
    assert nodes['B']['name'] == 'NodeB' and nodes['B']['type'] == 'continuous'

def test_parse_edges(network_parser, sample_bns_content, tmp_path):
    file_path = tmp_path / "test_network.bns"
    file_path.write_text(sample_bns_content)
    
    network_parser.parse_file(str(file_path))
    edges = network_parser.get_edges()
    
    assert len(edges) == 1
    assert edges[0] == ('A', 'B')

def test_create_network(network_parser, sample_bns_content, tmp_path):
    file_path = tmp_path / "test_network.bns"
    file_path.write_text(sample_bns_content)
    
    network = network_parser.parse_file(str(file_path))
    
    assert isinstance(network, DynamicBayesianNetwork)
    assert network.name == 'TestNetwork'
    assert len(network.nodes) == 2
    assert len(network.edges) == 1

def test_parse_dbn_timeslices(network_parser, sample_bns_content, tmp_path):
    file_path = tmp_path / "test_network.bns"
    file_path.write_text(sample_bns_content)
    
    network = network_parser.parse_file(str(file_path))
    
    assert isinstance(network, DynamicBayesianNetwork)
    assert len(network.timeslices) == 1
    assert network.timeslices[0]['intra_slice_edges'] == [('A', 'B')]
    assert network.timeslices[0]['inter_slice_edges'] == [('A_t', 'A_t+1')]

def test_file_not_found(network_parser):
    with pytest.raises(IOError):
        network_parser.parse_file("non_existent_file.bns")

def test_invalid_network_structure(network_parser, tmp_path):
    invalid_content = """
    NODE A NodeA discrete
    EDGE A B
    """
    file_path = tmp_path / "invalid_network.bns"
    file_path.write_text(invalid_content)
    
    with pytest.raises(ValueError):
        network_parser.parse_file(str(file_path))