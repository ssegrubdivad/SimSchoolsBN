# tests/test_input_parsing/test_cpt_parser.py

import pytest
import os
from src.input_parsing import CPTParser
from src.probability_distribution import DiscreteDistribution, ContinuousDistribution, CLGDistribution

@pytest.fixture
def cpt_parser():
    return CPTParser()

@pytest.fixture
def sample_cpt_content():
    return """
    # Conditional Probability Table File (.cpt)
    META network_name TestNetwork
    META author John Doe

    CPT A NodeA
      TYPE DISCRETE
      PARENTS B C
      STATES true,false
      TABLE
        state1,state1 | 0.7,0.3
        state1,state2 | 0.6,0.4
        state2,state1 | 0.5,0.5
        state2,state2 | 0.2,0.8
    END_CPT

    CPT B NodeB
      TYPE CONTINUOUS
      PARENTS
      DISTRIBUTION
        type = gaussian
        mean = 0
        variance = 1
    END_CPT

    CPT C NodeC
      TYPE CLG
      PARENTS A B
      DISTRIBUTION
        continuous_parents = B
        mean_base = 10
        variance = 2
        coefficients = [1.5]
    END_CPT
    """

def test_parse_metadata(cpt_parser, sample_cpt_content, tmp_path):
    file_path = tmp_path / "test_cpt.cpt"
    file_path.write_text(sample_cpt_content)
    
    cpt_parser.parse_file(str(file_path))
    metadata = cpt_parser.get_metadata()
    
    assert metadata['network_name'] == 'TestNetwork'
    assert metadata['author'] == 'John Doe'

def test_parse_discrete_cpt(cpt_parser, sample_cpt_content, tmp_path):
    file_path = tmp_path / "test_cpt.cpt"
    file_path.write_text(sample_cpt_content)
    
    cpts = cpt_parser.parse_file(str(file_path))
    
    assert 'A' in cpts
    assert isinstance(cpts['A'], DiscreteDistribution)
    assert cpts['A'].variable == 'NodeA'
    assert cpts['A'].parents == ['B', 'C']
    assert cpts['A'].values == ['true', 'false']
    assert len(cpts['A'].probabilities) == 4

def test_parse_continuous_cpt(cpt_parser, sample_cpt_content, tmp_path):
    file_path = tmp_path / "test_cpt.cpt"
    file_path.write_text(sample_cpt_content)
    
    cpts = cpt_parser.parse_file(str(file_path))
    
    assert 'B' in cpts
    assert isinstance(cpts['B'], ContinuousDistribution)
    assert cpts['B'].variable == 'NodeB'
    assert cpts['B'].parents == []
    assert cpts['B'].distribution_type == 'gaussian'

def test_parse_clg_cpt(cpt_parser, sample_cpt_content, tmp_path):
    file_path = tmp_path / "test_cpt.cpt"
    file_path.write_text(sample_cpt_content)
    
    cpts = cpt_parser.parse_file(str(file_path))
    
    assert 'C' in cpts
    assert isinstance(cpts['C'], CLGDistribution)
    assert cpts['C'].variable == 'NodeC'
    assert cpts['C'].continuous_parents == ['B']
    assert cpts['C'].discrete_parents == ['A']

def test_file_not_found(cpt_parser):
    with pytest.raises(IOError):
        cpt_parser.parse_file("non_existent_file.cpt")

def test_invalid_cpt_structure(cpt_parser, tmp_path):
    invalid_content = """
    CPT A NodeA
      TYPE INVALID
    END_CPT
    """
    file_path = tmp_path / "invalid_cpt.cpt"
    file_path.write_text(invalid_content)
    
    with pytest.raises(ValueError):
        cpt_parser.parse_file(str(file_path))

def test_invalid_table_format(cpt_parser, tmp_path):
    invalid_content = """
    CPT A NodeA
      TYPE DISCRETE
      PARENTS B
      STATES true,false
      TABLE
        state1 | 0.7,0.3,0.0
    END_CPT
    """
    file_path = tmp_path / "invalid_cpt.cpt"
    file_path.write_text(invalid_content)
    
    with pytest.raises(ValueError):
        cpt_parser.parse_file(str(file_path))

def test_invalid_distribution_format(cpt_parser, tmp_path):
    invalid_content = """
    CPT B NodeB
      TYPE CONTINUOUS
      PARENTS
      DISTRIBUTION
        type = invalid_type
    END_CPT
    """
    file_path = tmp_path / "invalid_cpt.cpt"
    file_path.write_text(invalid_content)
    
    with pytest.raises(ValueError):
        cpt_parser.parse_file(str(file_path))