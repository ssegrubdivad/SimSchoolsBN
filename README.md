# SimSchools BN (Bayesian Network for Educational Simulations)

SimSchools BN is a comprehensive tool for creating, analyzing, and querying Bayesian Networks with a specific focus on simulating schools, colleges, universities and education systems. While the core application can work with various types of Bayesian Networks, it is tailored for educational simulations.

## Overview

SimSchools BN provides a comprehensive platform for modeling educational systems using Bayesian Networks, with specific focus on:
- Exact probabilistic inference without approximations
- Mixed networks combining discrete and continuous variables
- Educational control level integration
- Complex conditional relationships in school systems

### Key Features

1. **Network Creation and Management**
   - Upload and parse .bns (network structure) files
   - Upload and validate .cpt (conditional probability table) files
   - Mixed network support (discrete, continuous, CLG nodes)
   - Strict validation of network structure and parameters

2. **Distribution Types**
   - Discrete distributions with complete probability tables
   - Continuous Gaussian distributions
   - Truncated Gaussian distributions with bounds
   - Conditional Linear Gaussian (CLG) distributions
   - Multivariate Gaussian distributions

3. **Control Level Integration**
   - Hierarchical control structure (Student → District Admin)
   - Authority validation for operations
   - Control-aware evidence handling
   - Influence weight propagation

4. **Inference Capabilities**
   - Exact inference without approximations
   - Multiple query types (marginal, conditional, MAP, MPE)
   - Evidence incorporation with control validation
   - Complete error bound tracking

5. **Educational Model Support**
   - Specialized variables for educational contexts
   - Resource allocation modeling
   - Performance prediction
   - Hierarchical dependencies

6. **Visualization**
   - Interactive network visualization
   - Control-level based views
   - Authority path display
   - Influence weight visualization

## Basic Network Visualization

![Alt text](docs/images/graph1.png?raw=true "Basic Network Visualization Example")

## Installation

### Prerequisites
- Python 3.7+
- MySQL Database
- Node.js and npm (for frontend development)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/simschools_bn.git
cd simschools_bn
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
flask db upgrade
```

## Usage

### Running the Application
```bash
python app.py
```
Access the application at `http://localhost:8000`

### File Formats

#### Network Structure (.bns)
```
# Bayesian Network Structure File
META network_name ExampleNetwork
META author Author Name

NODE StudentPerformance continuous
  DESCRIPTION Student academic performance
  RANGE 0,100

NODE TeacherQuality discrete
  DESCRIPTION Teacher quality rating
  STATES low,medium,high

EDGE TeacherQuality StudentPerformance
```

#### Conditional Probability Tables (.cpt)
```
# CPT File
META network_name ExampleNetwork

CPT TeacherQuality
  TYPE DISCRETE
  STATES low,medium,high
  TABLE
    0.2,0.5,0.3

CPT StudentPerformance
  TYPE CLG
  PARENTS TeacherQuality
  DISTRIBUTION
    type = clg
    continuous_parents = []
    mean_base = 70.0
    coefficients = []
    variance = 100.0
```

## Project Structure
```
simschools_bn/
├── app.py              # Main Flask application
├── src/
│   ├── inference_engine/     # Core inference components
│   ├── input_parsing/       # File parsing modules
│   ├── network_structure/   # Network components
│   ├── probability_distribution/ # Distribution implementations
│   └── visualization/      # Visualization components
├── templates/          # Frontend templates
├── static/            # Static files
├── tests/             # Test suites
└── docs/              # Documentation
```

## Mathematical Guarantees

The system maintains strict mathematical guarantees:
- Exact probability computations without approximations
- Explicit error bound tracking throughout computations
- Numerical stability monitoring with condition number checks
- Complete validation of all probability distributions
- Precise control influence weight propagation

## Testing

Run the test suite:
```bash
pytest tests/
```

The test suite includes:
- Unit tests for all components
- Integration tests for end-to-end functionality
- Precision tests for mathematical guarantees
- Control hierarchy validation tests

## Documentation

Detailed documentation is available in the `docs/` directory:
- `algorithm_specifications.md`: Detailed algorithm descriptions
- `mathematical_foundation.md`: Mathematical basis and guarantees
- `api_reference.md`: API documentation
- `message_passing.md`: Message passing implementation details
- `locus_control.md`: Control level integration details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

Please ensure:
- All tests pass
- Mathematical guarantees are maintained
- Control level integration is preserved
- Documentation is updated
