# SimSchools BN 
### Bayesian Network for Educational Simulations

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

#### Example Network Structure (.bns)
```
# Bayesian Network Structure File (.bns)
# Version: 1.1
# Description: Educational Finance Impact Network

META network_name EducationalFinanceNetwork
META author SimSchools_BN
META date_created 2024-10-23

# Continuous Variables
NODE AnnualBudget continuous
  DESCRIPTION Annual school budget in millions of dollars
  RANGE 1.0,10.0

NODE TeacherSalary continuous
  DESCRIPTION Average teacher salary in thousands of dollars
  RANGE 30.0,80.0

NODE StudentTeacherRatio continuous
  DESCRIPTION Number of students per teacher
  RANGE 10.0,30.0

# Discrete Variables
NODE ResourceAllocation discrete
  DESCRIPTION How the budget is primarily allocated
  STATES instruction,facilities,technology

NODE TeacherQuality discrete
  DESCRIPTION Overall teacher quality rating
  STATES low,medium,high

NODE StudentPerformance discrete
  DESCRIPTION Overall student academic performance
  STATES poor,average,good,excellent

# Edge Definitions
EDGE AnnualBudget ResourceAllocation
EDGE AnnualBudget TeacherSalary
EDGE AnnualBudget StudentTeacherRatio
EDGE TeacherSalary TeacherQuality
EDGE StudentTeacherRatio TeacherQuality
EDGE ResourceAllocation StudentPerformance
EDGE TeacherQuality StudentPerformance

# End of file marker
END_NETWORK
```

#### Example Conditional Probability Tables (.cpt)
```
# Conditional Probability Table File (.cpt)
# Version: 1.1
# Description: CPT for Educational Finance Network

# This network models:

# Continuous Variables:
# - AnnualBudget: School's annual budget (in millions)
# - TeacherSalary: Average teacher salary (in thousands)
# - StudentTeacherRatio: Students per teacher

# Discrete Variables:
# - ResourceAllocation: How budget is allocated (instruction/facilities/technology)
# - TeacherQuality: Quality rating (low/medium/high)
# - StudentPerformance: Academic performance (poor/average/good/excellent)

# The relationships modeled include:
# - Budget affects resource allocation, teacher salaries, and student-teacher ratio
# - Teacher salaries and student-teacher ratio influence teacher quality
# - Resource allocation and teacher quality influence student performance

# The network includes examples of:
# - Pure continuous distributions (AnnualBudget)
# - Continuous Linear Gaussian (CLG) distributions (TeacherSalary, StudentTeacherRatio)
# - Discrete CPTs with continuous parents (ResourceAllocation, TeacherQuality)
# - Discrete CPTs with discrete parents (StudentPerformance)

META network_name EducationalFinanceNetwork
META author SimSchools_BN
META date_created 2024-10-23

# Annual Budget (root node, continuous)
CPT AnnualBudget
  TYPE CONTINUOUS
  PARENTS
  DISTRIBUTION
    type = truncated_gaussian
    mean = 5.5
    variance = 2.25
    lower = 1.0
    upper = 10.0

# Resource Allocation (discrete with continuous parent)
CPT ResourceAllocation
  TYPE DISCRETE
  PARENTS AnnualBudget
  STATES instruction,facilities,technology
  TABLE
    # Lower budget (1-4 million) favors instruction
    (1.0,4.0) | 0.60, 0.30, 0.10
    # Medium budget (4-7 million) more balanced
    (4.0,7.0) | 0.45, 0.35, 0.20
    # Higher budget (7-10 million) enables more technology
    (7.0,10.0) | 0.40, 0.35, 0.25
  END_TABLE

# Teacher Salary (continuous with continuous parent)
CPT TeacherSalary
  TYPE CLG
  PARENTS AnnualBudget
  DISTRIBUTION
    continuous_parents = AnnualBudget
    mean_base = 35.0
    coefficients = [5.0]
    variance = 16.0

# Student-Teacher Ratio (continuous with continuous parent)
CPT StudentTeacherRatio
  TYPE CLG
  PARENTS AnnualBudget
  DISTRIBUTION
    continuous_parents = AnnualBudget
    mean_base = 28.0
    coefficients = [-1.5]
    variance = 4.0

# Teacher Quality (discrete with continuous parents)
CPT TeacherQuality
  TYPE DISCRETE
  PARENTS TeacherSalary StudentTeacherRatio
  STATES low,medium,high
  TABLE
    # Format: TeacherSalary range, StudentTeacherRatio range | probabilities
    # Low salary (30-45K)
    (30.0,45.0),(10.0,15.0) | 0.50, 0.40, 0.10
    (30.0,45.0),(15.0,20.0) | 0.60, 0.35, 0.05
    (30.0,45.0),(20.0,30.0) | 0.70, 0.25, 0.05
    # Medium salary (45-60K)
    (45.0,60.0),(10.0,15.0) | 0.30, 0.50, 0.20
    (45.0,60.0),(15.0,20.0) | 0.35, 0.45, 0.20
    (45.0,60.0),(20.0,30.0) | 0.45, 0.40, 0.15
    # High salary (60-80K)
    (60.0,80.0),(10.0,15.0) | 0.15, 0.45, 0.40
    (60.0,80.0),(15.0,20.0) | 0.20, 0.45, 0.35
    (60.0,80.0),(20.0,30.0) | 0.25, 0.50, 0.25
  END_TABLE

# Student Performance (discrete with discrete parents)
CPT StudentPerformance
  TYPE DISCRETE
  PARENTS ResourceAllocation TeacherQuality
  STATES poor,average,good,excellent
  TABLE
    instruction,low | 0.30, 0.40, 0.20, 0.10
    instruction,medium | 0.20, 0.35, 0.30, 0.15
    instruction,high | 0.10, 0.25, 0.40, 0.25
    facilities,low | 0.35, 0.40, 0.15, 0.10
    facilities,medium | 0.25, 0.40, 0.25, 0.10
    facilities,high | 0.15, 0.35, 0.35, 0.15
    technology,low | 0.25, 0.45, 0.20, 0.10
    technology,medium | 0.15, 0.40, 0.30, 0.15
    technology,high | 0.05, 0.30, 0.40, 0.25
  END_TABLE
  
END_CPT_FILE
```

## Project Structure
```
simschools_bn/
├── app.py                        # Main Flask application
├── src/
│   ├── inference_engine/         # Core inference components
│   ├── input_parsing/            # File parsing modules
│   ├── network_structure/        # Network components
│   ├── probability_distribution/ # Distribution implementations
│   └── visualization/            # Visualization components
├── templates/                    # Frontend templates
├── static/                       # Static files
├── tests/                        # Test suites
└── docs/                         # Documentation
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
