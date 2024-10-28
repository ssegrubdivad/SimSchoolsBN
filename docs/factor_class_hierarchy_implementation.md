## SimSchools BN Project
# Factor Classes Implementation Documentation

### Overview

This document describes the implementation of factor classes in the SimSchools BN project, which form the mathematical foundation for exact inference in mixed Bayesian networks containing both discrete and continuous variables. The implementation strictly enforces mathematical rigor without compromising for computational convenience.

### Key Design Principles

1. **Explicit Specification**
   - All parameters must be explicitly specified
   - No default values or automatic parameter generation
   - No silent type conversions or approximations

2. **Mathematical Rigor**
   - Exact computation requirements
   - Strict validation of mathematical properties
   - Precise numerical tolerances (e.g., 1e-10 for probability sums)

3. **Error Handling**
   - Clear, specific error messages
   - Complete validation before operations
   - No silent failure modes

### Factor Class Hierarchy

#### Base Class: Factor
```python
class Factor(ABC):
    def __init__(self, name: str, variables: List[str])
    def validate(self) -> ValidationResult
    def get_value(self, assignment: Dict[str, Union[float, str]]) -> float
    def multiply(self, other: Factor) -> Factor
    def marginalize(self, variables: List[str]) -> Factor
```

The abstract base class defines the interface all factors must implement.

### Factor Types

#### 1. Discrete Factor
**Purpose**: Represents discrete probability distributions

**Mathematical Properties**:
- Finite state space: X ∈ {x₁, x₂, ..., xₙ}
- Probability axioms: ∀x: P(X = x) ≥ 0, ∑P(X = x) = 1
- Complete specification required

**Validation Requirements**:
```python
def validate(self) -> ValidationResult:
    # Checks:
    # 1. Complete probability table
    # 2. Exact probability sum (tolerance: 1e-10)
    # 3. No missing states or probabilities
```

**Operations**:
- Multiplication: Standard discrete factor multiplication
- Marginalization: Exact summation over states

#### 2. Gaussian Factor
**Purpose**: Represents univariate Gaussian distributions

**Mathematical Properties**:
- PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
- Domain: x ∈ ℝ
- Parameters: μ (mean), σ² (variance)

**Validation Requirements**:
```python
def validate(self) -> ValidationResult:
    # Checks:
    # 1. Both mean and variance specified
    # 2. Variance strictly positive
    # 3. No default parameters
```

**Operations**:
- Multiplication: Precision-weighted parameter combination
- Marginalization: Analytical integration

#### 3. Truncated Gaussian Factor
**Purpose**: Represents bounded Gaussian distributions

**Mathematical Properties**:
- PDF: f(x) = φ((x-μ)/σ) / (σ(Φ((b-μ)/σ) - Φ((a-μ)/σ)))
- Domain: x ∈ [a,b]
- Parameters: μ (mean), σ² (variance), a (lower), b (upper)

**Validation Requirements**:
```python
def validate(self) -> ValidationResult:
    # Checks:
    # 1. All parameters specified
    # 2. Variance strictly positive
    # 3. a < b (strict ordering)
    # 4. No default bounds
```

**Operations**:
- Multiplication: Intersection of bounds, precision-weighted parameters
- Marginalization: Analytical integration within bounds

#### 4. CLG Factor
**Purpose**: Represents Conditional Linear Gaussian distributions

**Mathematical Properties**:
- PDF: f(x|y,z) = N(α(z) + β(z)ᵀy, σ²(z))
- Parameters for each discrete parent configuration z:
  - α(z): base mean
  - β(z): coefficient vector
  - σ²(z): variance

**Validation Requirements**:
```python
def validate(self) -> ValidationResult:
    # Checks:
    # 1. Complete parameter sets for all configurations
    # 2. Coefficient vector length matches parents
    # 3. Variance positive for all configurations
    # 4. No default parameters
```

**Operations**:
- Multiplication: Combines linear dependencies
- Marginalization: Analytical for continuous, sum for discrete

#### 5. Multivariate Gaussian Factor
**Purpose**: Represents multivariate Gaussian distributions

**Mathematical Properties**:
- PDF: f(x) = (2π)^(-k/2)|Σ|^(-1/2)exp(-(x-μ)ᵀΣ^(-1)(x-μ)/2)
- Parameters:
  - μ: mean vector
  - Σ: covariance matrix

**Validation Requirements**:
```python
def validate(self) -> ValidationResult:
    # Checks:
    # 1. Mean vector and covariance matrix dimensions match
    # 2. Covariance matrix symmetric
    # 3. Covariance matrix positive definite
    # 4. Complete specification
```

**Operations**:
- Multiplication: Precision-weighted combination
- Marginalization: Matrix operations for subsets

### Numerical Considerations

#### 1. Log-Space Computations
To maintain numerical stability:
```python
def get_log_value(self, assignment: Dict[str, Union[float, str]]) -> float:
    # Compute probabilities in log space
    # Handle underflow/overflow conditions
    # No approximations in conversion
```

#### 2. Matrix Operations
For multivariate distributions:
```python
# Use Cholesky decomposition for stability
L = np.linalg.cholesky(Σ)
solved = np.linalg.solve(L, diff)
```

### Error Handling

#### 1. Validation Results
```python
@dataclass
class ValidationResult:
    is_valid: bool
    message: str
    details: Optional[Dict] = None
```

#### 2. Operation Errors
All operations must:
1. Validate inputs before computation
2. Provide specific error messages
3. Never silently fail or approximate

### Usage Requirements

#### 1. Factor Creation
```python
# Example: Creating a Gaussian factor
factor = GaussianFactor("X1", ["X1"])
factor.parameters = {
    'mean': 0.0,
    'variance': 1.0
}
# Must validate before use
result = factor.validate()
if not result.is_valid:
    raise ValueError(result.message)
```

#### 2. Factor Operations
```python
# Multiplication
product = factor1.multiply(factor2)

# Marginalization
marginalized = factor.marginalize(["X1"])
```

### Testing Requirements

#### 1. Unit Tests
- Parameter validation
- Operation correctness
- Error handling
- Numerical precision

#### 2. Integration Tests
- Complex operations
- Multiple factor types
- Error propagation

### Future Extensions

When adding new factor types:
1. Inherit from Factor base class
2. Implement all abstract methods
3. Provide complete parameter validation
4. Document mathematical properties

### Performance Considerations

1. Use log-space computations when possible
2. Cache intermediate results in large operations
3. Use efficient matrix operations
4. Parallelize independent computations

### Notes on Implementation

1. **No Approximations**
   The implementation never compromises mathematical correctness for computational convenience.

2. **Complete Specifications**
   All parameters must be explicitly provided; no defaults are assumed.

3. **Error Propagation**
   Invalid operations fail early with clear error messages.

4. **Numerical Stability**
   Uses appropriate numerical methods for stable computation.

### Future Considerations

While the current implementation provides a complete and mathematically rigorous foundation, two potential areas for future optimization have been identified:

1. **Advanced Log-Space Operations**
   - Systematic log-space computation across all factor types
   - Direct log-space multiplication implementations
   - Enhanced numerical stability guarantees
   - Explicit underflow/overflow prevention

2. **Factor Chain Optimizations**
   - Operation sequence optimization
   - Memory management for large operation chains
   - Operation cost estimation and optimization
   - Efficient handling of long inference chains

These optimizations would maintain mathematical rigor while potentially improving performance for complex networks. They should be considered only when specific performance requirements are identified during practical application.

### Conclusion

This implementation provides a rigorous foundation for exact inference in mixed Bayesian networks. It prioritizes mathematical correctness and explicit specification over computational convenience, ensuring reliable results for educational modeling applications.