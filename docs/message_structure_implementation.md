## SimSchools BN Project
# Message Passing Framework Documentation

### Overview

The message passing framework provides a rigorous implementation for exact inference in mixed Bayesian networks. It supports discrete, continuous, and hybrid (CLG) distributions while maintaining strict mathematical correctness without approximations.

### Message Types

#### 1. DiscreteMessage
**Purpose**: Represents discrete probability distributions in factor graphs.

**Mathematical Properties**:
```
P(X = x) where X ∈ {x₁, x₂, ..., xₙ}
∀x: P(X = x) ≥ 0
∑P(X = x) = 1
```

**Validation Requirements**:
- Complete probability table specification
- Exact probability sum (tolerance: 1e-10)
- All states explicitly defined

**Operations**:
```python
# Combination
P₃(X) = normalize(P₁(X) * P₂(X))

# Marginalization
P(Y) = ∑ₓ P(X, Y)
```

#### 2. GaussianMessage
**Purpose**: Represents univariate Gaussian distributions.

**Mathematical Properties**:
```
f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
x ∈ ℝ
σ² > 0
```

**Validation Requirements**:
- Mean explicitly specified
- Variance positive and explicitly specified
- Single variable only

**Operations**:
```python
# Combination (precision form for stability)
τᵢ = 1/σᵢ²  # precision
τ = τ₁ + τ₂
μ = (τ₁μ₁ + τ₂μ₂)/τ
σ² = 1/τ
```

#### 3. TruncatedGaussianMessage
**Purpose**: Represents bounded Gaussian distributions.

**Mathematical Properties**:
```
f(x) = φ((x-μ)/σ) / (σ(Φ((b-μ)/σ) - Φ((a-μ)/σ)))
x ∈ [a,b]
σ² > 0
a < b
```

**Validation Requirements**:
- Mean and variance explicitly specified
- Bounds explicitly specified and ordered
- Single variable only

**Operations**:
```python
# Combination
New bounds: [max(a₁,a₂), min(b₁,b₂)]
Parameters: Precision-weighted as in Gaussian case
```

#### 4. CLGMessage
**Purpose**: Represents Conditional Linear Gaussian distributions.

**Mathematical Properties**:
```
f(x|y,z) = N(α(z) + β(z)ᵀy, σ²(z))
where:
- x is continuous variable
- y are continuous parents
- z are discrete parents
```

**Validation Requirements**:
- Complete parameter sets for all discrete configurations
- Coefficient vector length matches continuous parents
- Variance positive for all configurations

**Operations**:
```python
# Combination
For each discrete configuration z:
  Combine Gaussian components using precision weighting

# Marginalization
Discrete: Sum over configurations
Continuous: Analytical integration
```

#### 5. MultivariateGaussianMessage
**Purpose**: Represents multivariate Gaussian distributions.

**Mathematical Properties**:
```
f(x) = (2π)^(-k/2)|Σ|^(-1/2)exp(-(x-μ)ᵀΣ^(-1)(x-μ)/2)
Σ must be positive definite
```

**Validation Requirements**:
- Mean vector dimension matches variables
- Covariance matrix symmetric and positive definite
- Complete specification of all parameters

**Operations**:
```python
# Combination (precision form)
Λᵢ = Σᵢ⁻¹  # precision matrices
Λ = Λ₁ + Λ₂
μ = Λ⁻¹(Λ₁μ₁ + Λ₂μ₂)
```

### Message Passing Protocol

#### 1. Message Direction
Messages can flow in two directions:
- `factor_to_variable`: From factor nodes to variable nodes
- `variable_to_factor`: From variable nodes to factor nodes

#### 2. Message Validation

All messages must be validated before use:
```python
validation = message.validate()
if not validation.is_valid:
    raise ValueError(validation.message)
```

#### 3. Message Operations

##### Combination
```python
# Combine compatible messages
result = message1.combine(message2)

# Messages must:
# 1. Have same variables
# 2. Be validated
# 3. Have compatible types
```

##### Marginalization
```python
# Marginalize out variables
result = message.marginalize(variables_to_remove)

# Must maintain:
# 1. Mathematical correctness
# 2. Distribution properties
# 3. Numerical stability
```

### Numerical Considerations

#### 1. Log-Space Computations
For numerical stability:
```python
# Instead of direct multiplication
p = p1 * p2

# Use log space
log_p = log_p1 + log_p2
```

#### 2. Matrix Operations
For multivariate distributions:
```python
# Use Cholesky decomposition
L = np.linalg.cholesky(covariance)
solved = np.linalg.solve(L, diff)
```

### Error Handling

#### 1. Validation Errors
```python
@dataclass
class ValidationResult:
    is_valid: bool
    message: str
    details: Optional[Dict] = None
```

#### 2. Operation Errors
All operations must:
- Validate inputs before computation
- Provide specific error messages
- Never silently fail or approximate

### Usage Examples

#### 1. Creating Messages
```python
# Discrete Message
discrete_msg = DiscreteMessage(
    source_id="F1",
    target_id="X1",
    variables=["X1"],
    direction="factor_to_variable",
    states={"X1": ["0", "1"]},
    probabilities={(("0",)): 0.3, (("1",)): 0.7}
)

# Gaussian Message
gaussian_msg = GaussianMessage(
    source_id="F2",
    target_id="X2",
    variables=["X2"],
    direction="factor_to_variable",
    mean=0.0,
    variance=1.0
)
```

#### 2. Message Operations
```python
# Combination
combined = msg1.combine(msg2)

# Marginalization
marginalized = msg.marginalize(["X1"])
```

### Implementation Requirements

#### 1. New Message Types
When adding new message types:
```python
class NewMessage(Message):
    def validate(self) -> ValidationResult:
        # Must implement strict validation
        pass
    
    def combine(self, other: Message) -> Message:
        # Must maintain mathematical properties
        pass
    
    def marginalize(self, variables: List[str]) -> Message:
        # Must preserve distribution characteristics
        pass
```

#### 2. Testing Requirements
- Unit tests for all message types
- Validation of mathematical properties
- Edge case handling
- Numerical stability tests

### Performance Considerations

1. **Memory Management**:
   - Efficient matrix storage
   - Smart caching of intermediate results
   - Clear cleanup of unused messages

2. **Computational Efficiency**:
   - Log-space computations
   - Cholesky decomposition for matrix operations
   - Efficient marginalization algorithms

### Future Extensions

When extending the framework:
1. Maintain mathematical rigor
2. Implement complete validation
3. Ensure numerical stability
4. Document mathematical properties

## Additional Considerations for Message Passing Framework

### Numerical Precision Guarantees

#### 1. Discrete Messages
- **Probability Sums**: |∑P(X = x) - 1| < 1e-10
- **Individual Probabilities**: Stored in double precision
- **Multiplication Chain Limits**: Maximum 1000 operations before renormalization to prevent underflow
```python
# Example of maintaining precision in long chains:
def safe_multiply_probabilities(probs: List[float]) -> float:
    log_prob = 0.0
    for p in probs:
        if p <= 0:
            return 0.0
        log_prob += np.log(p)
        # Renormalize if needed
        if log_prob < -700:  # Approaching float64 underflow limit
            return 0.0
    return np.exp(log_prob)
```

#### 2. Gaussian Messages
- **Variance Positivity**: σ² > 1e-14
- **Precision Matrix Condition Number**: Must be < 1e13 for numerical stability
```python
def check_precision_stability(precision_matrix: np.ndarray) -> bool:
    cond_num = np.linalg.cond(precision_matrix)
    return cond_num < 1e13
```

#### 3. CLG Messages
- **Coefficient Precision**: Maintained to relative error < 1e-12
- **Mixed Operation Stability**: Separate tracking for discrete and continuous components
```python
def validate_clg_coefficients(coefficients: np.ndarray, 
                            continuous_values: np.ndarray) -> bool:
    # Check numerical stability of linear combination
    max_coef = np.max(np.abs(coefficients))
    max_val = np.max(np.abs(continuous_values))
    return max_coef * max_val < 1e308  # Below float64 overflow
```

#### 4. Multivariate Gaussian Messages
- **Covariance Matrix Conditioning**: κ(Σ) < 1e13
- **Cholesky Decomposition Stability**: Diagonal elements > 1e-13
```python
def stable_covariance_inverse(covariance: np.ndarray) -> np.ndarray:
    """Compute numerically stable inverse of covariance matrix."""
    try:
        # Try Cholesky decomposition first
        L = np.linalg.cholesky(covariance)
        return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(covariance))))
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition if matrix is poorly conditioned
        eigvals, eigvecs = np.linalg.eigh(covariance)
        # Zero out small eigenvalues
        eigvals[eigvals < 1e-13] = 0
        return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
```

### Error Propagation Analysis

#### 1. Message Combination Chains
Error accumulation in message combinations follows:
```
ε_n ≤ ε_1 + ε_2 + ... + ε_n + (n-1)ε_machine
```
where ε_i is the error in the i-th message and ε_machine is machine epsilon.

**Mitigation Strategies**:
```python
class MessageChain:
    def __init__(self):
        self.error_bound = 0.0
        self.operation_count = 0
    
    def update_error_bound(self, new_operation_error: float):
        """Track accumulated error."""
        self.error_bound += new_operation_error + np.finfo(float).eps
        self.operation_count += 1
        
        if self.error_bound > ERROR_THRESHOLD:
            # Trigger renormalization or alternative computation path
            self.renormalize()
```

#### 2. Marginalization Error Propagation
- **Discrete**: Error bounded by sum of input errors
- **Continuous**: Error depends on integration method precision
- **Mixed**: Larger of discrete/continuous component errors

```python
def track_marginalization_error(message: Message, 
                              vars_to_marginalize: List[str]) -> float:
    """Estimate error introduced by marginalization."""
    if isinstance(message, DiscreteMessage):
        return sum_marginalization_error(message, vars_to_marginalize)
    elif isinstance(message, GaussianMessage):
        return gaussian_marginalization_error(message, vars_to_marginalize)
    elif isinstance(message, CLGMessage):
        return mixed_marginalization_error(message, vars_to_marginalize)
```

#### 3. Cumulative Error Bounds
For a sequence of n operations:
```python
def compute_cumulative_error_bound(operations: List[MessageOperation],
                                 initial_error: float) -> float:
    """
    Compute strict upper bound on cumulative error.
    
    Args:
        operations: List of message operations
        initial_error: Initial error in messages
        
    Returns:
        float: Upper bound on cumulative error
    """
    error_bound = initial_error
    for op in operations:
        if op.type == "combination":
            error_bound = combination_error_bound(error_bound)
        elif op.type == "marginalization":
            error_bound = marginalization_error_bound(error_bound)
            
        if error_bound > ERROR_THRESHOLD:
            raise NumericalInstabilityError(
                f"Error bound {error_bound} exceeds threshold"
            )
    
    return error_bound
```

### Message Normalization Requirements

#### 1. Discrete Message Normalization
Required after:
- Message combination
- Marginalization over evidence variables
- Long chains of operations

```python
def normalize_discrete_message(message: DiscreteMessage) -> DiscreteMessage:
    """
    Normalize discrete message probabilities.
    Maintains relative probabilities while ensuring sum to 1.
    """
    total = sum(message.probabilities.values())
    if abs(total - 1.0) > 1e-10:  # Only normalize if necessary
        message.probabilities = {
            k: v/total for k, v in message.probabilities.items()
        }
```

#### 2. CLG Message Normalization
Required to maintain consistency between:
- Discrete probability components
- Continuous density scaling
- Cross-component relationships

```python
def normalize_clg_message(message: CLGMessage) -> CLGMessage:
    """
    Normalize CLG message components while maintaining relationships.
    """
    # First normalize discrete components
    discrete_total = sum(message.discrete_probabilities.values())
    message.discrete_probabilities = {
        k: v/discrete_total 
        for k, v in message.discrete_probabilities.items()
    }
    
    # Adjust continuous components to maintain consistency
    for config in message.parameters:
        # Ensure proper scaling of continuous density
        mean_base = message.parameters[config]['mean_base']
        variance = message.parameters[config]['variance']
        scale_factor = compute_density_scale_factor(mean_base, variance)
        message.parameters[config]['scale'] = scale_factor
```

#### 3. Normalization in Mixed Operations
When combining different message types:
```python
def normalize_mixed_operation(discrete_part: Dict[str, float],
                            continuous_part: Dict[str, np.ndarray]) -> Tuple:
    """
    Normalize results of mixed discrete-continuous operations.
    Maintains proper relationship between components.
    """
    # Normalize discrete probabilities
    discrete_sum = sum(discrete_part.values())
    normalized_discrete = {k: v/discrete_sum for k, v in discrete_part.items()}
    
    # Adjust continuous densities to maintain consistency
    scale_factor = compute_mixed_scale_factor(discrete_sum)
    normalized_continuous = {
        k: scale_density(v, scale_factor) 
        for k, v in continuous_part.items()
    }
    
    return normalized_discrete, normalized_continuous
```

### Impact on Implementation

These considerations affect implementation in several ways:

1. **Operation Ordering**
   - Operations should be ordered to minimize error accumulation
   - Normalization points should be strategically placed

2. **Memory Management**
   - Intermediate results must maintain full precision
   - Garbage collection timing becomes critical for large networks

3. **Performance Implications**
   - Error tracking adds computational overhead
   - Strategic normalization may require additional operations

4. **Validation Requirements**
   - All numerical guarantees must be explicitly checked
   - Error bounds must be verified at each step

These additions provide concrete guarantees for numerical stability and error management, ensuring reliable results even in complex inference scenarios.


### Conclusion

This message passing framework provides a rigorous foundation for exact inference in mixed Bayesian networks. It prioritizes mathematical correctness and explicit specification over computational convenience, ensuring reliable results for educational modeling applications.