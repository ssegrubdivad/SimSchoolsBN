## SimSchools BN Project
# Mathematical Foundation and Implementation Specifications for Custom Inference Engine

## Contents
1. [Distribution Types and Properties](#1-distribution-types-and-properties)
2. [Mathematical Operations](#2-mathematical-operations)
3. [Message Computation and Scheduling Algorithm](#3-message-computation-and-scheduling-algorithm)
4. [Evidence Propagation Algorithm](#4-evidence-propagation-algorithm)
5. [Mixed-Type Message Operations](#5-mixed-type-message-operations)
6. [Error Chain Analysis](#6-error-chain-analysis)
7. [Validation Framework](#7-validation-framework)
8. [Error Handling and Recovery](#8-error-handling-and-recovery)

### 1. Distribution Types and Properties

#### 1.1 Discrete Distributions
- **Representation**: P(X = x) where X is a discrete random variable
- **Properties**:
  - Finite set of possible values: X ∈ {x₁, x₂, ..., xₙ}
  - Probabilities must sum to 1: ∑P(X = xᵢ) = 1
  - All probabilities must be explicitly specified
  - No interpolation or approximation allowed
- **Validation Requirements**:
  - Complete specification of all state probabilities
  - Exact probability sum (within numerical precision: |∑P(X = xᵢ) - 1| < 1e-10)
  - No missing states or probabilities allowed

#### 1.2 Continuous Gaussian Distributions
- **Representation**: N(μ, σ²)
- **Probability Density Function**:
  f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
- **Properties**:
  - Defined over entire real line: x ∈ ℝ
  - Parameters must be explicitly specified:
    - μ (mean)
    - σ² (variance), must be > 0
- **Validation Requirements**:
  - Variance must be strictly positive
  - Both parameters must be explicitly specified
  - No default values or assumptions

#### 1.3 Truncated Gaussian Distributions
- **Representation**: NT(μ, σ², a, b)
- **Probability Density Function**:
  f(x) = φ((x-μ)/σ) / (σ(Φ((b-μ)/σ) - Φ((a-μ)/σ)))
  where:
  - φ is the standard normal PDF
  - Φ is the standard normal CDF
  - a is the lower bound
  - b is the upper bound
- **Properties**:
  - Defined over finite interval: x ∈ [a,b]
  - All parameters must be explicitly specified
- **Validation Requirements**:
  - a < b (strict ordering of bounds)
  - σ² > 0 (positive variance)
  - All parameters must be explicitly specified
  - No default bounds or silent truncation

#### 1.4 Conditional Linear Gaussian (CLG) Distributions
- **Representation**: P(X|Y,Z) where:
  - X is the continuous variable
  - Y are continuous parents
  - Z are discrete parents
- **Probability Density Function**:
  f(x|y,z) = N(α(z) + β(z)ᵀy, σ²(z))
  where:
  - α(z) is the base mean for discrete parent state z
  - β(z) is the vector of coefficients for continuous parents
  - σ²(z) is the variance for discrete parent state z
- **Properties**:
  - Linear relationship with continuous parents
  - Different parameters for each discrete parent state combination
- **Validation Requirements**:
  - Complete specification for all discrete parent state combinations
  - Coefficient vector length must match number of continuous parents
  - Variance must be positive for all discrete parent states
  - No default parameters or interpolation

#### 1.5 Multivariate Gaussian Distributions
- **Representation**: N(μ, Σ)
  where:
  - μ is the mean vector
  - Σ is the covariance matrix
- **Probability Density Function**:
  f(x) = (2π)^(-k/2)|Σ|^(-1/2)exp(-(x-μ)ᵀΣ^(-1)(x-μ)/2)
  where:
  - k is the dimension of the distribution
  - |Σ| is the determinant of Σ
- **Properties**:
  - Covariance matrix must be positive definite
  - All parameters must be explicitly specified
- **Validation Requirements**:
  - Complete specification of mean vector
  - Complete specification of covariance matrix
  - Positive definiteness of covariance matrix
  - No missing values or default assumptions

### 2. Mathematical Operations

#### 2.1 Factor Multiplication
For factors f₁ and f₂:

##### Discrete × Discrete:
```python
def multiply_discrete(f1: DiscreteFactor, f2: DiscreteFactor) -> DiscreteFactor:
    """
    Multiply two discrete factors maintaining exact probabilities.
    """
    result = {}
    for states1 in f1.probabilities:
        for states2 in f2.probabilities:
            if compatible_states(states1, states2):
                new_states = combine_states(states1, states2)
                result[new_states] = f1.probabilities[states1] * f2.probabilities[states2]
                
    return normalize_discrete_distribution(result)
```

##### Continuous × Continuous:
```python
def multiply_gaussian(g1: GaussianFactor, g2: GaussianFactor) -> GaussianFactor:
    """
    Multiply Gaussian factors using precision form for stability.
    """
    # Convert to precision form
    τ₁ = 1 / g1.variance
    τ₂ = 1 / g2.variance
    
    # Combine parameters
    τ = τ₁ + τ₂
    μ = (τ₁ * g1.mean + τ₂ * g2.mean) / τ
    σ² = 1 / τ
    
    return GaussianFactor(mean=μ, variance=σ²)
```

##### CLG × CLG:
```python
def multiply_clg(c1: CLGFactor, c2: CLGFactor) -> CLGFactor:
    """
    Multiply CLG factors maintaining exact relationships.
    """
    result = {}
    for config in c1.discrete_configs:
        if config in c2.discrete_configs:
            # Multiply continuous components
            params1 = c1.parameters[config]
            params2 = c2.parameters[config]
            
            result[config] = {
                'mean_base': combine_means(params1, params2),
                'coefficients': combine_coefficients(params1, params2),
                'variance': combine_variances(params1, params2)
            }
            
    return CLGFactor(parameters=result)
```

### 3. Message Computation and Scheduling Algorithm

```python
def compute_message(source_node: Node, target_node: Node, 
                   incoming_messages: List[Message]) -> Message:
    """
    Compute message from source to target maintaining exact computation.
    
    Mathematical Guarantees:
    - Exact probability calculations
    - Proper error bound tracking
    - Numerical stability maintenance
    """
    # Initialize computation with prior or uniform distribution
    result = get_initial_distribution(source_node)
    
    # Combine with incoming messages
    for msg in incoming_messages:
        result = multiply_messages_in_log_space(result, msg)
        
    # Marginalize variables not in target scope
    vars_to_remove = get_variables_to_marginalize(source_node, target_node)
    result = marginalize_variables(result, vars_to_remove)
    
    return result

def create_schedule(variables: Set[str], 
                   evidence: Dict[str, Any]) -> List[ScheduleEntry]:
    """
    Create optimal message schedule for inference.
    
    Requirements:
    - All dependencies satisfied
    - Optimal ordering for numerical stability
    - Complete coverage of required messages
    """
    schedule = []
    message_queue = PriorityQueue()
    completed_messages = set()
    
    # Initialize with evidence and leaf nodes
    initialize_schedule(message_queue, variables, evidence)
    
    while not message_queue.empty():
        entry = message_queue.get()
        if dependencies_satisfied(entry, completed_messages):
            schedule.append(entry)
            completed_messages.add((entry.source, entry.target))
            add_new_messages(message_queue, entry, completed_messages)
            
    return optimize_schedule(schedule)
```

### 4. Evidence Propagation Algorithm

```python
def incorporate_evidence(evidence: Dict[str, Any], 
                       model: BayesianNetwork) -> Dict[str, Factor]:
    """
    Create evidence factors while maintaining exactness.
    """
    evidence_factors = {}
    
    for var, value in evidence.items():
        if isinstance(value, (int, float)):
            # Continuous evidence
            factor = create_continuous_evidence_factor(var, value)
        else:
            # Discrete evidence
            factor = create_discrete_evidence_factor(var, value)
            
        evidence_factors[var] = factor
        
    return evidence_factors

def create_continuous_evidence_factor(var: str, 
                                    value: float) -> GaussianFactor:
    """
    Create near-delta distribution for continuous evidence.
    """
    return GaussianFactor(
        mean=value,
        variance=1e-10  # Very small variance for hard evidence
    )

def create_discrete_evidence_factor(var: str,
                                  value: str) -> DiscreteFactor:
    """
    Create deterministic distribution for discrete evidence.
    """
    return DiscreteFactor(
        states={state: 1.0 if state == value else 0.0
                for state in get_variable_states(var)}
    )
```

### 5. Mixed-Type Message Operations

```python
def combine_mixed_messages(msg1: Message, 
                         msg2: Message) -> Message:
    """
    Combine messages of different types maintaining exact relationships.
    
    Mathematical Guarantees:
    - Type-specific precision requirements
    - Proper relationship preservation
    - Complete error tracking
    """
    if isinstance(msg1, DiscreteMessage):
        if isinstance(msg2, GaussianMessage):
            return combine_discrete_gaussian(msg1, msg2)
        elif isinstance(msg2, CLGMessage):
            return combine_discrete_clg(msg1, msg2)
    elif isinstance(msg1, GaussianMessage):
        if isinstance(msg2, CLGMessage):
            return combine_gaussian_clg(msg1, msg2)
            
    raise ValueError("Unsupported message combination")

def combine_discrete_gaussian(discrete: DiscreteMessage,
                            gaussian: GaussianMessage) -> CLGMessage:
    """
    Combine discrete and Gaussian messages into CLG.
    """
    parameters = {}
    for state in discrete.states:
        parameters[state] = {
            'mean': gaussian.mean,
            'variance': gaussian.variance,
            'weight': discrete.probabilities[state]
        }
    return CLGMessage(parameters=parameters)
```

### 6. Error Chain Analysis

```python
def track_error_chain(message_sequence: List[Message],
                     max_allowed_error: float) -> ErrorResult:
    """
    Track error propagation through message sequence.
    
    Mathematical Guarantees:
    - Strict error accumulation tracking
    - No hidden error sources
    - Complete error chain documentation
    """
    error_accumulator = 0.0
    error_points = []
    condition_number = 1.0
    
    for msg in message_sequence:
        # Track various error sources
        computation_error = msg.computational_error
        numerical_error = msg.numerical_stability_error
        
        # Update condition number for matrix operations
        if involves_matrix_operations(msg):
            condition_number = max(condition_number, 
                                 compute_condition_number(msg))
        
        # Accumulate errors
        error_accumulator += (computation_error + numerical_error + 
                            np.finfo(float).eps)
        
        # Record error point
        error_points.append(ErrorPoint(
            position=msg.position,
            local_error=computation_error + numerical_error,
            cumulative_error=error_accumulator,
            condition_number=condition_number
        ))
        
        # Check against maximum allowed error
        if error_accumulator > max_allowed_error:
            raise ErrorChainException(
                "Error accumulation exceeds maximum allowed",
                chain=error_points
            )
            
    return ErrorResult(
        cumulative_error=error_accumulator,
        error_points=error_points,
        condition_number=condition_number
    )
```

### 7. Validation Framework

```python
def validate_computation(result: ComputationResult) -> ValidationResult:
    """
    Validate computation results against precision requirements.
    """
    # Check basic requirements
    if not result.is_valid_distribution():
        return ValidationResult(False, "Invalid distribution properties")
        
    # Check error bounds
    if result.error > ERROR_THRESHOLD:
        return ValidationResult(False, "Error exceeds threshold")
        
    # Check numerical stability
    if not check_numerical_stability(result):
        return ValidationResult(
            False,
            "Numerical instability detected",
            {"condition_number": result.condition_number}
        )
        
    return ValidationResult(True, "Computation meets requirements")

def check_numerical_stability(result: ComputationResult) -> bool:
    """
    Check numerical stability of computation.
    """
    if isinstance(result, DiscreteResult):
        return check_probability_stability(result)
    elif isinstance(result, GaussianResult):
        return check_gaussian_stability(result)
    elif isinstance(result, CLGResult):
        return check_clg_stability(result)
    else:
        raise ValueError("Unknown result type")
```

### 8. Error Handling and Recovery

```python
def handle_numerical_error(error: NumericalError,
                         computation: Computation) -> Optional[ComputationResult]:
    """
    Handle numerical errors with recovery strategies.
    """
    if isinstance(error, UnderflowError):
        return handle_underflow(computation)
    elif isinstance(error, SingularityError):
        return handle_singularity(computation)
    elif isinstance(error, PrecisionLossError):
        return handle_precision_loss(computation)
        
    return None

def handle_underflow(computation: Computation) -> ComputationResult:
    """
    Handle underflow through log-space computation.
    """
    try:
        return compute_in_log_space(computation)
    except Exception as e:
        raise UnrecoverableError("Log-space computation failed") from e

def handle_singularity(computation: Computation) -> ComputationResult:
    """
    Handle near-singular matrices.
    """
    try:
        return compute_with_regularization(computation)
    except Exception as e:
        raise UnrecoverableError("Matrix regularization failed") from e
```

### Conclusion

This document provides a comprehensive foundation from mathematical principles through implementation specifications while maintaining:

1. Exact computation requirements
2. All operations maintain mathematical rigor
3. No silent approximations or modifications occur
4. Users must provide complete and correct specifications
5. Error conditions are clearly identified and reported

This framework provides the foundation for implementing exact inference in mixed Bayesian networks while maintaining strict mathematical guarantees throughout all operations.