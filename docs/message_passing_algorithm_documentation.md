## SimSchools BN Project
# Message Passing Algorithm Documentation

### Overview

The message passing implementation provides an exact inference framework for mixed Bayesian networks, handling discrete, continuous, and hybrid distributions. The system maintains mathematical rigor without approximations, ensuring reliable results for educational modeling applications.

### Core Components

#### 1. Graph Structure

##### Node Types
```python
class NodeType(Enum):
    VARIABLE = "variable"
    FACTOR = "factor"
```

The factor graph consists of two types of nodes:
- **Variable Nodes**: Represent random variables (discrete or continuous)
- **Factor Nodes**: Represent probability distributions or potentials

##### Node Connections
- Strictly alternating: Variable nodes connect only to factor nodes and vice versa
- Each edge represents a probability dependency
- Messages flow in both directions along edges

#### 2. Message Flow Protocol

##### Direction Types
```python
message.direction ∈ {"variable_to_factor", "factor_to_factor"}
```

##### Variable to Factor Messages
- Product of all incoming messages except from target
- Evidence incorporation when variable is observed
- Validation at each step

##### Factor to Variable Messages
- Product of factor with incoming messages
- Marginalization over non-target variables
- Maintenance of numerical precision

### Message Computation

#### 1. Variable Node Messages
```python
def compute_message(self, target_id: str) -> Message:
    """
    For variable node V sending to factor node F:
    m_V→F = ∏_{F' ≠ F} m_F'→V
    """
```

**Properties**:
- Exactness maintained through careful product operations
- Evidence properly incorporated when present
- Numerical stability through log-space computations when needed

#### 2. Factor Node Messages
```python
def compute_message(self, target_id: str) -> Message:
    """
    For factor node F sending to variable node V:
    m_F→V = ∑_{X\V} (f(X) ∏_{V' ≠ V} m_V'→F)
    where X\V denotes all variables except V
    """
```

**Properties**:
- Exact marginalization without approximation
- Proper handling of mixed distributions
- Maintenance of distribution properties

### Belief Propagation Algorithm

#### 1. Main Loop Structure
```python
def run_belief_propagation(max_iterations: int, tolerance: float) -> bool:
    """
    Iterative process:
    1. Variable to factor messages
    2. Factor to variable messages
    3. Convergence check
    4. Repeat until convergence or max iterations
    """
```

#### 2. Convergence Criteria
For messages m_t at iteration t:
```
|m_t - m_{t-1}| < tolerance

where tolerance = 1e-6 for discrete messages
      tolerance = 1e-10 for continuous components
```

#### 3. Belief Computation
Final beliefs computed as:
```
B(X) = ∏_F m_F→X

where X is a variable node
      F ranges over all neighboring factors
```

### Message Operations

#### 1. Message Multiplication
```python
def combine_messages(msg1: Message, msg2: Message) -> Message:
    """
    Exact multiplication of messages:
    - Discrete × Discrete: Direct multiplication with normalization
    - Gaussian × Gaussian: Precision-weighted combination
    - CLG components: Separate handling of discrete/continuous parts
    """
```

#### 2. Message Marginalization
```python
def marginalize_message(msg: Message, vars_to_remove: List[str]) -> Message:
    """
    Exact marginalization:
    - Discrete: Sum over variables
    - Continuous: Analytical integration when possible
    - Mixed: Separate handling for each component
    """
```

### Evidence Handling

#### 1. Hard Evidence
For observed variable X with value x:
```python
def incorporate_evidence(message: Message, evidence_value: Any) -> Message:
    """
    Discrete: P(X = x') = δ(x' - x)
    Continuous: N(x, σ²) with σ² → 0
    """
```

#### 2. Soft Evidence
For uncertain observations:
```python
def incorporate_soft_evidence(message: Message, 
                            likelihood: Distribution) -> Message:
    """
    Multiply message by likelihood function:
    m'(x) = m(x)L(x)
    """
```

### Numerical Considerations

#### 1. Numerical Stability
```python
class MessagePassing:
    def _stable_message_product(self, messages: List[Message]) -> Message:
        """
        Use log-space for products:
        log(∏ mᵢ) = ∑ log(mᵢ)
        """
```

#### 2. Convergence Testing
```python
def _check_convergence(old_messages: Dict, new_messages: Dict) -> bool:
    """
    Separate tolerances for:
    - Discrete probabilities: 1e-6
    - Continuous parameters: 1e-10
    - Mixed distributions: strictest applicable tolerance
    """
```

### Error Handling

#### 1. Validation Points
- Message creation
- Message combination
- Message passing
- Convergence checking
- Belief computation

#### 2. Error Categories
```python
class MessagePassingError(Exception):
    """Base class for message passing errors."""
    pass

class NumericalInstabilityError(MessagePassingError):
    """Raised when numerical instability is detected."""
    pass

class ConvergenceError(MessagePassingError):
    """Raised when convergence issues occur."""
    pass
```

### Performance Considerations

#### 1. Memory Management
- Message caching for repeated computations
- Efficient storage of factor parameters
- Clear cleanup protocols

#### 2. Computational Efficiency
- Parallel message computation where possible
- Efficient matrix operations for continuous variables
- Strategic message scheduling

### Implementation Requirements

#### 1. New Message Types
When adding new message types:
```python
class NewMessageType(Message):
    def combine(self, other: Message) -> Message:
        # Must maintain exact computation
        pass
    
    def marginalize(self, variables: List[str]) -> Message:
        # Must preserve distribution properties
        pass
```

#### 2. Validation Requirements
All implementations must:
- Validate input messages
- Check numerical stability
- Verify operation results
- Maintain distribution properties

### Usage Example

```python
# Create factor graph
message_passing = MessagePassing()

# Add nodes
message_passing.add_node(VariableNode("X1", "discrete"))
message_passing.add_node(FactorNode("F1", factor))

# Add edges
message_passing.add_edge("X1", "F1")

# Set evidence
message_passing.get_node("X1").set_evidence("value1")

# Run inference
converged = message_passing.run_belief_propagation(
    max_iterations=100,
    tolerance=1e-6
)

# Get beliefs
beliefs = message_passing.get_beliefs()
```

### Testing Requirements

#### 1. Unit Tests
- Message computation correctness
- Convergence behavior
- Evidence handling
- Numerical stability

#### 2. Integration Tests
- End-to-end inference
- Complex network structures
- Mixed distribution handling

### Mathematical Guarantees

#### 1. Message Computation Exactness
- **Discrete Messages**
  - Exact to floating-point precision (≈2^-53 for IEEE 754 double)
  - Probability sums maintained to |∑p - 1| < 1e-10
  - No silent normalization or approximation
  
- **Gaussian Messages**
  - Guaranteed positive definiteness maintenance
  - Precision matrix condition number < 1e13
  - Cholesky decomposition stability guaranteed
  
- **CLG Messages**
  - Exact conditional relationships preserved
  - Linear coefficients maintained to relative error < 1e-12
  - Separate precision tracking for discrete and continuous components

#### 2. Convergence Properties
- **Tree-Structured Networks**
  - Local convergence guaranteed
  - Exact convergence in number of edges iterations
  - Message ordering independence

- **Convergence Criteria by Type**
  ```python
  # Discrete messages
  for all probabilities p:
      |p_t - p_{t-1}| < 1e-6
      
  # Gaussian messages
  ||μ_t - μ_{t-1}|| < 1e-10
  ||Σ_t - Σ_{t-1}||_F < 1e-10
  
  # CLG messages
  discrete_converged && continuous_converged
  ```

- **Convergence Tracking**
  - Per-message convergence monitoring
  - Global convergence state tracking
  - Early stopping on guaranteed convergence

#### 3. Precision Loss Prevention
- **Error Accumulation Tracking**
  ```python
  class PrecisionTracker:
      def accumulate_error(self, operation_error: float):
          """
          Track accumulated numerical error.
          Raises PrecisionLossError if bounds exceeded.
          """
          self.current_error += operation_error
          if self.current_error > self.threshold:
              self.trigger_rescaling()
  ```

- **Automatic Rescaling**
  - Log-space conversion for small probabilities
  - Matrix rescaling for poor conditioning
  - Explicit tracking of scaling operations

- **Distribution-Specific Bounds**
  ```python
  ERROR_BOUNDS = {
      'discrete': 1e-10,  # Probability sum tolerance
      'gaussian': {
          'variance_min': 1e-13,  # Minimum allowed variance
          'precision_max': 1e13   # Maximum condition number
      },
      'clg': {
          'coefficient_precision': 1e-12,  # Relative error in coefficients
          'discrete_tolerance': 1e-10      # Discrete component tolerance
      }
  }
  ```

### Edge Cases and Failure Handling

#### 1. Zero Probability Events
- **Detection and Handling**
  ```python
  def handle_zero_probability(self, probabilities: Dict[str, float]) -> Dict[str, float]:
      """
      Handle zero probabilities while maintaining valid distribution.
      
      Args:
          probabilities: Original probability distribution
          
      Returns:
          Modified distribution handling zero probabilities
          
      Raises:
          InvalidDistributionError: If all probabilities are zero
      """
      if all(p == 0 for p in probabilities.values()):
          raise InvalidDistributionError("All probabilities are zero")
          
      # Handle zeros while maintaining other relationships
      return self._redistribute_probability_mass(probabilities)
  ```

- **Marginalization Protection**
  ```python
  def safe_marginalize(self, message: Message, vars_to_remove: List[str]) -> Message:
      """
      Safe marginalization preventing invalid operations.
      
      Raises:
          MarginalizationError: If operation would create invalid distribution
      """
      if self._would_create_invalid_distribution(message, vars_to_remove):
          raise MarginalizationError("Marginalization would create invalid distribution")
  ```

- **Conditional Relationship Maintenance**
  - Explicit tracking of conditional dependencies
  - Validation of relationship preservation
  - Error bounds for conditional probabilities

#### 2. Ill-Conditioned Matrices
- **Condition Number Monitoring**
  ```python
  def check_matrix_conditioning(self, matrix: np.ndarray) -> bool:
      """
      Monitor numerical stability of matrix operations.
      
      Returns:
          bool: Whether matrix is well-conditioned
      """
      condition_number = np.linalg.cond(matrix)
      return condition_number < self.MAX_CONDITION_NUMBER
  ```

- **Stabilization Procedures**
  ```python
  def stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
      """
      Ensure covariance matrix stability.
      
      Uses eigendecomposition for poorly conditioned matrices.
      """
      if self.check_matrix_conditioning(covariance):
          return covariance
          
      return self._stabilize_via_eigendecomposition(covariance)
  ```

- **Operation Bounds**
  - Maximum condition number: 1e13
  - Minimum eigenvalue: 1e-13
  - Maximum scaling factor: 1e8

#### 3. Non-Convergence Scenarios
- **Oscillation Detection**
  ```python
  def detect_oscillation(self, message_history: List[Message]) -> bool:
      """
      Detect oscillating message patterns.
      
      Returns:
          bool: Whether messages are oscillating
      """
      if len(message_history) < 4:
          return False
          
      return self._check_oscillation_pattern(message_history)
  ```

- **Deterministic Relationship Handling**
  - Detection of deterministic dependencies
  - Special handling for zero-variance components
  - Preservation of exact relationships

- **Evidence Consistency**
  ```python
  def validate_evidence(self, evidence: Dict[str, Any]) -> bool:
      """
      Check consistency of provided evidence.
      
      Raises:
          InconsistentEvidenceError: If evidence violates model constraints
      """
      return self._check_evidence_consistency(evidence)
  ```

### Implementation Invariants

#### 1. Message Properties
- **Probability Constraints**
  ```python
  def validate_probability_distribution(self, probabilities: Dict) -> bool:
      """All probabilities must sum to 1.0 ± 1e-10"""
      return abs(sum(probabilities.values()) - 1.0) < 1e-10
  ```

- **Matrix Properties**
  ```python
  def validate_covariance_matrix(self, matrix: np.ndarray) -> bool:
      """
      Covariance matrices must be:
      1. Symmetric
      2. Positive definite
      3. Well-conditioned
      """
      return (
          self._is_symmetric(matrix) and
          self._is_positive_definite(matrix) and
          self._is_well_conditioned(matrix)
      )
  ```

- **Relationship Preservation**
  ```python
  def validate_conditional_relationships(self, message: Message) -> bool:
      """
      Verify:
      1. Conditional independence preserved
      2. Factorization properties maintained
      3. Markov properties satisfied
      """
      return all([
          self._check_conditional_independence(message),
          self._check_factorization(message),
          self._check_markov_properties(message)
      ])
  ```

#### 2. Operation Properties
- **Combination Invariants**
  - Exactness of probability calculations
  - Preservation of independence relationships
  - Maintenance of numerical stability

- **Marginalization Requirements**
  - Conservation of probability mass
  - Preservation of conditional relationships
  - Maintenance of distribution properties

- **Evidence Incorporation Rules**
  - Consistency with model structure
  - Preservation of valid distributions
  - Maintenance of numerical bounds

#### 3. System Properties
- **Graph Consistency**
  - Bipartite structure maintained
  - Valid connection patterns
  - Proper message flow paths

- **Message Scheduling**
  - Convergence guarantees
  - Proper information propagation
  - Deadlock prevention

- **Numerical Guarantees**
  - Error bound maintenance
  - Precision tracking
  - Stability assurance

### Summary

These guarantees and invariants ensure that the message passing implementation maintains mathematical rigor while handling the complexities of mixed Bayesian networks. The explicit handling of edge cases and failure modes provides robustness, while the implementation invariants ensure consistency and correctness throughout the inference process.

### Conclusion

This message passing implementation provides a mathematically rigorous framework for exact inference in mixed Bayesian networks. It maintains precision and correctness while handling the complexities of educational modeling applications.

### Appendix: Time Complexity

For a graph with:
- n variables
- m factors
- d maximum domain size
- k maximum factor size

The time complexity per iteration is:
O(m * d^k) for discrete networks
O(m * k^3) for Gaussian networks
O(m * d^k * k^3) for mixed networks

Space complexity is O(m * d^k) for message storage.