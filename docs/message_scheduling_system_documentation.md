## SimSchools BN Project
# Message Scheduling System Documentation

### Overview

The message scheduling system determines the optimal order of message computations in the Bayesian network inference process. It ensures both mathematical correctness and numerical stability while maintaining computational efficiency.

### Core Components

#### 1. Priority System

```python
class NodePriority(Enum):
    EVIDENCE = 1      # Highest priority: Evidence nodes
    LEAF = 2         # Simple leaf node messages
    MIXED = 3        # CLG messages requiring careful handling
    DISCRETE = 4     # Pure discrete messages
    CONTINUOUS = 5   # Pure continuous messages
    STANDARD = 6     # Default priority
```

**Priority Rationale**
- Evidence nodes first: Establishes firm probability constraints
- Leaf nodes next: Simple messages without dependencies
- Mixed nodes: Require balanced handling of discrete/continuous components
- Type-specific priorities: Manages numerical stability

#### 2. Message Dependencies

```python
@dataclass
class ScheduleEntry:
    source_id: str
    target_id: str
    priority: NodePriority
    dependencies: Set[Tuple[str, str]]
```

**Dependency Rules**
1. Direct Dependencies:
   - Messages from neighboring nodes
   - Evidence propagation paths
   - Query-relevant paths

2. Indirect Dependencies:
   - Transitive closure of direct dependencies
   - Cross-component influences
   - Numerical stability requirements

### Scheduling Algorithm

#### 1. Initialization Phase
```python
def create_schedule(query_nodes: Set[str], 
                   evidence_nodes: Set[str]) -> List[ScheduleEntry]:
    """
    Creates optimal message schedule:
    1. Initialize with evidence messages
    2. Add leaf node messages
    3. Process remaining messages by priority
    """
```

**Steps**:
1. Evidence Message Initialization
   ```python
   for node_id in evidence_nodes:
       add_evidence_messages(node_id)
   ```

2. Leaf Node Processing
   ```python
   for node in leaf_nodes:
       add_leaf_messages(node)
   ```

3. Dependency Resolution
   ```python
   while pending_messages:
       process_next_available_message()
   ```

#### 2. Optimization Phase

```python
def optimize_schedule(schedule: List[ScheduleEntry]) -> List[ScheduleEntry]:
    """
    Optimizes schedule for:
    1. Numerical stability
    2. Computational efficiency
    3. Memory usage
    """
```

**Optimization Criteria**:
1. Message Type Grouping
   - Group similar message types
   - Minimize type transitions
   - Maintain dependencies

2. Memory Management
   - Minimize peak memory usage
   - Clear messages when possible
   - Reuse computation space

3. Numerical Stability
   - Process stable computations first
   - Group similar-scale operations
   - Monitor error accumulation

### Validation System

#### 1. Schedule Validation
```python
def validate_schedule(schedule: List[ScheduleEntry]) -> bool:
    """
    Validates schedule properties:
    1. Dependency satisfaction
    2. Completeness
    3. Correctness
    """
```

**Validation Checks**:
1. Dependency Order
   ```python
   for message in schedule:
       assert all_dependencies_scheduled_before(message)
   ```

2. Completeness
   ```python
   assert all_required_messages_included(schedule)
   ```

3. Correctness
   ```python
   assert no_circular_dependencies(schedule)
   ```

#### 2. Runtime Validation

```python
def validate_runtime_execution(message: ScheduleEntry) -> bool:
    """
    Validates during execution:
    1. Resource availability
    2. Numerical stability
    3. Error bounds
    """
```

### Error Handling

#### 1. Schedule Creation Errors
```python
class SchedulingError(Exception):
    """Base class for scheduling errors."""
    pass

class DependencyError(SchedulingError):
    """Raised for dependency violations."""
    pass

class ResourceError(SchedulingError):
    """Raised for resource constraints."""
    pass
```

#### 2. Recovery Strategies
```python
def handle_scheduling_failure(error: SchedulingError) -> Optional[Schedule]:
    """
    Recovery strategies:
    1. Reschedule with relaxed constraints
    2. Split schedule into sub-schedules
    3. Modify priorities
    """
```

### Performance Considerations

#### 1. Time Complexity
- Schedule Creation: O(|E| log |V|)
  - |E| = number of edges
  - |V| = number of vertices
- Optimization: O(|M| log |M|)
  - |M| = number of messages

#### 2. Memory Usage
- Schedule Storage: O(|M|)
- Dependency Tracking: O(|M|²)
- Runtime State: O(|V|)

#### 3. Optimization Opportunities
```python
def optimize_memory_usage(schedule: Schedule) -> Schedule:
    """
    Memory optimization strategies:
    1. Message lifetime minimization
    2. Computation reuse
    3. Strategic message clearing
    """
```

### Usage Examples

#### 1. Basic Usage
```python
# Create scheduler
scheduler = MessageScheduler(factor_graph)

# Create schedule
schedule = scheduler.create_schedule(
    query_nodes={'X1', 'X2'},
    evidence_nodes={'E1'}
)

# Optimize schedule
optimized = scheduler.optimize_schedule(schedule)

# Execute schedule
for entry in optimized:
    compute_message(entry)
```

#### 2. Custom Scheduling
```python
# Define custom priorities
class CustomPriority(NodePriority):
    SPECIAL = 0  # Higher priority than EVIDENCE

# Create scheduler with custom settings
scheduler = MessageScheduler(
    factor_graph,
    priority_system=CustomPriority
)
```

### Integration Guidelines

#### 1. Factor Graph Integration
```python
class FactorGraph:
    def initialize_scheduler(self) -> MessageScheduler:
        """
        Initialize scheduler with:
        1. Graph structure
        2. Node types
        3. Custom priorities
        """
```

#### 2. Inference Engine Integration
```python
class InferenceEngine:
    def run_inference(self, query_nodes: Set[str]) -> None:
        """
        1. Create schedule
        2. Optimize schedule
        3. Execute messages
        4. Compute final results
        """
```

### Testing Requirements

#### 1. Unit Tests
```python
def test_schedule_creation():
    """
    Test:
    1. Priority enforcement
    2. Dependency satisfaction
    3. Completeness
    """

def test_schedule_optimization():
    """
    Test:
    1. Numerical stability
    2. Memory efficiency
    3. Performance optimization
    """
```

#### 2. Integration Tests
```python
def test_inference_integration():
    """
    Test:
    1. Full inference process
    2. Evidence handling
    3. Query processing
    """
```

### Future Extensions

#### 1. Dynamic Scheduling
- Runtime priority adjustment
- Adaptive optimization
- Load balancing

#### 2. Parallel Scheduling
- Independent message identification
- Resource allocation
- Synchronization points

#### 3. Advanced Optimizations
- Cache-aware scheduling
- GPU-optimized ordering
- Distributed computation support

## Future Enhancements: Message Scheduling System

### Educational Model Specific Enhancements

#### 1. Educational Priority System
```python
class EducationalNodePriority(Enum):
    EVIDENCE = 1
    STUDENT_PERFORMANCE = 2    # Core educational outcomes
    RESOURCE_ALLOCATION = 3    # Resource dependencies
    DEMOGRAPHIC = 4           # Background variables
    STANDARD = 5
```

**Rationale**: Better alignment with educational variable importance and dependencies

#### 2. Scale-Aware Message Ordering
- Handle common educational scales:
  * Test scores (0-100)
  * Budgets (millions)
  * Ratios (0-1)
  * Student-teacher ratios
  * Standardized measures

### Implementation Notes
These enhancements should be considered when:
1. Educational model usage patterns demonstrate need
2. Scale-related numerical issues arise
3. Priority refinement shows measurable benefits

### Integration Requirements
Must maintain:
1. Current mathematical guarantees
2. Numerical stability
3. Existing validation framework

### Conclusion

The message scheduling system provides a robust foundation for efficient and numerically stable inference in mixed Bayesian networks. Its priority-based scheduling, comprehensive validation, and optimization capabilities ensure reliable results while maintaining computational efficiency.