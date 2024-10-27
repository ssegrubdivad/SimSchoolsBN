## SimSchools BN Project
# API Reference

## QueryProcessor

The `QueryProcessor` class is responsible for handling various types of queries on the Bayesian Network.

### Methods

#### `__init__(self, model: BayesianNetwork)`

Initializes the QueryProcessor with a given Bayesian Network model.

#### `process_query(self, query_type: str, query_vars: List[str], evidence: Dict[str, Any], interventions: Dict[str, Any] = None) -> Dict[str, Any]`

Processes a query on the Bayesian Network.

- `query_type`: Type of query ('marginal', 'conditional', 'interventional', 'map', 'mpe')
- `query_vars`: List of variables to query
- `evidence`: Dictionary of evidence variables and their values
- `interventions`: Dictionary of intervention variables and their values (for interventional queries)

Returns a dictionary with query results.

#### `set_inference_algorithm(self, algorithm: str)`

Sets the inference algorithm to use.

- `algorithm`: Either "variable_elimination" or "junction_tree"

#### `temporal_query(self, query_vars: List[str], time_steps: int, evidence: Dict[str, Any] = None) -> Dict[str, List[float]]`

Performs a temporal query on a Dynamic Bayesian Network.

- `query_vars`: List of variables to query
- `time_steps`: Number of time steps to consider
- `evidence`: Dictionary of evidence variables and their values

Returns a dictionary with query results over time.

## JunctionTree

The `JunctionTree` class implements the Junction Tree algorithm for efficient exact inference in Bayesian Networks.

### Methods

#### `__init__(self, model: BayesianNetwork)`

Initializes the JunctionTree with a given Bayesian Network model.

#### `query(self, variables: List[str], evidence: Dict[str, Any] = None) -> Dict[str, np.ndarray]`

Performs a query using the Junction Tree algorithm.

- `variables`: List of variables to query
- `evidence`: Dictionary of evidence variables and their values

Returns a dictionary mapping variable names to their probability distributions.

#### `map_query(self, variables: List[str], evidence: Dict[str, Any] = None) -> Dict[str, Any]`

Performs a Maximum a Posteriori (MAP) query using the Junction Tree algorithm.

- `variables`: List of variables to query
- `evidence`: Dictionary of evidence variables and their values

Returns a dictionary mapping variable names to their MAP values.

#### `mpe_query(self, evidence: Dict[str, Any] = None) -> Dict[str, Any]`

Performs a Most Probable Explanation (MPE) query using the Junction Tree algorithm.

- `evidence`: Dictionary of evidence variables and their values

Returns a dictionary representing the most probable explanation.
