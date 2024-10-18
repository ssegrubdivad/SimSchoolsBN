# SimSchools BN

SimSchools BN is a comprehensive tool for creating, analyzing, and querying Bayesian Networks with a specific focus on simulating schools and education systems. While the core application can work with various types of Bayesian Networks, it is tailored for educational simulations.

## Features

- Input parsing for .bns and .cpt files
- Support for discrete and continuous variables relevant to educational contexts
- Multiple inference algorithms (Variable Elimination, MAP, MPE, MCMC, Junction Tree)
- Query interface for probabilistic reasoning in educational scenarios
- Visualization of network structure and query results
- Extensibility framework for custom node types and distributions specific to educational modeling
- Learning module for structure and parameter learning from educational data
- Sensitivity analysis capabilities for educational factors

## Basic Network Visualization

![Alt text](docs/images/graph1.png?raw=true "Basic Network Visualization Example")

## Project Structure

- `src/`: Source code for the application
- `tests/`: Unit and integration tests
- `docs/`: Documentation files
- `examples/`: Example Bayesian Networks for school and education system simulations
- `scripts/`: Utility scripts for development and deployment

## Implementation Notes

### Query Processing

The current implementation uses a basic approach for Maximum a Posteriori (MAP) and Most Probable Explanation (MPE) queries. This approach is suitable for the current scale and complexity of educational models in the system.

Future Consideration: If educational contexts prove to involve more complex or larger networks, we may need to implement or integrate more specialized estimator methods for improved performance and accuracy in MAP and MPE queries.

## Getting Started

[Instructions for setting up and running SimSchools BN will be added soon]

## Contributing

[Guidelines for contributing to the project will be added soon]

## License

[License information will be added soon]

## Future Directions

While currently focused on K-12 education, future versions may include capabilities for simulating universities and colleges.
