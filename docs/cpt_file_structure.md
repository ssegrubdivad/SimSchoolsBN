## SimSchools BN Project
# Conditional Probability Table File (.cpt)
### Version: 1.2
## Description: [Optional description of the CPT file]

## Metadata
META network_name [name]
META author [author name]
META date_created [YYYY-MM-DD]
META version [version number]

## CPT Definitions
CPT [node_id] [node_name]
  TYPE [DISCRETE | CONTINUOUS]
  PARENTS [comma-separated list of parent node ids]
  
  ### For Discrete Variables with No Parents
  STATES [comma-separated list of possible states]
  TABLE
    [prob_state_1], [prob_state_2], ...
  END_TABLE

  ### For Discrete Variables with Discrete Parents
  STATES [comma-separated list of possible states]
  TABLE
    [parent1_state], [parent2_state], ... | [prob_state_1], [prob_state_2], ...
  END_TABLE

  ### For Discrete Variables with Continuous Parents
  STATES [comma-separated list of possible states]
  TABLE
    #### Each continuous parent value must be specified as a range using parentheses
    ([min1],[max1]), ([min2],[max2]), ... | [prob_state_1], [prob_state_2], ...
    #### Example for one continuous parent:
    (1.0,4.0) | 0.60, 0.30, 0.10
    # Example for multiple continuous parents:
    (1.0,4.0), (30.0,40.0) | 0.60, 0.30, 0.10
    # Example mixing continuous and discrete parents:
    (1.0,4.0), high | 0.60, 0.30, 0.10
  END_TABLE

  ### For Continuous Variables (Root Nodes)
  DISTRIBUTION
    #### For Gaussian distribution:
    type = gaussian
    mean = [value]
    variance = [value]

    #### For truncated Gaussian distribution:
    type = truncated_gaussian
    mean = [value]
    variance = [value]
    lower = [lower bound]
    upper = [upper bound]

  ### For Continuous Variables with Parents (CLG Distribution)
  DISTRIBUTION
    type = clg
    continuous_parents = [comma-separated list of continuous parent nodes]
    mean_base = [base mean value]
    coefficients = [list of coefficients for continuous parents]
    variance = [variance value]
    #### Optional truncation:
    lower = [lower bound]
    upper = [upper bound]

  ### Optional: Temporal dependencies for Dynamic Bayesian Networks
  TEMPORAL_DEPENDENCY
    [specification of how probabilities change over time]
  
  ### Optional: Context-specific independence
  CONTEXT_SPECIFIC_INDEPENDENCE
    CONTEXT [context specification]
      [modified probability specification]
    END_CONTEXT

## General Rules:
### 1. All continuous ranges must be specified using parentheses: (min,max)
### 2. All numbers in continuous ranges should be specified as floats (e.g., 1.0 not 1)
### 3. Probabilities must sum to 1.0 for each combination of parent values
### 4. Comments can be included using #
### 5. Empty lines are ignored
### 6. For discrete variables with continuous parents, all ranges must be non-overlapping and must cover the entire domain of the parent variable

## Example with Multiple Types of Variables:
CPT Temperature continuous
  DISTRIBUTION
    type = truncated_gaussian
    mean = 20.0
    variance = 4.0
    lower = 15.0
    upper = 25.0

CPT Comfort discrete
  PARENTS Temperature
  STATES low,medium,high
  TABLE
    (15.0,18.0) | 0.70, 0.25, 0.05
    (18.0,22.0) | 0.10, 0.60, 0.30
    (22.0,25.0) | 0.05, 0.35, 0.60
  END_TABLE

### End of file marker
END_CPT_FILE
