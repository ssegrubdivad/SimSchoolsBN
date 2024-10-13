# Conditional Probability Table File (.cpt)
## Version: 1.1
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
  
  ### For Discrete Variables
  STATES [comma-separated list of possible states]
  TABLE
    [parent_state_1], [parent_state_2], ... | [prob_state_1], [prob_state_2], ...
    ...
  END_TABLE

  ### For Continuous Variables
  DISTRIBUTION [distribution_type]
    PARAMETERS
      [param1_name] = [param1_value or formula]
      [param2_name] = [param2_value or formula]
      ...
    END_PARAMETERS
    TRUNCATION [Optional: min,max]
  
  ### Optional: Temporal dependencies for Dynamic Bayesian Networks
  TEMPORAL_DEPENDENCY
    [specification of how probabilities change over time]
  
  ### Optional: Context-specific independence
  CONTEXT_SPECIFIC_INDEPENDENCE
    CONTEXT [context specification]
      [modified probability specification]
    END_CONTEXT
    ...

END_CPT

## Multiple CPTs can be defined in the same file
...

## End of file marker
END_CPT_FILE
