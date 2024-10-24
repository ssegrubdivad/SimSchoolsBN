# Bayesian Network Structure File (.bns)
### Version: 1.2
## Description: [Optional description of the network]

## Metadata
META network_name [name]
META author [author name]
META date_created [YYYY-MM-DD]
META version [version number]

## Node Definitions
NODE [node_id] [node_name] [variable_type] [additional_specs]
  DESCRIPTION [Optional description of the node]
  ### For discrete variables:
  STATES [comma-separated list of possible states]
  ### For continuous variables:
  RANGE [min,max]  # Required for continuous variables
  TEMPORAL_INFO [Optional: specifications for temporal nodes]
  PLATE [Optional: plate notation for repeated structures]

## Valid variable_type values are:
###   discrete - For variables with enumerated states
###   continuous - For variables with continuous numeric values

## Edge Definitions
EDGE [parent_node_id] [child_node_id]
  WEIGHT [Optional: edge weight or strength]
  TYPE [Optional: type of relationship, e.g., "causal", "correlational"]

## Group Definitions (for hierarchical or nested structures)
GROUP [group_id] [group_name]
  NODES [comma-separated list of node_ids in this group]
  DESCRIPTION [Optional description of the group]

## Dynamic Bayesian Network Specifications (if applicable)
DBN_TIMESLICE
  INTRA_SLICE_EDGES
    EDGE [parent_node_id] [child_node_id]
  INTER_SLICE_EDGES
    EDGE [parent_node_id_t] [child_node_id_t+1]

## End of file marker
END_NETWORK