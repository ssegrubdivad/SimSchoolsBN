# Bayesian Network Structure File (.bns)
# Version: 1.1
# Description: [Optional description of the network]

# Metadata
META network_name [name]
META author [author name]
META date_created [YYYY-MM-DD]
META version [version number]

# Node Definitions
NODE [node_id] [node_name] [variable_type] [additional_specs]
  DESCRIPTION [Optional description of the node]
  STATES [comma-separated list of possible states for discrete variables]
  RANGE [min,max] [For continuous variables, optional]
  TEMPORAL_INFO [Optional: specifications for temporal nodes]
  PLATE [Optional: plate notation for repeated structures]

# Edge Definitions
EDGE [parent_node_id] [child_node_id]
  WEIGHT [Optional: edge weight or strength]
  TYPE [Optional: type of relationship, e.g., "causal", "correlational"]

# Group Definitions (for hierarchical or nested structures)
GROUP [group_id] [group_name]
  NODES [comma-separated list of node_ids in this group]
  DESCRIPTION [Optional description of the group]

# Dynamic Bayesian Network Specifications (if applicable)
DBN_TIMESLICE
  INTRA_SLICE_EDGES
    EDGE [parent_node_id] [child_node_id]
  INTER_SLICE_EDGES
    EDGE [parent_node_id_t] [child_node_id_t+1]

# End of file marker
END_NETWORK
