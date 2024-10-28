# src/inference_engine/message_scheduling.py

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging
from enum import Enum
from dataclasses import dataclass
from queue import PriorityQueue

class NodePriority(Enum):
    """Priority levels for message scheduling."""
    EVIDENCE = 1      # Nodes with evidence
    LEAF = 2         # Leaf nodes
    MIXED = 3        # Nodes with both discrete and continuous components
    DISCRETE = 4     # Pure discrete nodes
    CONTINUOUS = 5   # Pure continuous nodes
    STANDARD = 6     # All other nodes

@dataclass
class ScheduleEntry:
    """Represents a scheduled message computation."""
    source_id: str
    target_id: str
    priority: NodePriority
    dependencies: Set[Tuple[str, str]]  # Set of (source, target) pairs this message depends on
    
    def __lt__(self, other: 'ScheduleEntry') -> bool:
        """Compare entries for priority queue."""
        return self.priority.value < other.priority.value

class MessageScheduler:
    """
    Controls the scheduling of message computations.
    Ensures optimal ordering while maintaining numerical stability.
    """
    def __init__(self, factor_graph: 'FactorGraph'):
        """
        Initialize scheduler with factor graph.
        
        Args:
            factor_graph: The factor graph to schedule messages for
        """
        self.graph = factor_graph
        self.logger = logging.getLogger(__name__)
        self.pending_messages: PriorityQueue[ScheduleEntry] = PriorityQueue()
        self.completed_messages: Set[Tuple[str, str]] = set()
        self.message_dependencies: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        self._initialize_dependencies()

    def _initialize_dependencies(self) -> None:
        """Initialize message dependencies based on graph structure."""
        for node in self.graph.nodes.values():
            for neighbor_id in node.neighbors:
                msg_key = (node.id, neighbor_id)
                dependencies = self._compute_message_dependencies(node.id, neighbor_id)
                self.message_dependencies[msg_key] = dependencies
                
                self.logger.debug(
                    f"Dependencies for message {node.id}->{neighbor_id}: {dependencies}"
                )

    def _compute_message_dependencies(self, source_id: str, 
                                    target_id: str) -> Set[Tuple[str, str]]:
        """
        Compute dependencies for a message.
        A message depends on all incoming messages to its source node except
        from the target node.
        """
        source_node = self.graph.nodes[source_id]
        dependencies = set()
        
        for neighbor_id in source_node.neighbors:
            if neighbor_id != target_id:
                dependencies.add((neighbor_id, source_id))
                
        return dependencies

    def create_schedule(self, query_nodes: Set[str], 
                       evidence_nodes: Set[str]) -> List[ScheduleEntry]:
        """
        Create a schedule for computing messages given query and evidence.
        
        Args:
            query_nodes: Set of nodes we're querying
            evidence_nodes: Set of nodes with evidence
            
        Returns:
            List of message computations in optimal order
        """
        self.logger.info("Creating message schedule")
        self.logger.debug(f"Query nodes: {query_nodes}")
        self.logger.debug(f"Evidence nodes: {evidence_nodes}")
        
        # Reset state
        self.pending_messages = PriorityQueue()
        self.completed_messages.clear()
        
        # Initialize schedule with evidence messages
        self._initialize_schedule(query_nodes, evidence_nodes)
        
        # Process messages until done
        schedule = []
        while not self.pending_messages.empty():
            entry = self.pending_messages.get()
            
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(entry):
                # Put back in queue with reduced priority
                new_priority = NodePriority(min(entry.priority.value + 1, 
                                              NodePriority.STANDARD.value))
                self.pending_messages.put(ScheduleEntry(
                    entry.source_id,
                    entry.target_id,
                    new_priority,
                    entry.dependencies
                ))
                continue
                
            schedule.append(entry)
            self.completed_messages.add((entry.source_id, entry.target_id))
            
            # Add new messages that might now be computable
            self._add_dependent_messages(entry)
            
        self.logger.info(f"Created schedule with {len(schedule)} messages")
        return schedule

    def _initialize_schedule(self, query_nodes: Set[str], 
                           evidence_nodes: Set[str]) -> None:
        """Initialize schedule with messages from evidence and leaf nodes."""
        # Add messages from evidence nodes first
        for node_id in evidence_nodes:
            node = self.graph.nodes[node_id]
            for neighbor_id in node.neighbors:
                self._add_message_to_schedule(node_id, neighbor_id, NodePriority.EVIDENCE)
                
        # Add messages from leaf nodes (nodes with only one neighbor)
        for node in self.graph.nodes.values():
            if len(node.neighbors) == 1 and node.id not in evidence_nodes:
                neighbor_id = next(iter(node.neighbors))
                self._add_message_to_schedule(node.id, neighbor_id, NodePriority.LEAF)
                
        # Add messages required for query nodes
        for node_id in query_nodes:
            self._ensure_query_messages(node_id)

    def _add_message_to_schedule(self, source_id: str, target_id: str, 
                               priority: NodePriority) -> None:
        """Add a message to the schedule with given priority."""
        source_node = self.graph.nodes[source_id]
        msg_key = (source_id, target_id)
        
        # Adjust priority based on node type
        if isinstance(source_node, CLGNode):
            priority = min(priority, NodePriority.MIXED)
        elif isinstance(source_node, DiscreteNode):
            priority = min(priority, NodePriority.DISCRETE)
        elif isinstance(source_node, ContinuousNode):
            priority = min(priority, NodePriority.CONTINUOUS)
            
        self.pending_messages.put(ScheduleEntry(
            source_id,
            target_id,
            priority,
            self.message_dependencies[msg_key]
        ))
        
        self.logger.debug(
            f"Added message {source_id}->{target_id} to schedule with priority {priority}"
        )

    def _dependencies_satisfied(self, entry: ScheduleEntry) -> bool:
        """Check if all dependencies for a message are satisfied."""
        return all(dep in self.completed_messages for dep in entry.dependencies)

    def _add_dependent_messages(self, completed_entry: ScheduleEntry) -> None:
        """Add messages that depend on the completed message."""
        completed_key = (completed_entry.source_id, completed_entry.target_id)
        
        # Find messages that depend on the completed message
        for msg_key, dependencies in self.message_dependencies.items():
            if completed_key in dependencies and msg_key not in self.completed_messages:
                source_id, target_id = msg_key
                # Add with standard priority - will be adjusted in _add_message_to_schedule
                self._add_message_to_schedule(source_id, target_id, NodePriority.STANDARD)

    def _ensure_query_messages(self, query_node_id: str) -> None:
        """Ensure all messages needed for query node are in schedule."""
        node = self.graph.nodes[query_node_id]
        
        # Add messages to query node from all neighbors
        for neighbor_id in node.neighbors:
            if (neighbor_id, query_node_id) not in self.completed_messages:
                self._add_message_to_schedule(neighbor_id, query_node_id, NodePriority.STANDARD)

    def validate_schedule(self, schedule: List[ScheduleEntry]) -> bool:
        """
        Validate that a message schedule is correct and complete.
        
        Args:
            schedule: The message schedule to validate
            
        Returns:
            bool: Whether the schedule is valid
        """
        # Check that all dependencies are satisfied in order
        completed = set()
        for entry in schedule:
            if not all(dep in completed for dep in entry.dependencies):
                self.logger.error(
                    f"Invalid schedule: Message {entry.source_id}->{entry.target_id} "
                    f"scheduled before its dependencies"
                )
                return False
            completed.add((entry.source_id, entry.target_id))
            
        # Check that all required messages are included
        required_messages = set(self.message_dependencies.keys())
        scheduled_messages = {(entry.source_id, entry.target_id) for entry in schedule}
        
        missing_messages = required_messages - scheduled_messages
        if missing_messages:
            self.logger.error(f"Invalid schedule: Missing messages {missing_messages}")
            return False
            
        return True

    def optimize_schedule(self, schedule: List[ScheduleEntry]) -> List[ScheduleEntry]:
        """
        Optimize a valid schedule for better numerical stability.
        
        Args:
            schedule: Valid message schedule to optimize
            
        Returns:
            Optimized schedule
        """
        # Group messages by their numerical properties
        discrete_messages = []
        continuous_messages = []
        mixed_messages = []
        
        for entry in schedule:
            source_node = self.graph.nodes[entry.source_id]
            if isinstance(source_node, DiscreteNode):
                discrete_messages.append(entry)
            elif isinstance(source_node, ContinuousNode):
                continuous_messages.append(entry)
            else:
                mixed_messages.append(entry)
                
        # Reorder while maintaining dependencies
        optimized = []
        
        # Process discrete messages first (most numerically stable)
        optimized.extend(self._optimize_group(discrete_messages))
        
        # Then mixed messages
        optimized.extend(self._optimize_group(mixed_messages))
        
        # Finally continuous messages
        optimized.extend(self._optimize_group(continuous_messages))
        
        assert self.validate_schedule(optimized), "Optimization produced invalid schedule"
        return optimized

    def _optimize_group(self, entries: List[ScheduleEntry]) -> List[ScheduleEntry]:
        """Optimize ordering within a group of messages."""
        # Create dependency graph for this group
        dep_graph = defaultdict(set)
        for entry in entries:
            msg_key = (entry.source_id, entry.target_id)
            for dep in entry.dependencies:
                if dep in {(e.source_id, e.target_id) for e in entries}:
                    dep_graph[msg_key].add(dep)
                    
        # Topologically sort while prioritizing numerical stability
        ordered = []
        visited = set()
        
        def visit(msg_key: Tuple[str, str]):
            if msg_key in visited:
                return
            visited.add(msg_key)
            
            # Process dependencies first
            for dep in sorted(dep_graph[msg_key]):
                visit(dep)
                
            # Add this message
            entry = next(e for e in entries 
                       if (e.source_id, e.target_id) == msg_key)
            ordered.append(entry)
            
        # Process all messages
        for entry in entries:
            msg_key = (entry.source_id, entry.target_id)
            visit(msg_key)
            
        return ordered

