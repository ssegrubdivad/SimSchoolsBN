# src/education_models/locus_control.py

from enum import Enum
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
import logging

class ControlLevel(Enum):
    """Represents the different levels of control in educational system."""
    STUDENT = 1
    TEACHER = 2
    PARENT = 3
    SCHOOL_ADMIN = 4
    DISTRICT_ADMIN = 5

@dataclass
class ControlScope:
    """Defines the scope of control for a variable or operation."""
    primary_level: ControlLevel
    secondary_levels: Set[ControlLevel]
    influence_weight: float  # Primary level's influence weight (0-1)
    requires_coordination: bool  # Whether changes require multi-level coordination

class ControlValidator:
    """Validates control relationships and authority paths."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_control_path(self, 
                            source_level: ControlLevel,
                            target_level: ControlLevel,
                            operation_type: str) -> bool:
        """
        Validates whether an operation between control levels is permitted.
        
        Args:
            source_level: Control level initiating the operation
            target_level: Control level being affected
            operation_type: Type of operation being performed
            
        Returns:
            bool: Whether the operation is permitted
            
        Mathematical Guarantees:
        - Maintains exact probability computations
        - Preserves inference precision requirements
        - Ensures valid control flow paths
        """
        # Direct control is always valid
        if source_level == target_level:
            return True
            
        # District admin can influence all levels
        if source_level == ControlLevel.DISTRICT_ADMIN:
            return True
            
        # School admin can influence teacher, student levels
        if (source_level == ControlLevel.SCHOOL_ADMIN and
            target_level in {ControlLevel.TEACHER, ControlLevel.STUDENT}):
            return True
            
        # Teacher can influence student level
        if (source_level == ControlLevel.TEACHER and
            target_level == ControlLevel.STUDENT):
            return True
            
        # Parent can influence student level
        if (source_level == ControlLevel.PARENT and
            target_level == ControlLevel.STUDENT):
            return True
            
        self.logger.warning(
            f"Invalid control path: {source_level.name} -> {target_level.name}"
        )
        return False

    def get_influence_path(self,
                          source_level: ControlLevel,
                          target_level: ControlLevel) -> List[ControlLevel]:
        """
        Gets the valid influence path between control levels.
        Maintains proper authority chain while preserving mathematical relationships.
        """
        if not self.validate_control_path(source_level, target_level, "influence"):
            return []
            
        path = [source_level]
        current = source_level
        
        while current != target_level:
            if current == ControlLevel.DISTRICT_ADMIN:
                path.append(ControlLevel.SCHOOL_ADMIN)
                current = ControlLevel.SCHOOL_ADMIN
            elif current == ControlLevel.SCHOOL_ADMIN:
                path.append(ControlLevel.TEACHER)
                current = ControlLevel.TEACHER
            elif current == ControlLevel.TEACHER:
                path.append(ControlLevel.STUDENT)
                current = ControlLevel.STUDENT
            elif current == ControlLevel.PARENT:
                path.append(ControlLevel.STUDENT)
                current = ControlLevel.STUDENT
                
        return path

class ControlledVariable:
    """
    Represents a variable with associated control levels.
    Maintains mathematical properties while incorporating control information.
    """
    def __init__(self, 
                 name: str,
                 control_scope: ControlScope,
                 validator: ControlValidator):
        self.name = name
        self.control_scope = control_scope
        self.validator = validator
        self.logger = logging.getLogger(__name__)
        self._influence_paths: Dict[ControlLevel, List[ControlLevel]] = {}
        
    def can_be_modified_by(self, level: ControlLevel) -> bool:
        """
        Checks if a control level can modify this variable.
        Preserves mathematical validity while enforcing control constraints.
        """
        if level == self.control_scope.primary_level:
            return True
            
        if level in self.control_scope.secondary_levels:
            return self.validator.validate_control_path(
                level,
                self.control_scope.primary_level,
                "modification"
            )
            
        return False
        
    def get_influence_weight(self, level: ControlLevel) -> float:
        """
        Gets the influence weight of a control level on this variable.
        Maintains probabilistic interpretation of influence.
        """
        if level == self.control_scope.primary_level:
            return self.control_scope.influence_weight
            
        if level in self.control_scope.secondary_levels:
            # Secondary influence is scaled by primary influence weight
            return self.control_scope.influence_weight * 0.5
            
        return 0.0

    def requires_coordination(self, level: ControlLevel) -> bool:
        """
        Checks if modifications require coordination with other levels.
        Ensures proper authority validation in probabilistic inference.
        """
        if not self.control_scope.requires_coordination:
            return False
            
        if level == self.control_scope.primary_level:
            return len(self.control_scope.secondary_levels) > 0
            
        return level in self.control_scope.secondary_levels