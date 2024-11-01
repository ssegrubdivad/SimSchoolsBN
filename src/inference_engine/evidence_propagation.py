# src/inference_engine/evidence_propagation.py

from typing import Dict, Set, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import logging
from enum import Enum

from ..education_models.locus_control import (
    ControlLevel,
    ControlScope,
    ControlValidator,
    ControlledVariable
)

class EvidenceType(Enum):
    """Types of evidence that can be incorporated."""
    HARD = "hard"       # Exact observations
    SOFT = "soft"       # Likelihood evidence
    VIRTUAL = "virtual" # Pseudo-observations

@dataclass
class Evidence:
    """
    Represents evidence for a variable.
    Maintains exact representation without approximation.
    """
    variable: str
    value: Any
    evidence_type: EvidenceType
    likelihood: Optional[Union[Dict[str, float], 'Distribution']] = None
    precision: Optional[float] = None  # For continuous evidence

    def validate(self) -> bool:
        """Validate evidence specification."""
        if self.evidence_type == EvidenceType.HARD:
            return self._validate_hard_evidence()
        elif self.evidence_type == EvidenceType.SOFT:
            return self._validate_soft_evidence()
        elif self.evidence_type == EvidenceType.VIRTUAL:
            return self._validate_virtual_evidence()
        return False

    def _validate_hard_evidence(self) -> bool:
        """Validate hard evidence."""
        if self.likelihood is not None:
            return False  # Hard evidence shouldn't have likelihood
        if isinstance(self.value, (int, float, str)):
            return True
        return False

    def _validate_soft_evidence(self) -> bool:
        """Validate soft evidence."""
        if self.likelihood is None:
            return False
        if isinstance(self.likelihood, dict):
            # Validate discrete likelihood
            return abs(sum(self.likelihood.values()) - 1.0) < 1e-10
        elif isinstance(self.likelihood, Distribution):
            # Validate continuous likelihood
            return True
        return False

    def _validate_virtual_evidence(self) -> bool:
        """Validate virtual evidence."""
        if self.likelihood is None:
            return False
        # Virtual evidence has similar requirements to soft evidence
        return self._validate_soft_evidence()

class EvidencePropagator:
    """
    Manages evidence incorporation and propagation.
    Ensures exact handling of evidence without approximation.
    """
    def __init__(self, model: 'BayesianNetwork'):
        """
        Initialize evidence propagator.
        
        Args:
            model: The Bayesian network model
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.evidence: Dict[str, Evidence] = {}
        self.processed_evidence: Set[str] = set()

    def add_evidence(self, evidence: Evidence) -> bool:
        """
        Add evidence for a variable.
        
        Args:
            evidence: Evidence to add
            
        Returns:
            bool: Whether evidence was successfully added
            
        Raises:
            ValueError: If evidence is invalid or inconsistent
        """
        if not evidence.validate():
            raise ValueError(f"Invalid evidence for variable {evidence.variable}")

        # Check consistency with existing evidence
        if evidence.variable in self.evidence:
            if not self._check_evidence_consistency(evidence):
                raise ValueError(
                    f"Inconsistent evidence for variable {evidence.variable}"
                )

        # Validate evidence type matches variable type
        variable_node = self.model.nodes[evidence.variable]
        if not self._validate_evidence_type(evidence, variable_node):
            raise ValueError(
                f"Evidence type incompatible with variable {evidence.variable}"
            )

        self.evidence[evidence.variable] = evidence
        self.logger.info(f"Added evidence for variable {evidence.variable}")
        return True

    def _check_evidence_consistency(self, new_evidence: Evidence) -> bool:
        """Check if new evidence is consistent with existing evidence."""
        existing = self.evidence[new_evidence.variable]
        
        if existing.evidence_type == EvidenceType.HARD and \
           new_evidence.evidence_type == EvidenceType.HARD:
            # Hard evidence must match exactly
            return existing.value == new_evidence.value
            
        elif existing.evidence_type == EvidenceType.SOFT and \
             new_evidence.evidence_type == EvidenceType.SOFT:
            # Soft evidence likelihoods must be compatible
            return self._check_likelihood_compatibility(
                existing.likelihood,
                new_evidence.likelihood
            )
            
        # Mixed evidence types need special handling
        return self._check_mixed_evidence_compatibility(existing, new_evidence)

    def _validate_evidence_type(self, evidence: Evidence, 
                              variable_node: 'Node') -> bool:
        """Validate evidence type matches variable type."""
        if isinstance(variable_node, DiscreteNode):
            if evidence.evidence_type == EvidenceType.HARD:
                return evidence.value in variable_node.states
            elif evidence.evidence_type == EvidenceType.SOFT:
                return isinstance(evidence.likelihood, dict) and \
                       all(state in variable_node.states 
                           for state in evidence.likelihood.keys())
                           
        elif isinstance(variable_node, ContinuousNode):
            if evidence.evidence_type == EvidenceType.HARD:
                return isinstance(evidence.value, (int, float))
            elif evidence.evidence_type == EvidenceType.SOFT:
                return isinstance(evidence.likelihood, (
                    GaussianDistribution,
                    TruncatedGaussianDistribution
                ))
                
        return True  # Other cases (e.g., CLG nodes) handled separately

    def create_evidence_factors(self) -> Dict[str, 'Factor']:
        """
        Create factors representing evidence.
        
        Returns:
            Dict mapping variable names to evidence factors
        """
        evidence_factors = {}
        
        for var, evidence in self.evidence.items():
            factor = self._create_evidence_factor(evidence)
            if factor is not None:
                evidence_factors[var] = factor
                
        return evidence_factors

    def _create_evidence_factor(self, evidence: Evidence) -> Optional['Factor']:
        """Create appropriate factor for evidence type."""
        if evidence.evidence_type == EvidenceType.HARD:
            return self._create_hard_evidence_factor(evidence)
        elif evidence.evidence_type == EvidenceType.SOFT:
            return self._create_soft_evidence_factor(evidence)
        elif evidence.evidence_type == EvidenceType.VIRTUAL:
            return self._create_virtual_evidence_factor(evidence)
        return None

    def _create_hard_evidence_factor(self, evidence: Evidence) -> 'Factor':
        """Create factor for hard evidence."""
        variable_node = self.model.nodes[evidence.variable]
        
        if isinstance(variable_node, DiscreteNode):
            # Create deterministic discrete factor
            probabilities = {
                state: 1.0 if state == evidence.value else 0.0
                for state in variable_node.states
            }
            return DiscreteFactor(
                variables=[evidence.variable],
                probabilities=probabilities
            )
            
        elif isinstance(variable_node, ContinuousNode):
            # Create near-delta continuous factor
            return GaussianFactor(
                variables=[evidence.variable],
                mean=float(evidence.value),
                variance=1e-10  # Very small variance for hard evidence
            )
            
        elif isinstance(variable_node, CLGNode):
            # Handle CLG nodes based on evidence type
            if isinstance(evidence.value, (int, float)):
                return self._create_clg_continuous_evidence(evidence)
            else:
                return self._create_clg_discrete_evidence(evidence)

    def _create_soft_evidence_factor(self, evidence: Evidence) -> 'Factor':
        """Create factor for soft evidence."""
        variable_node = self.model.nodes[evidence.variable]
        
        if isinstance(variable_node, DiscreteNode):
            # Create factor from likelihood
            return DiscreteFactor(
                variables=[evidence.variable],
                probabilities=evidence.likelihood
            )
            
        elif isinstance(variable_node, ContinuousNode):
            # Create continuous factor from likelihood distribution
            likelihood = evidence.likelihood
            if isinstance(likelihood, GaussianDistribution):
                return GaussianFactor(
                    variables=[evidence.variable],
                    mean=likelihood.mean,
                    variance=likelihood.variance
                )
            elif isinstance(likelihood, TruncatedGaussianDistribution):
                return TruncatedGaussianFactor(
                    variables=[evidence.variable],
                    mean=likelihood.mean,
                    variance=likelihood.variance,
                    lower_bound=likelihood.lower_bound,
                    upper_bound=likelihood.upper_bound
                )
                
        return None

    def _create_virtual_evidence_factor(self, evidence: Evidence) -> 'Factor':
        """Create factor for virtual evidence."""
        # Virtual evidence is similar to soft evidence but doesn't require normalization
        return self._create_soft_evidence_factor(evidence)

    def _check_likelihood_compatibility(self, 
                                     likelihood1: Union[Dict[str, float], 'Distribution'],
                                     likelihood2: Union[Dict[str, float], 'Distribution']) -> bool:
        """Check if two likelihoods are compatible."""
        if isinstance(likelihood1, dict) and isinstance(likelihood2, dict):
            # For discrete likelihoods, check support overlap
            common_states = set(likelihood1.keys()) & set(likelihood2.keys())
            return len(common_states) > 0
            
        elif isinstance(likelihood1, Distribution) and \
             isinstance(likelihood2, Distribution):
            # For continuous likelihoods, check overlap in significant regions
            if isinstance(likelihood1, TruncatedGaussianDistribution) and \
               isinstance(likelihood2, TruncatedGaussianDistribution):
                return self._check_truncated_gaussian_overlap(likelihood1, likelihood2)
            return True  # Default to compatible for other distributions
            
        return False

    def _check_truncated_gaussian_overlap(self,
                                        dist1: 'TruncatedGaussianDistribution',
                                        dist2: 'TruncatedGaussianDistribution') -> bool:
        """Check if two truncated Gaussian distributions overlap."""
        # Check if bounds overlap
        if dist1.upper_bound < dist2.lower_bound or \
           dist2.upper_bound < dist1.lower_bound:
            return False
            
        # Check if means are reasonably close
        mean_diff = abs(dist1.mean - dist2.mean)
        std1 = np.sqrt(dist1.variance)
        std2 = np.sqrt(dist2.variance)
        
        # Consider overlap significant if means are within 3 std devs
        return mean_diff < 3 * (std1 + std2)

    def _check_mixed_evidence_compatibility(self,
                                          evidence1: Evidence,
                                          evidence2: Evidence) -> bool:
        """Check compatibility between different types of evidence."""
        # Hard evidence must be compatible with soft evidence likelihood
        if evidence1.evidence_type == EvidenceType.HARD and \
           evidence2.evidence_type == EvidenceType.SOFT:
            return self._check_hard_soft_compatibility(evidence1, evidence2)
            
        elif evidence1.evidence_type == EvidenceType.SOFT and \
             evidence2.evidence_type == EvidenceType.HARD:
            return self._check_hard_soft_compatibility(evidence2, evidence1)
            
        # Virtual evidence is always compatible
        if EvidenceType.VIRTUAL in (evidence1.evidence_type, 
                                  evidence2.evidence_type):
            return True
            
        return False

    def _check_hard_soft_compatibility(self,
                                     hard_evidence: Evidence,
                                     soft_evidence: Evidence) -> bool:
        """Check if hard evidence is compatible with soft evidence."""
        if isinstance(soft_evidence.likelihood, dict):
            # For discrete variables
            return soft_evidence.likelihood.get(hard_evidence.value, 0.0) > 0
            
        elif isinstance(soft_evidence.likelihood, Distribution):
            # For continuous variables
            if isinstance(soft_evidence.likelihood, TruncatedGaussianDistribution):
                return (soft_evidence.likelihood.lower_bound <= hard_evidence.value <= 
                       soft_evidence.likelihood.upper_bound)
            return True  # Default to compatible for other distributions
            
        return False

    def get_evidence_impact(self, variable: str) -> float:
        """
        Calculate the impact of evidence on a variable.
        
        Args:
            variable: Variable to check
            
        Returns:
            float: Measure of evidence impact (0 to 1)
        """
        if variable not in self.evidence:
            return 0.0
            
        evidence = self.evidence[variable]
        
        if evidence.evidence_type == EvidenceType.HARD:
            return 1.0
            
        elif evidence.evidence_type == EvidenceType.SOFT:
            if isinstance(evidence.likelihood, dict):
                # For discrete variables, use entropy reduction
                return self._calculate_discrete_impact(evidence.likelihood)
            else:
                # For continuous variables, use variance reduction
                return self._calculate_continuous_impact(evidence.likelihood)
                
        elif evidence.evidence_type == EvidenceType.VIRTUAL:
            # Virtual evidence impact is typically less than soft evidence
            return 0.5
            
        return 0.0

    def _calculate_discrete_impact(self, likelihood: Dict[str, float]) -> float:
        """Calculate impact for discrete evidence using entropy reduction."""
        # Use normalized entropy as impact measure
        entropy = -sum(p * np.log(p) for p in likelihood.values() if p > 0)
        max_entropy = -np.log(1.0 / len(likelihood))
        return 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

    def _calculate_continuous_impact(self, likelihood: Distribution) -> float:
        """Calculate impact for continuous evidence using variance reduction."""
        if isinstance(likelihood, GaussianDistribution):
            # Use precision as impact measure
            return 1.0 - np.exp(-1.0 / likelihood.variance)
        elif isinstance(likelihood, TruncatedGaussianDistribution):
            # Consider both variance and truncation
            range_width = likelihood.upper_bound - likelihood.lower_bound
            return 1.0 - np.exp(-1.0 / (likelihood.variance * range_width))
        return 0.0

@dataclass
class ControlledEvidence(Evidence):
    """
    Evidence with control level information.
    Maintains all Evidence properties with added control awareness.
    """
    control_level: ControlLevel
    requires_coordination: bool = False
    coordination_levels: Optional[Set[ControlLevel]] = None
    authority_path: Optional[List[ControlLevel]] = None

class ControlAwareEvidencePropagator(EvidencePropagator):
    """
    Enhanced evidence propagator with locus of control awareness.
    Maintains exact evidence handling while enforcing control constraints.
    """
    def __init__(self, model: BayesianNetwork):
        super().__init__(model)
        self.control_validator = ControlValidator()
        self.variable_controls: Dict[str, ControlledVariable] = {}
        self._initialize_control_mappings()
        self.pending_coordination: Dict[str, Set[ControlLevel]] = {}

    def _initialize_control_mappings(self) -> None:
        """Initialize control mappings for all variables."""
        for node_id, node in self.model.nodes.items():
            if hasattr(node, 'control_scope'):
                self.variable_controls[node_id] = ControlledVariable(
                    name=node_id,
                    control_scope=node.control_scope,
                    validator=self.control_validator
                )

    def add_evidence(self, 
                    evidence: Union[Evidence, ControlledEvidence],
                    control_level: Optional[ControlLevel] = None) -> bool:
        """
        Add evidence with control level validation.
        
        Args:
            evidence: Evidence to add
            control_level: Control level providing evidence
            
        Returns:
            bool: Whether evidence was successfully added
            
        Raises:
            ValueError: If evidence is invalid or control level lacks authority
        """
        # Convert to ControlledEvidence if necessary
        if isinstance(evidence, Evidence) and not isinstance(evidence, ControlledEvidence):
            if control_level is None:
                raise ValueError("Control level must be provided for non-controlled evidence")
                
            evidence = self._convert_to_controlled_evidence(evidence, control_level)

        # Validate evidence
        if not self._validate_controlled_evidence(evidence):
            raise ValueError(f"Invalid evidence for variable {evidence.variable}")

        # Check control level authority
        if not self._check_evidence_authority(evidence):
            raise ValueError(
                f"Control level {evidence.control_level.name} lacks authority "
                f"for variable {evidence.variable}"
            )

        # Check if coordination is required
        if self._requires_coordination(evidence):
            self._handle_coordination_requirement(evidence)
            return False  # Evidence pending coordination

        # Add evidence with control information
        return super().add_evidence(evidence)

    def _convert_to_controlled_evidence(self,
                                      evidence: Evidence,
                                      control_level: ControlLevel) -> ControlledEvidence:
        """Convert standard evidence to controlled evidence."""
        variable = evidence.variable
        requires_coordination = False
        coordination_levels = None
        authority_path = None
        
        if variable in self.variable_controls:
            controlled_var = self.variable_controls[variable]
            requires_coordination = controlled_var.requires_coordination(control_level)
            if requires_coordination:
                coordination_levels = controlled_var.control_scope.secondary_levels
            
            primary_level = controlled_var.control_scope.primary_level
            authority_path = self.control_validator.get_influence_path(
                control_level,
                primary_level
            )

        return ControlledEvidence(
            variable=evidence.variable,
            value=evidence.value,
            evidence_type=evidence.evidence_type,
            likelihood=evidence.likelihood,
            precision=evidence.precision,
            control_level=control_level,
            requires_coordination=requires_coordination,
            coordination_levels=coordination_levels,
            authority_path=authority_path
        )

    def _validate_controlled_evidence(self, evidence: ControlledEvidence) -> bool:
        """
        Validate controlled evidence.
        Maintains all standard evidence validation plus control validation.
        """
        # First perform standard evidence validation
        if not super().validate(evidence):
            return False

        # Check if variable has control information
        if evidence.variable in self.variable_controls:
            controlled_var = self.variable_controls[evidence.variable]
            
            # Validate control level can modify variable
            if not controlled_var.can_be_modified_by(evidence.control_level):
                self.logger.warning(
                    f"Control level {evidence.control_level.name} cannot modify "
                    f"variable {evidence.variable}"
                )
                return False
                
            # Validate authority path if provided
            if evidence.authority_path:
                if not self._validate_authority_path(
                    evidence.authority_path,
                    controlled_var.control_scope.primary_level
                ):
                    return False

        return True

    def _check_evidence_authority(self, evidence: ControlledEvidence) -> bool:
        """Check if control level has authority to provide evidence."""
        if evidence.variable not in self.variable_controls:
            return True  # No control restrictions
            
        controlled_var = self.variable_controls[evidence.variable]
        return controlled_var.can_be_modified_by(evidence.control_level)

    def _requires_coordination(self, evidence: ControlledEvidence) -> bool:
        """Check if evidence requires coordination with other control levels."""
        if not evidence.requires_coordination:
            return False
            
        if evidence.variable not in self.variable_controls:
            return False
            
        controlled_var = self.variable_controls[evidence.variable]
        return controlled_var.requires_coordination(evidence.control_level)

    def _handle_coordination_requirement(self, evidence: ControlledEvidence) -> None:
        """Handle evidence requiring coordination."""
        if evidence.coordination_levels:
            self.pending_coordination[evidence.variable] = evidence.coordination_levels.copy()
            
            self.logger.info(
                f"Evidence for {evidence.variable} requires coordination with "
                f"levels: {[level.name for level in evidence.coordination_levels]}"
            )

    def coordinate_evidence(self,
                          variable: str,
                          control_level: ControlLevel,
                          approve: bool) -> None:
        """
        Coordinate evidence approval across control levels.
        
        Args:
            variable: Variable requiring coordination
            control_level: Control level providing coordination
            approve: Whether this level approves the evidence
        """
        if variable not in self.pending_coordination:
            raise ValueError(f"No pending coordination for variable {variable}")
            
        if control_level not in self.pending_coordination[variable]:
            raise ValueError(
                f"Control level {control_level.name} not required for "
                f"coordination of variable {variable}"
            )
            
        self.pending_coordination[variable].remove(control_level)
        
        if not approve:
            # Clear pending coordination if any level disapproves
            del self.pending_coordination[variable]
            self.logger.info(
                f"Evidence for {variable} rejected by {control_level.name}"
            )
        elif not self.pending_coordination[variable]:
            # All levels have approved
            self.logger.info(
                f"Evidence for {variable} approved by all required levels"
            )
            del self.pending_coordination[variable]
            # Process the evidence here

    def _validate_authority_path(self,
                               path: List[ControlLevel],
                               target_level: ControlLevel) -> bool:
        """Validate an authority path is correct."""
        if not path:
            return False
            
        if path[-1] != target_level:
            self.logger.warning(
                f"Authority path does not lead to target level {target_level.name}"
            )
            return False
            
        for i in range(len(path) - 1):
            if not self.control_validator.validate_control_path(
                path[i],
                path[i + 1],
                "influence"
            ):
                self.logger.warning(
                    f"Invalid control path segment: {path[i].name} -> {path[i+1].name}"
                )
                return False
                
        return True

    def create_evidence_factors(self) -> Dict[str, 'Factor']:
        """
        Create evidence factors with control level influence.
        Maintains exact factor creation while incorporating control weights.
        """
        evidence_factors = {}
        
        for var, evidence in self.evidence.items():
            if isinstance(evidence, ControlledEvidence):
                # Create factor with control level influence
                factor = self._create_controlled_evidence_factor(evidence)
            else:
                # Create standard evidence factor
                factor = super()._create_evidence_factor(evidence)
                
            if factor is not None:
                evidence_factors[var] = factor
                
        return evidence_factors

    def _create_controlled_evidence_factor(self,
                                         evidence: ControlledEvidence) -> Optional['Factor']:
        """
        Create evidence factor with control level influence.
        Maintains factor properties while incorporating control weights.
        """
        # Get base evidence factor
        factor = super()._create_evidence_factor(evidence)
        if factor is None:
            return None
            
        # Apply control level influence weight if applicable
        if evidence.variable in self.variable_controls:
            controlled_var = self.variable_controls[evidence.variable]
            influence_weight = controlled_var.get_influence_weight(evidence.control_level)
            
            if influence_weight < 1.0:
                # Scale factor by influence weight
                factor = self._scale_evidence_factor(factor, influence_weight)
                
        return factor