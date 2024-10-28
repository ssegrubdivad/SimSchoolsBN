# src/inference_engine/evidence_propagation.py

from typing import Dict, Set, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import logging
from enum import Enum

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