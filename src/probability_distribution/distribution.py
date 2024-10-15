# src/probability_distribution/distribution.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Distribution(ABC):
    def __init__(self, variable: str, parents: List[str]):
        self.variable = variable
        self.parents = parents

    @abstractmethod
    def get_probability(self, value: Any, parent_values: Dict[str, Any]) -> float:
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def sample(self, parent_values: Dict[str, Any]) -> Any:
        pass