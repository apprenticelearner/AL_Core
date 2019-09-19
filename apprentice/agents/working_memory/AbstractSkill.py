from abc import ABC, abstractmethod
from typing import Collection, Dict, Callable

from apprentice.agents.working_memory import AbstractCondition


class AbstractSkill(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def conditions(self) -> Collection[AbstractCondition]:
        pass

    @abstractmethod
    def function(self, state: Dict) -> Callable:
        pass