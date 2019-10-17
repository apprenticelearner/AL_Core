from abc import ABCMeta, abstractmethod
from typing import Mapping, Collection

from apprentice.working_memory.representation import Activation


class WhenLearner(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, state: Mapping[str, any], action: Activation) -> float:
        ...

    @abstractmethod
    def update(self, state: dict, action: Activation, reward: float, next_state: dict,
               next_actions: Collection[Activation]) -> None:
        ...
