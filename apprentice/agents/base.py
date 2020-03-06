from abc import ABCMeta
from abc import abstractmethod
from typing import Dict

from apprentice.working_memory.representation import Sai


class BaseAgent(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        """
        Creates an agent with the provided skills.
        """
        pass

    @abstractmethod
    def request(self, state: Dict, **kwargs) -> Dict:
        """
        Returns a dict containing a Selection, Action, Input.

        :param state: a state represented as a dict (parsed from JSON)
        """
        pass

    @abstractmethod
    def train(self, state: Dict, sai: Sai, reward: float, **kwargs):
        """
        Accepts a JSON/Dict object representing the state,
        a JSON/Dict object representing the state after the SAI is invoked,
        a string representing the skill label,
        a list of strings representing the foas,
        a string representation the selection action and inputs,
        a reward
        """
        pass

    @abstractmethod
    def check(self, state: Dict, sai: Sai, **kwargs) -> float:
        """
        Checks the correctness (reward) of an SAI action in a given state.
        """
        pass


if __name__ == "__main__":
    pass
