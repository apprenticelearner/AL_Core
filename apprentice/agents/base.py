from abc import ABCMeta, abstractmethod
from typing import Dict
from typing import Collection


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
    def train(self, state: Dict, selection: str, action: str, inputs:
              Collection[str], reward: float, **kwargs):
        """
        Accepts a JSON/Dict object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        pass

    @abstractmethod
    def check(self, state: Dict, selection: str, action: str, inputs:
              Collection[str], **kwargs) -> float:
        """
        Checks the correctness (reward) of an SAI action in a given state.
        """
        pass


if __name__ == "__main__":
    pass
