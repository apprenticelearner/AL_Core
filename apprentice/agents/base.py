from abc import ABCMeta, abstractmethod
from typing import Dict

from jsondiff import diff


class BaseAgent(metaclass=ABCMeta):
    prior_state = None

    def __init__(self, prior_skills: Collection[Skills]):
        """
        Creates an agent with the provided skills.
        """
        pass

    def request(self, state: Dict) -> Dict:
        """
        Returns a dict containing a Selection, Action, Input.

        :param state: a state represented as a dict (parsed from JSON)
        """
        return self.request_diff(diff(state, self.prior_state))

    @abstractmethod
    def request_diff(self, state_diff: Dict) -> Dict:
        """
        :param diff: a diff object that is the output of JSON diff
        """
        pass

    def train(self, state: Dict, selection: str, action: str, inputs
              Collection[str], reward: float, skill_label: str,
              foci_of_attention: Collection[str]):
        """
        Accepts a JSON/Dict object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        return self.train_diff(diff(state, self.prior_state), selection,
                               action, inputs, reward, skill_label,
                               foci_of_attention)

    @abstractmethod
    def train_diff(self, state_diff, selection, action, inputs, reward,
                   skill_label, foci_of_attention):
        """
        Updates the state by some provided diff, then trains on the provided
        demonstration in this state.
        """
        pass

    @abstractmethod
    def train_last_state(self, selection, action, inputs, reward, skill_label,
                         foci_of_attention):
        """
        Trains on the provided demonstration in the last / current state.
        """
        pass


if __name__ == "__main__":
    pass
