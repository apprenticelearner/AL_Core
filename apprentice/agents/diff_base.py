from abc import abstractmethod
from typing import Dict
from jsondiff import diff

from apprentice.agents.base import BaseAgent
from apprentice.working_memory.representation import Sai


class DiffBaseAgent(BaseAgent):
    prior_state = {}

    def request(self, state: Dict, **kwargs) -> Dict:
        """
        Returns a dict containing a Selection, Action, Input.

        :param state: a state represented as a dict (parsed from JSON)
        """
        d = diff(self.prior_state, state)
        self.prior_state = state
        return self.request_diff(d)

    @abstractmethod
    def request_diff(self, state_diff: Dict) -> Dict:
        """
        :param diff: a diff object that is the output of JSON diff
        """
        pass

    def train(self, state: Dict, sai: Sai, reward: float, next_state: Dict,
              **kwargs):
        """
        Accepts a JSON/Dict object representing the state,
        a JSON/Dict object representing the state after the SAI is invoked,
        a string representing the skill label,
        a list of strings representing the foas,
        a string representation the selection action and inputs,
        a reward
        """
        state_diff = diff(self.prior_state, state)
        next_state_diff = diff(state, next_state)
        self.prior_state = next_state
        return self.train_diff(state_diff, next_state_diff, sai, reward)

    @abstractmethod
    def train_diff(self, state_diff, next_state_diff, sai, reward):
        """
        Updates the state by some provided diff, then trains on the provided
        demonstration in this state.
        """
        pass


if __name__ == "__main__":
    pass
