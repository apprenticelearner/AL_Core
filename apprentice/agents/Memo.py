from typing import Dict

from apprentice.agents.base import BaseAgent
from apprentice.working_memory.representation import Sai


def freeze(obj):
    """Freeze a state (dict), for memoizing."""
    if isinstance(obj, dict):
        return frozenset({k: freeze(v) for k, v in obj.items()}.items())
    if isinstance(obj, list):
        return tuple([freeze(v) for v in obj])
    return obj


class Memo(BaseAgent):
    """
    Memorizes the state actions pairs and responds with the highest reward,
    demonstrated action for a given request.

    Made for testing the API.
    """
    def __init__(self, **kwargs):
        self.lookup = {}

    def request(self, state: Dict, **kwargs) -> Dict:
        # print(state)
        state = freeze(state)
        resp = self.lookup.get(state, None)

        if resp is None:
            return {}

        return {'skill_label': resp[0],
                'selection': resp[1].selection,
                'action': resp[1].action,
                'inputs': resp[1].inputs}

    def train(self, state: Dict, sai: Sai, reward: float, **kwargs):
        state = freeze(state)
        resp = self.lookup.get(state, None)
        if ((resp is None and reward > 0) or
           (resp is not None and reward >= resp[2])):
            skill_label = kwargs.get('skill_label',"")
            self.lookup[state] = (skill_label, sai, reward)

        if (resp is not None and reward < 0 and skill_label == resp[0] and
           sai == resp[1]):
            del self.lookup[state]

    def check(self, state: Dict, sai: Sai, **kwargs) -> float:
        state = freeze(state)
        resp = self.lookip.get(state, None)

        if resp is None:
            return 0.0
        else:
            return resp[2]
