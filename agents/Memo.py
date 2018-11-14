from agents.BaseAgent import BaseAgent


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
    def __init__(self, feature_set, function_set):
        self.lookup = {}

    def request(self, state):
        # print(state)
        state = freeze(state)
        resp = self.lookup.get(state, None)

        if resp is None:
            return {}

        return {'skill_label': resp[0],
                'selection': resp[1],
                'action': resp[2],
                'inputs': resp[3],
                'foci_of_attention': resp[4]}

    def train(self, state, selection, action, inputs, reward,
              skill_label="NO_LABEL", foci_of_attention=None):
        state = freeze(state)
        resp = self.lookup.get(state, None)
        if ((resp is None and reward > 0) or
           (resp is not None and reward >= resp[5])):
            self.lookup[state] = (skill_label, selection, action, inputs,
                                  foci_of_attention, reward)

        if (resp is not None and reward < 0 and skill_label == resp[0] and
                selection == resp[1] and action == resp[2] and inputs ==
                resp[3] and foci_of_attention == resp[4]):
            del self.lookup[state]
