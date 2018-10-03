from agents.BaseAgent import BaseAgent


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

        resp = self.lookup.get(state, None)

        if resp is None and resp[5] > 0 or reward >= resp[5]:
            self.lookup[state] = (skill_label, selection, action, inputs,
                                  foci_of_attention, reward)
